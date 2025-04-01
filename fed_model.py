from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import requests
import os
from threading import Lock
import time
import json

app = Flask(__name__)

# Read configuration from config.json
config_path = os.path.join(os.path.dirname(__file__), "config.json")
try:
    with open(config_path, "r") as f:
        config_data = json.load(f)
    local_port = config_data.get("localPort", "5000")
    node_id = int(config_data.get("nodeId", "1"))
    num_nodes = len(config_data.get("nodes", []))
    print(f"[INFO] Loaded config: port {local_port}, nodeID {node_id}, nodes {num_nodes}")
except Exception as e:
    print(f"[WARNING] Could not load config file: {e}")
    local_port = "5000"
    node_id = 1
    num_nodes = 2

# Suzuki-Kasami variables
RN = {i+1: 0 for i in range(num_nodes)}  # Request numbers for all nodes
TOKEN = {
    "has_token": (node_id == 1),  # Node 1 starts with token
    "queue": [],
    "LN": {i+1: 0 for i in range(num_nodes)} if (node_id == 1) else {}
}
received_weights = []

def load_preprocess():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    samples_per_node = len(x_train) // num_nodes
    start_idx = (node_id - 1) * samples_per_node
    end_idx = node_id * samples_per_node
    return x_train[start_idx:end_idx], y_train[start_idx:end_idx], x_test, y_test

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

x_train, y_train, x_test, y_test = load_preprocess()
model = create_model()

def send_with_retry(url, payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return None

def request_token_from_node(target_node, from_node_id, rn_value):
    try:
        payload = {"type": "token_request", "from_node": from_node_id, "rn": rn_value}
        requests.post(f"http://{target_node}/token_message", json=payload, timeout=5)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to request token from {target_node}: {e}")

def request_critical_section():
    global RN
    RN[node_id] += 1
    print(f"[SUZUKI] Node {node_id} requesting CS (RN={RN[node_id]})")
    for node in config_data.get("nodes", []):
        if node != config_data.get("myAddress", ""):
            request_token_from_node(node, node_id, RN[node_id])
    while not TOKEN["has_token"]:
        time.sleep(0.1)

def release_critical_section():
    global TOKEN
    print(f"[SUZUKI] Node {node_id} releasing CS")
    while TOKEN["queue"]:
        next_node = TOKEN["queue"].pop(0)
        send_token_to_node(next_node)

def send_token_to_node(target_node):
    TOKEN["has_token"] = False
    payload = {"type": "token", "token": {"LN": TOKEN["LN"], "queue": TOKEN["queue"]}}
    try:
        requests.post(f"http://{target_node}/token_message", json=payload, timeout=5)
        print(f"[SUZUKI] Sent token to {target_node}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Token send failed: {e}")
        TOKEN["has_token"] = True

@app.route('/token_message', methods=['POST'])
def token_message():
    data = request.json
    if data["type"] == "token_request":
        from_node = data["from_node"]
        RN[from_node] = max(RN[from_node], data["rn"])
        if TOKEN["has_token"] and (RN[from_node] > TOKEN["LN"].get(from_node, 0)):
            send_token_to_node(from_node)
    elif data["type"] == "token":
        TOKEN["has_token"] = True
        TOKEN["LN"] = data["token"]["LN"]
        TOKEN["queue"] = data["token"]["queue"]
        print("[SUZUKI] Token received")
    return jsonify({"status": "processed"})

@app.route('/train_local', methods=['GET'])
def train_local():
    # Local training
    model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=1)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    weights = [w.tolist() for w in model.get_weights()]
    
    # Use Suzuki-Kasami to send weights
    request_critical_section()
    leader_addr = request.args.get('leader', '')
    if not leader_addr:
        return jsonify({"error": "Leader address missing"}), 400
    
    leader_ip = leader_addr.split(':')[0]
    leader_python_url = f"http://{leader_ip}:6002/send_model"
    response = send_with_retry(leader_python_url, {"weights": weights})
    release_critical_section()
    
    if response:
        print(f"[INFO] Weights sent to leader. Test accuracy: {test_accuracy:.4f}")
    else:
        print("[ERROR] Weight send failed after retries")
    
    return jsonify({"message": "Training done", "test_accuracy": float(test_accuracy)})

@app.route('/send_model', methods=['POST'])
def receive_model_for_aggregation():
    received_weights.append(request.json['weights'])
    print(f"[INFO] Received weights ({len(received_weights)}/{num_nodes})")
    return jsonify({"message": "Weights stored"})

@app.route('/aggregate_models', methods=['GET'])
def aggregate_models():
    request_critical_section()
    if len(received_weights) != num_nodes:
        release_critical_section()
        return jsonify({"error": "Not all weights received"}), 400
    
    avg_weights = [np.mean(layer_weights, axis=0) for layer_weights in zip(*[
        [np.array(w) for w in weights] for weights in received_weights
    ])]
    model.set_weights(avg_weights)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    received_weights.clear()
    
    # Distribute aggregated model
    serialized = [w.tolist() for w in avg_weights]
    response = send_with_retry(f"http://localhost:{local_port}/distributeModel", {"weights": serialized})
    release_critical_section()
    
    return jsonify({"message": "Aggregation done", "accuracy": float(test_acc)})

@app.route('/update_model', methods=['POST'])
def update_model():
    weights = [np.array(w) for w in request.json['weights']]
    model.set_weights(weights)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return jsonify({"accuracy": float(test_acc)})

if __name__ == '__main__':
    print(f"[INFO] Starting node {node_id} on port 6002")
    app.run(host="0.0.0.0", port=6002)
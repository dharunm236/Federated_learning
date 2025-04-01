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
    node_id = int(config_data.get("nodeID", "1"))
    print(f"[INFO] Loaded local port and node_ID from config: {local_port}")
except Exception as e:
    print(f"[WARNING] Could not load config file: {e}")
    local_port = "5000"  # Default fallback
    node_id = 1
    print(f"[INFO] Using default local port: {local_port}")

# Configuration
try:
    num_nodes = len(config_data.get("nodes", []))
    print(f"[INFO] Number of nodes from config: {num_nodes}")
except:
    num_nodes = 2  # Default fallback
    print(f"[INFO] Using default number of nodes: {num_nodes}")
weights_lock = Lock()
received_weights = []

# Add these global variables for storing prepared state
prepared_weights = None
model_prepared = False

def load_preprocess():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Partition data for federated learning
    samples_per_node = len(x_train) // num_nodes
    start_idx = (node_id - 1) * samples_per_node
    end_idx = node_id * samples_per_node
    
    local_x_train = x_train[start_idx:end_idx]
    local_y_train = y_train[start_idx:end_idx]
    
    return local_x_train, local_y_train, x_test, y_test

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Load data and create model
x_train, y_train, x_test, y_test = load_preprocess()
model = create_model()

def send_with_retry(url, payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    return None

@app.route('/train_local', methods=['GET'])
def train_local():
    # Train the model for one local epoch
    batch_size = 64
    model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    # Get model weights and convert to serializable format
    weights = model.get_weights()
    serializable_weights = [w.tolist() for w in weights]
    
    # Send weights to leader node via Go server
    payload = {"weights": serializable_weights}
    response = send_with_retry(f"http://localhost:{local_port}/receiveModelParam", payload)
    if response:
        print(f"[INFO] Sent model weights to leader. Test accuracy: {test_accuracy:.4f}")
    else:
        print("[ERROR] Failed to send weights after retries")
    
    return jsonify({
        "message": "Local training completed",
        "test_accuracy": float(test_accuracy)
    })

@app.route('/aggregate_models', methods=['GET'])
def aggregate_models():
    with weights_lock:
        if len(received_weights) == 0:
            return jsonify({"error": "No weights received yet"}), 400
        
        # Perform aggregation
        weights_list = [[np.array(w) for w in weights] for weights in received_weights]
        avg_weights = [np.mean(weights_per_layer, axis=0) for weights_per_layer in zip(*weights_list)]
        
        # Update the model with new weights
        model.set_weights(avg_weights)
        
        # Add the requested message after aggregation
        print("[INFO] Weights aggregated successfully")
        
        # Evaluate the aggregated model
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"[INFO] Federated model aggregated. Test accuracy: {test_accuracy:.4f}")
        
        # Convert aggregated weights to serializable format for distribution
        serializable_weights = [w.tolist() for w in avg_weights]
        
        # Reset received weights for next round
        received_weights.clear()
        
        # Send the aggregated model back to the Go server for distribution
        payload = {"weights": serializable_weights}
        response = send_with_retry(f"http://localhost:{local_port}/distributeModel", payload)
        if response:
            print("[INFO] Sent aggregated model to Go server for distribution")
        else:
            print("[ERROR] Failed to send aggregated model after retries")
        
        return jsonify({
            "message": "Models aggregated successfully",
            "accuracy": float(test_accuracy)
        })

@app.route('/update_model', methods=['POST'])
def update_model():
    data = request.json
    weights = [np.array(w) for w in data['weights']]
    
    model.set_weights(weights)
    print("[INFO] Updated model with aggregated parameters")
    
    # Evaluate the updated model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"[INFO] Updated model test accuracy: {test_accuracy:.4f}")

    return jsonify({"message": "Model updated successfully", "accuracy": float(test_accuracy)})

@app.route('/send_model', methods=['POST'])
def receive_model_for_aggregation():
    data = request.json
    with weights_lock:
        received_weights.append(data['weights'])
        print(f"[INFO] Received model weights from a node. Total received: {len(received_weights)}/{num_nodes}")
    
    return jsonify({"message": "Model received successfully"})

@app.route('/prepare_model_update', methods=['POST'])
def prepare_model_update():
    global prepared_weights, model_prepared
    
    try:
        data = request.json
        prepared_weights = [np.array(w) for w in data['weights']]
        
        # Validate that we can use these weights (basic check)
        temp_model = create_model()
        temp_model.set_weights(prepared_weights)
        
        # Test that the weights are valid by running a small evaluation
        temp_model.evaluate(x_test[:10], y_test[:10], verbose=0)
        
        model_prepared = True
        print("[INFO] Successfully prepared model update")
        return jsonify({"message": "Model update prepared"}), 200
    except Exception as e:
        model_prepared = False
        print(f"[ERROR] Failed to prepare model update: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/commit_model_update', methods=['GET'])
def commit_model_update():
    global model_prepared, prepared_weights
    
    if not model_prepared or prepared_weights is None:
        return jsonify({"error": "No prepared model update to commit"}), 400
    
    try:
        # Apply the prepared weights to the actual model
        model.set_weights(prepared_weights)
        
        # Evaluate the updated model
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"[INFO] Committed model update. Test accuracy: {test_accuracy:.4f}")
        
        # Reset preparation state
        model_prepared = False
        return jsonify({"message": "Model update committed", "accuracy": float(test_accuracy)}), 200
    except Exception as e:
        print(f"[ERROR] Failed to commit model update: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/abort_model_update', methods=['GET'])
def abort_model_update():
    global model_prepared, prepared_weights
    
    model_prepared = False
    prepared_weights = None
    print("[INFO] Model update aborted")
    
    return jsonify({"message": "Model update aborted"}), 200

if __name__ == '__main__':
    print("[INFO] Starting federated learning server...")
    print(f"[INFO] Local dataset size: {len(x_train)} samples")
    app.run(host="0.0.0.0", port=6002)
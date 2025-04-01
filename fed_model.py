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
    local_port = config_data.get("pythonPort", "6002")  # Use the pythonPort from config
    node_id = int(config_data.get("nodeId", "1"))  # Use the nodeId from config
    my_address = config_data.get("myAddress", "")
    all_nodes = config_data.get("nodes", [])
    go_port = my_address.split(':')[1] if ':' in my_address else "5000" #add go port
    print(f"[INFO] Loaded config: node_id={node_id}, local_port={local_port}, my_address={my_address}")
except Exception as e:
    print(f"[WARNING] Could not load config file: {e}")
    local_port = "6002"  # Default fallback
    node_id = 1
    my_address = ""
    all_nodes = []
    print(f"[INFO] Using default configuration")

# Configuration
try:
    num_nodes = len(all_nodes)
    print(f"[INFO] Number of nodes from config: {num_nodes}")
except:
    num_nodes = 2  # Default fallback
    print(f"[INFO] Using default number of nodes: {num_nodes}")

weights_lock = Lock()
received_weights = []

# ----------------------------------
# Suzuki-Kasami mutual exclusion setup
# ----------------------------------

# Global request sequence number for each node
RN = {i+1: 0 for i in range(num_nodes)}  # request number array: { nodeId: int }

# Global token
TOKEN = {
    "has_token": node_id == 1,  # Node 1 starts with the token
    "queue": [],
    "LN": {i+1: 0 for i in range(num_nodes)}  # last request number used: { nodeId: int }
}

print(f"[INFO] Initial token status: Node {node_id} has token: {TOKEN['has_token']}")

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
    # ... existing model creation code ...
    # Keep the model architecture unchanged
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

# Helper to request the token from another node
def request_token_from_node(target_node, from_node_id, rn_value):
    try:
        node_ip = target_node.split(':')[0] 
        payload = {
            "type": "token_request",
            "from_node": from_node_id,
            "rn": rn_value
        }
        requests.post(f"http://{node_ip}:6002/token_message", json=payload, timeout=5)
        print(f"[INFO] Requested token from node at {node_ip}:6002")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to request token from node {target_node}: {e}")

# Function to request critical section
def request_critical_section():
    global RN, TOKEN
    
    # If we already have the token, no need to request
    if TOKEN["has_token"]:
        print(f"[INFO] Node {node_id} already has the token, entering CS")
        return
        
    # Increment own request number
    RN[node_id] = RN.get(node_id, 0) + 1
    print(f"[INFO] Node {node_id} requesting CS with RN={RN[node_id]}")

    # Broadcast request to all other nodes
    for target in all_nodes:
        if target != my_address:
            # Extract the IP from the address (remove port)
            request_token_from_node(target, node_id, RN[node_id])

    # Wait until we have the token
    wait_start = time.time()
    while not TOKEN["has_token"]:
        time.sleep(0.1)
        # Add timeout to prevent infinite waiting
        if time.time() - wait_start > 30:  # 30-second timeout
            print("[WARN] Token request timed out, continuing without token")
            break
    
    print(f"[INFO] Node {node_id} entered critical section")

# Function to release critical section
def release_critical_section():
    global TOKEN, RN
    
    print(f"[INFO] Node {node_id} releasing CS")
    # Update the LN entry for current node
    TOKEN["LN"][node_id] = RN[node_id]
    
    # Check if any node should get the token
    for i in range(1, num_nodes + 1):
        if i != node_id and RN.get(i, 0) > TOKEN["LN"].get(i, 0):
            if i not in TOKEN["queue"]:
                TOKEN["queue"].append(i)
    
    # Pass the token to next node in queue if any
    if TOKEN["queue"]:
        next_node = TOKEN["queue"].pop(0)
        send_token_to_node(next_node)
    else:
        print(f"[INFO] No pending requests, keeping the token")

# Function to pass the token to another node
def send_token_to_node(target_node_id):
    global TOKEN
    
    # Find the target node's address
    target_address = None
    for node in all_nodes:
        # Simple way to match node with ID - this might need adjustment
        if all_nodes.index(node) + 1 == target_node_id:
            target_address = node
            break
            
    if not target_address:
        print(f"[ERROR] Could not find address for node {target_node_id}")
        return
        
    TOKEN["has_token"] = False
    payload = {
        "type": "token",
        "token": {
            "LN": TOKEN["LN"],
            "queue": TOKEN["queue"]
        }
    }
    
    try:
        node_ip = target_address.split(':')[0]
        requests.post(f"http://{node_ip}:6002/token_message", json=payload, timeout=5)
        print(f"[INFO] Sent token to node {target_node_id} at {node_ip}:6002")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to send token to node {target_node_id}: {e}")
        # If sending fails, token remains locally
        TOKEN["has_token"] = True

# Endpoint to handle incoming token or token_requests
@app.route('/token_message', methods=['POST'])
def token_message():
    global TOKEN, RN
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400

    msg_type = data.get("type")
    if msg_type == "token_request":
        from_node = data["from_node"]
        rn_val = data["rn"]
        
        # Update RN
        RN[from_node] = max(RN.get(from_node, 0), rn_val)
        print(f"[INFO] Received token request from node {from_node} with RN={rn_val}")
        
        # If we have the token and not in CS, decide if token can be sent
        if TOKEN["has_token"]:
            # If the requesting node needs the token based on RN > LN
            if RN.get(from_node, 0) > TOKEN["LN"].get(from_node, 0):
                send_token_to_node(from_node)
            else:
                print(f"[INFO] Node {from_node} doesn't need the token yet")

    elif msg_type == "token":
        # We received the token
        TOKEN["has_token"] = True
        TOKEN["LN"] = data["token"]["LN"]
        TOKEN["queue"] = data["token"]["queue"]
        print("[INFO] Token received")

    return jsonify({"message": "Token message processed"})

# Decide if it is safe to send the token
def safe_to_send_token(target_node):
    # If the target node's RN is greater than TOKEN["LN"], it needs the token
    return RN.get(target_node, 0) > TOKEN["LN"].get(target_node, 0)

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
    
    # Use Suzuki-Kasami to enter critical section before updating shared weights
    print(f"[INFO] Node {node_id} requesting CS to update shared weights")
    request_critical_section()
    
    # Add weights to the shared list
    with weights_lock:
        received_weights.append(serializable_weights)
        print(f"[INFO] Added weights to shared list. Total: {len(received_weights)}/{num_nodes}")
    
    # Release critical section
    release_critical_section()
    
    # Notify the Go server that training is complete
    try:
        payload = {"node_id": node_id, "complete": True}
        requests.post(f"http://localhost:{go_port}/notifyTrainingComplete", json=payload)
        print("[INFO] Notified Go server that training is complete")
    except Exception as e:
        print(f"[ERROR] Failed to notify Go server: {e}")
    
    return jsonify({
        "message": "Local training completed",
        "test_accuracy": float(test_accuracy)
    })

@app.route('/aggregate_models', methods=['GET','POST'])
def aggregate_models():
    print("[INFO] Starting model aggregation")
    global received_weights
    
    # Enter critical section to read the shared weights
    request_critical_section()
    
    weights_to_aggregate = []
    
    # Check if we've received weights from the request
    if request.method == 'POST' and request.json:
        try:
            # Try to get weights from the request body
            weights_from_request = request.json.get('weights_list')
            if weights_from_request and isinstance(weights_from_request, list):
                print(f"[INFO] Using {len(weights_from_request)} weights from request")
                
                # Process each weight entry
                for weight_entry in weights_from_request:
                    # Extract actual weights array from the weight entry
                    if isinstance(weight_entry, dict) and 'weights' in weight_entry:
                        # If it's in {'weights': [...]} format (from Go client)e
                        nested_weights = weight_entry.get('weights')
                        if nested_weights:
                            try:
                                # Convert to proper numeric arrays
                                numeric_weights = [np.array(w, dtype=np.float32) for w in nested_weights]
                                weights_to_aggregate.append(numeric_weights)
                            except (ValueError, TypeError) as e:
                                print(f"[ERROR] Could not convert weights to numeric arrays: {e}")
                    else:
                        # If it's already a list (direct from Python node)
                        try:
                            numeric_weights = [np.array(w, dtype=np.float32) for w in weight_entry]
                            weights_to_aggregate.append(numeric_weights)
                        except (ValueError, TypeError) as e:
                            print(f"[ERROR] Could not convert weights to numeric arrays: {e}")
                            
                print(f"[INFO] Successfully processed {len(weights_to_aggregate)} valid weight sets")
            
            # If no valid weights were found in the request, use the local received_weights
            if not weights_to_aggregate and received_weights:
                print(f"[INFO] Using {len(received_weights)} weights from shared list")
                weights_to_aggregate = [[np.array(w, dtype=np.float32) for w in weights] for weights in received_weights]
                
        except Exception as e:
            print(f"[ERROR] Failed to parse request weights: {e}")
            # Fall back to using the received_weights if available
            if received_weights:
                weights_to_aggregate = [[np.array(w, dtype=np.float32) for w in weights] for weights in received_weights]
    else:
        # Use the existing received_weights
        if received_weights:
            weights_to_aggregate = [[np.array(w, dtype=np.float32) for w in weights] for weights in received_weights]
    
    if not weights_to_aggregate:
        # Release CS if no weights
        release_critical_section()
        print("[ERROR] No weights available for aggregation")
        return jsonify({"error": "No weights received yet"}), 400

    # Perform aggregation in critical section
    try:
        # Make sure all weight sets have the same structure
        first_shape = [w.shape for w in weights_to_aggregate[0]]
        valid_weights = [weights for weights in weights_to_aggregate 
                         if len(weights) == len(first_shape) and all(w.shape == s for w, s in zip(weights, first_shape))]
        
        if not valid_weights:
            release_critical_section()
            print("[ERROR] No compatible weight structures found for aggregation")
            return jsonify({"error": "Weight structures are incompatible"}), 400
            
        # Aggregate the weights
        avg_weights = []
        for i in range(len(valid_weights[0])):
            layer_weights = [weights[i] for weights in valid_weights]
            avg_weights.append(np.mean(layer_weights, axis=0))
        
        print(f"[INFO] Successfully aggregated {len(valid_weights)} models")
        
        # Update model with aggregated weights
        model.set_weights(avg_weights)
        print("[INFO] Weights aggregated successfully")
        
        # Test the aggregated model
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"[INFO] Federated model aggregated. Test accuracy: {test_accuracy:.4f}")

        # Convert for distribution
        serializable_weights = [w.tolist() for w in avg_weights]
        
        # Clear the received weights after aggregation
        received_weights.clear()
        
        # Release critical section
        release_critical_section()
        
        # Send aggregated model to Go server for distribution
        payload = {"weights": serializable_weights}
        try:
            # Extract port from my_address or use fixed Go port
            response = requests.post(f"http://localhost:{go_port}/distributeModel", json=payload)
            print(f"[INFO] Sent aggregated model to Go server at port {go_port}")
        except Exception as e:
            print(f"[ERROR] Failed to send aggregated model: {e}")
            return jsonify({"error": "Failed to distribute model"}), 500

        return jsonify({
            "message": "Models aggregated successfully",
            "accuracy": float(test_accuracy)
        })
        
    except Exception as e:
        release_critical_section()
        print(f"[ERROR] Error during aggregation: {e}")
        return jsonify({"error": f"Aggregation failed: {str(e)}"}), 500

@app.route('/update_model', methods=['POST'])
def update_model():
    print("update model called")
    data = request.json
    weights = [np.array(w) for w in data['weights']]
    
    model.set_weights(weights)
    print("[INFO] Updated model with aggregated parameters")
    
    # Evaluate the updated model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"[INFO] Updated model test accuracy: {test_accuracy:.4f}")

    return jsonify({"message": "Model updated successfully", "accuracy": float(test_accuracy)})

# This endpoint is no longer used as nodes directly update shared weights
@app.route('/send_model', methods=['POST'])
def receive_model_for_aggregation():
    data = request.json
    print("[INFO] Received model data via legacy endpoint")
    return jsonify({"message": "Received, but using shared weights approach"})

@app.route('/get_model', methods=['GET'])
def get_model():
    # Return the model weights in JSON form with proper numeric format
    weights = model.get_weights()
    # Ensure all weights are proper numeric values, not strings
    serializable_weights = []
    for w in weights:
        if isinstance(w, np.ndarray):
            serializable_weights.append(w.tolist())
        else:
            print(f"[WARN] Non-ndarray weight found: {type(w)}")
            serializable_weights.append(w)
    
    print(f"[DEBUG] Returning {len(serializable_weights)} weight arrays")
    return jsonify({"weights": serializable_weights})

if __name__ == '__main__':
    print("[INFO] Starting federated learning server...")
    print(f"[INFO] Local dataset size: {len(x_train)} samples")
    print(f"[INFO] Node {node_id} has token: {TOKEN['has_token']}")
    app.run(host="0.0.0.0", port=int(local_port))
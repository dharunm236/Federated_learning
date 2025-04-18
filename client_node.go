package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

var nodes []string
var myAddress string // Will be read from config.json
var leader string
var mu sync.Mutex
var fedRound = 0
var maxRounds = 10

// Add these variables to track training status
var trainingComplete = make(map[string]bool)
var nodeResponses = 0
var trainingMutex sync.Mutex

// Add these global variables for 2PC
var modelUpdateConfirmations = make(map[string]bool)
var twoPhaseCommitMutex sync.Mutex

// Update the struct definition for proper JSON handling
type ModelParams struct {
	Weights json.RawMessage `json:"weights"`
}

type AggregatedModel struct {
	Weights json.RawMessage `json:"weights"`
}

// Add these types for the configuration
type Config struct {
    MyAddress string   `json:"myAddress"`
    Nodes     []string `json:"nodes"`
}

func loadConfig() {
    // Read config file
    configFile, err := ioutil.ReadFile("config.json")
    if err != nil {
        log.Fatalf("[ERROR] Failed to read config file: %v", err)
    }

    // Parse configuration
    var config Config
    err = json.Unmarshal(configFile, &config)
    if err != nil {
        log.Fatalf("[ERROR] Failed to parse config file: %v", err)
    }

    // Set global variables from config
    myAddress = config.MyAddress
    nodes = config.Nodes

    fmt.Printf("[INFO] Loaded configuration - MyAddress: %s, Nodes: %v\n", myAddress, nodes)
}

func startElection() {
	fmt.Println("Starting election...")
	mu.Lock()
	leader = ""
	mu.Unlock()

	higherIp := []string{}
	for _, node := range nodes {
		if node > myAddress {
			higherIp = append(higherIp, node)
		}
	}

	if len(higherIp) == 0 {
		declareLeader()
		return
	}

	// Wait for responses from higher-priority nodes
	responseChan := make(chan bool, len(higherIp))
	for _, node := range higherIp {
		go func(node string) {
			res, err := http.Get("http://" + node + "/election")
			if err == nil && res.StatusCode == http.StatusOK {
				responseChan <- true
			} else {
				responseChan <- false
			}
		}(node)
	}

	// Wait for responses or timeout
	timeout := time.After(5 * time.Second)
	responsesReceived := 0
	higherNodeResponded := false

	for responsesReceived < len(higherIp) {
		select {
		case response := <-responseChan:
			responsesReceived++
			if response {
				higherNodeResponded = true
			}
		case <-timeout:
			fmt.Println("Election timeout, no response from higher nodes")
			break
		}
	}

	if !higherNodeResponded {
		declareLeader()
	}
}

func declareLeader() {
	mu.Lock()
	leader = myAddress
	mu.Unlock()
	fmt.Println("The new leader is:", myAddress)

	for _, node := range nodes {
		if node != myAddress {
			_, err := http.Get("http://" + node + "/leader?address=" + myAddress)
			if err != nil {
				fmt.Printf("[ERROR] Failed to notify node %s about new leader: %v\n", node, err)
			}
		}
	}
}

func electionHandler(w http.ResponseWriter, r *http.Request) {
	go startElection()
	w.WriteHeader(http.StatusOK)
}

func leaderHandler(w http.ResponseWriter, r *http.Request) {
	newLeader := r.URL.Query().Get("address")
	mu.Lock()
	leader = newLeader
	mu.Unlock()
	fmt.Println("New leader elected:", newLeader)
	w.WriteHeader(http.StatusOK)
}

func monitorLeader() {
	for {
		time.Sleep(5 * time.Second)

		// Only start election if we don't have a leader
		mu.Lock()
		if leader == "" {
			mu.Unlock()
			fmt.Println("[WARN] No leader assigned, starting election...")
			startElection()
			continue
		}
		mu.Unlock()

		// Check if leader is alive, with retry logic
		isAlive := false
		for attempts := 0; attempts < 3; attempts++ {
			if isLeaderAlive() {
				isAlive = true
				break
			}
			time.Sleep(500 * time.Millisecond) // Small delay between retries
		}

		if !isAlive {
			fmt.Printf("[WARN] Leader %s appears to be down after multiple attempts, starting new election...\n", leader)
			startElection()
		}
	}
}

func isLeaderAlive() bool {
	client := http.Client{
		Timeout: 2 * time.Second, // Shorter timeout for responsiveness
	}
	res, err := client.Get("http://" + leader + "/ping")
	if err != nil {
		fmt.Printf("[DEBUG] Leader ping failed: %v\n", err)
		return false
	}
	defer res.Body.Close()
	return res.StatusCode == http.StatusOK
}

func pingHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func receiveModelParam(w http.ResponseWriter, r *http.Request) {
	body, _ := ioutil.ReadAll(r.Body)
	var params ModelParams
	json.Unmarshal(body, &params)

	if myAddress == leader {
		fmt.Println("[INFO] Leader received model weights for aggregation")
		sendToPython(params)
	} else {
		fmt.Println("[INFO] Forwarding model weights to the leader")
		forwardToLeader(params)
	}
	w.WriteHeader((http.StatusOK))
	fmt.Fprintf(w, `{"message": "Model parameters received"}`)
}

func forwardToLeader(params ModelParams) {
	data, _ := json.Marshal(params)
	client := http.Client{
		Timeout: 120 * time.Second,
	}
	res, err := client.Post("http://"+leader+"/receiveModelParam", "application/json", bytes.NewBuffer(data))
	if err != nil {
		fmt.Println("[ERROR] Failed to forward parameters to leader:", err)
		// Consider triggering a new election if the leader is unreachable
		if strings.Contains(err.Error(), "connection refused") {
			fmt.Println("[WARN] Leader appears to be down, starting new election")
			go startElection()
		}
	} else {
		fmt.Println("[INFO] Forwarded model parameters to leader:", res.Status)
		res.Body.Close()
	}
}

func sendToPython(params ModelParams) {
	data, _ := json.Marshal(params)
	res, err := http.Post("http://localhost:6002/send_model", "application/json", bytes.NewBuffer(data))
	if err != nil {
		fmt.Println("[ERROR] Failed to send the model parameters to Python service:", err)
	} else {
		fmt.Println("[INFO] Sent the model params to the Python service:", res.Status)
	}
}

func startTrainingHandler(w http.ResponseWriter, r *http.Request) {
	// Trigger local training on the Python service
	fmt.Printf("[INFO] Node %s starting local training\n", myAddress)

	res, err := http.Get("http://localhost:6002/train_local")
	if err != nil {
		fmt.Println("[ERROR] Failed to start local training:", err)
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintf(w, `{"error": "Failed to start training"}`)
		return
	}

	// Read and relay response from Python service
	body, _ := ioutil.ReadAll(res.Body)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(body)

	// Notify the leader that training is complete
	if leader != myAddress {
		notifyTrainingComplete()
	} else {
		// If I'm the leader, record my own completion
		trainingMutex.Lock()
		trainingComplete[myAddress] = true
		nodeResponses++
		fmt.Printf("[INFO] Leader completed training. Waiting for %d more nodes...\n", len(nodes)-nodeResponses)

		// Add a check here so the leader triggers aggregation if it’s the last to finish
		if nodeResponses == len(nodes) {
			fmt.Println("[INFO] All nodes are done. Triggering aggregation...")
			go startAggregation()
		}
		trainingMutex.Unlock()
	}
}

func notifyTrainingComplete() {
	// This function notifies the leader that this node has completed training
	client := http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(fmt.Sprintf("http://%s/trainingComplete?node=%s", leader, myAddress))
	if err != nil {
		fmt.Printf("[ERROR] Failed to notify leader about training completion: %v\n", err)
	} else {
		fmt.Println("[INFO] Notified leader about training completion")
		resp.Body.Close()
	}
}

func trainingCompleteHandler(w http.ResponseWriter, r *http.Request) {
	// Leader receives notification that a node completed training
	node := r.URL.Query().Get("node")

	trainingMutex.Lock()
	trainingComplete[node] = true
	nodeResponses++
	remaining := len(nodes) - nodeResponses
	fmt.Printf("[INFO] Node %s completed training. Waiting for %d more nodes...\n",
		node, remaining)
	trainingMutex.Unlock()

	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, `{"message": "Training completion recorded"}`)

	// If all nodes have completed, start aggregation
	if nodeResponses == len(nodes) {
		fmt.Println("aggregation function called!! ")
		go startAggregation()
	}
}

func startAggregation() {
	fmt.Println("[INFO] All nodes have completed training. Starting model aggregation...")
	res, err := http.Get("http://localhost:6002/aggregate_models")
	if err != nil {
		fmt.Printf("[ERROR] Failed to trigger model aggregation: %v\n", err)
	} else {
		fmt.Println("[INFO] Aggregation process started successfully")
		res.Body.Close()
	}
}

// Modify distributeModelHandler to implement Phase 1 of 2PC
func distributeModelHandler(w http.ResponseWriter, r *http.Request) {
    // This endpoint is called on the leader to distribute the aggregated model
    body, _ := ioutil.ReadAll(r.Body)
    var aggregatedModel AggregatedModel
    json.Unmarshal(body, &aggregatedModel)

    // Reset confirmations for this round
    twoPhaseCommitMutex.Lock()
    modelUpdateConfirmations = make(map[string]bool)
    twoPhaseCommitMutex.Unlock()

    // Update local model first
    updateLocalModel(aggregatedModel)

    // Phase 1: Distribute to all other nodes and wait for prepare confirmations
    prepareResponses := 0
    prepareSuccessful := true
    prepareChannel := make(chan bool, len(nodes)-1)
    
    for _, node := range nodes {
        if node != myAddress {
            go func(nodeAddr string) {
                res, err := http.Post("http://"+nodeAddr+"/prepareModelUpdate", 
                                     "application/json", bytes.NewBuffer(body))
                if err != nil || res.StatusCode != http.StatusOK {
                    fmt.Printf("[ERROR] Node %s failed to prepare model update: %v\n", 
                              nodeAddr, err)
                    prepareChannel <- false
                } else {
                    fmt.Printf("[INFO] Node %s prepared for model update\n", nodeAddr)
                    prepareChannel <- true
                    res.Body.Close()
                }
            }(node)
        }
    }
    
    // Wait for responses with timeout
    timeout := time.After(30 * time.Second)
    for i := 0; i < len(nodes)-1; i++ {
        select {
        case success := <-prepareChannel:
            prepareResponses++
            if !success {
                prepareSuccessful = false
            }
        case <-timeout:
            fmt.Println("[ERROR] Timeout waiting for prepare responses")
            prepareSuccessful = false
            break
        }
    }
    
    // Phase 2: If all prepared successfully, commit; otherwise abort
    if prepareSuccessful && prepareResponses == len(nodes)-1 {
        // Commit phase
        for _, node := range nodes {
            if node != myAddress {
                go func(nodeAddr string) {
                    _, err := http.Get("http://" + nodeAddr + "/commitModelUpdate")
                    if err != nil {
                        fmt.Printf("[ERROR] Failed to send commit to %s: %v\n", 
                                  nodeAddr, err)
                    } else {
                        fmt.Printf("[INFO] Sent commit to %s\n", nodeAddr)
                    }
                }(node)
            }
        }
        
        fedRound++
        fmt.Printf("[INFO] Completed federated learning round %d of %d\n", 
                  fedRound, maxRounds)
        
        // If we haven't reached max rounds, schedule the next round
        if fedRound < maxRounds {
            go scheduleNextRound()
        } else {
            fmt.Println("[INFO] Completed all federated learning rounds!")
        }
    } else {
        // Abort phase - could implement retry logic here
        fmt.Println("[ERROR] Not all nodes prepared successfully, aborting update")
        for _, node := range nodes {
            if node != myAddress {
                go func(nodeAddr string) {
                    http.Get("http://" + nodeAddr + "/abortModelUpdate")
                }(node)
            }
        }
    }

    w.WriteHeader(http.StatusOK)
    fmt.Fprintf(w, `{"message": "Model distribution process completed"}`)
}

// Add these new handlers for 2PC
func prepareModelUpdateHandler(w http.ResponseWriter, r *http.Request) {
    body, _ := ioutil.ReadAll(r.Body)
    var aggregatedModel AggregatedModel
    json.Unmarshal(body, &aggregatedModel)
    
    // Try to update the local model but don't commit yet
    success := prepareLocalModelUpdate(aggregatedModel)
    
    if success {
        w.WriteHeader(http.StatusOK)
    } else {
        w.WriteHeader(http.StatusInternalServerError)
    }
}

func prepareLocalModelUpdate(model AggregatedModel) bool {
    // Try to update model and verify it worked
    data, _ := json.Marshal(model)
    res, err := http.Post("http://localhost:6002/prepare_model_update", 
                         "application/json", bytes.NewBuffer(data))
    if err != nil {
        fmt.Println("[ERROR] Failed to prepare local model update:", err)
        return false
    }
    defer res.Body.Close()
    return res.StatusCode == http.StatusOK
}

func commitModelUpdateHandler(w http.ResponseWriter, r *http.Request) {
    // Commit the previously prepared model update
    success := commitLocalModelUpdate()
    
    if success {
        w.WriteHeader(http.StatusOK)
        fmt.Println("[INFO] Model update committed successfully")
    } else {
        w.WriteHeader(http.StatusInternalServerError)
        fmt.Println("[ERROR] Failed to commit model update")
    }
}

func commitLocalModelUpdate() bool {
    res, err := http.Get("http://localhost:6002/commit_model_update")
    if err != nil {
        fmt.Println("[ERROR] Failed to commit local model update:", err)
        return false
    }
    defer res.Body.Close()
    return res.StatusCode == http.StatusOK
}

func abortModelUpdateHandler(w http.ResponseWriter, r *http.Request) {
    // Abort the previously prepared model update
    http.Get("http://localhost:6002/abort_model_update")
    w.WriteHeader(http.StatusOK)
    fmt.Println("[INFO] Model update aborted")
}

func updateModelHandler(w http.ResponseWriter, r *http.Request) {
	// This endpoint is called on non-leader nodes to update their model
	body, _ := ioutil.ReadAll(r.Body)
	var aggregatedModel AggregatedModel
	json.Unmarshal(body, &aggregatedModel)

	updateLocalModel(aggregatedModel)

	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, `{"message": "Model updated successfully"}`)
}

func updateLocalModel(model AggregatedModel) {
	// Send the aggregated model to the local Python service
	data, _ := json.Marshal(model)
	res, err := http.Post("http://localhost:6002/update_model", "application/json", bytes.NewBuffer(data))
	if err != nil {
		fmt.Println("[ERROR] Failed to update local model:", err)
	} else {
		fmt.Println("[INFO] Updated local model:", res.Status)
	}
}

func scheduleNextRound() {
	// Wait a bit before starting the next round
	time.Sleep(5 * time.Second)

	// Reset training status tracking for new round
	trainingMutex.Lock()
	trainingComplete = make(map[string]bool)
	nodeResponses = 0
	trainingMutex.Unlock()

	fmt.Printf("[INFO] Starting round %d of federated learning\n", fedRound+1)

	// Broadcast to all nodes to start training
	for _, node := range nodes {
		go func(nodeAddr string) {
			_, err := http.Get("http://" + nodeAddr + "/startTraining")
			if err != nil {
				fmt.Printf("[ERROR] Failed to trigger training on %s: %v\n", nodeAddr, err)
			}
		}(node)
	}
}

func initiateTrainingHandler(w http.ResponseWriter, r *http.Request) {
	// Reset counters
	fedRound = 0

	// Start the first round of training on all nodes
	fmt.Println("[INFO] Initiating federated learning across all nodes")
	scheduleNextRound()

	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, `{"message": "Federated learning initiated"}`)
}

func main() {
    // Load configuration
    loadConfig()

    // Start HTTP server for handling election requests
    http.HandleFunc("/election", electionHandler)
    http.HandleFunc("/leader", leaderHandler)
    http.HandleFunc("/ping", pingHandler)
    http.HandleFunc("/receiveModelParam", receiveModelParam)
    http.HandleFunc("/startTraining", startTrainingHandler)
    http.HandleFunc("/distributeModel", distributeModelHandler)
    http.HandleFunc("/updateModel", updateModelHandler)
    http.HandleFunc("/initiateFederated", initiateTrainingHandler)
    http.HandleFunc("/trainingComplete", trainingCompleteHandler) // Add this new handler
    http.HandleFunc("/prepareModelUpdate", prepareModelUpdateHandler) // Add this new handler
    http.HandleFunc("/commitModelUpdate", commitModelUpdateHandler) // Add this new handler
    http.HandleFunc("/abortModelUpdate", abortModelUpdateHandler) // Add this new handler

    go monitorLeader()

    // Extract port from myAddress
    parts := strings.Split(myAddress, ":")
    port := ":" + parts[1]

    fmt.Println("[INFO] Node started:", myAddress)
    log.Fatal(http.ListenAndServe(port, nil))
}
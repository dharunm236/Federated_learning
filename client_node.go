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

// Suzuki-Kasami algorithm variables
var hasToken bool = false
var requestNum map[string]int
var tokenQueue []string
var tokenMutex sync.Mutex

// Shared model weights list
var receivedWeights []ModelParams
var weightsReceived int

// Token message structure
type TokenMessage struct {
    Queue []string          `json:"queue"`
    RN    map[string]int    `json:"request_numbers"`
}

var aggregatorInProgress bool = false

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

    // Initialize Suzuki-Kasami algorithm
    requestNum = make(map[string]int)
    for _, node := range nodes {
        requestNum[node] = 0
    }
    
    // Initialize token holder (lowest address gets the initial token)
    if len(nodes) > 0 {
        lowestNode := nodes[0]
        for _, node := range nodes {
            if node < lowestNode {
                lowestNode = node
            }
        }
        
        if myAddress == lowestNode {
            tokenMutex.Lock()
            hasToken = true
            tokenMutex.Unlock()
            fmt.Println("[INFO] This node initially holds the token")
        }
    }

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

	// After training is complete, request token to update shared weights
    go requestToken()
    
    // If I'm the leader, record my own completion
    if leader == myAddress {
        trainingMutex.Lock()
        trainingComplete[myAddress] = true
        nodeResponses++
        fmt.Printf("[INFO] Leader completed training. Waiting for %d more nodes...\n", len(nodes)-nodeResponses)
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
}

func startAggregation() {
	fmt.Println("[INFO] All nodes have completed training. Starting model aggregation using shared weights...")
    
    // Convert shared weights to JSON
    aggregationData := map[string]interface{}{
        "weights_list": receivedWeights,
    }
    data, _ := json.Marshal(aggregationData)
    
    // Send to Python service for aggregation
    res, err := http.Post("http://localhost:6002/aggregate_models", 
                           "application/json", bytes.NewBuffer(data))
    if err != nil {
        fmt.Printf("[ERROR] Failed to trigger model aggregation: %v\n", err)
    } else {
        fmt.Println("[INFO] Aggregation process started successfully")
        res.Body.Close()
        
        // Reset for next round
        receivedWeights = []ModelParams{}
        weightsReceived = 0
    }
    aggregatorInProgress = false
}

func distributeModelHandler(w http.ResponseWriter, r *http.Request) {
	// This endpoint is called on the leader to distribute the aggregated model
	body, _ := ioutil.ReadAll(r.Body)
	var aggregatedModel AggregatedModel
	json.Unmarshal(body, &aggregatedModel)

	// Update local model first
	updateLocalModel(aggregatedModel)

	// Distribute to all other nodes
	for _, node := range nodes {
		if node != myAddress {
			res, err := http.Post("http://"+node+"/updateModel", "application/json", bytes.NewBuffer(body))
			if err != nil {
				fmt.Printf("[ERROR] Failed to send updated model to %s: %v\n", node, err)
			} else {
				fmt.Printf("[INFO] Sent updated model to %s: %s\n", node, res.Status)
			}
		}
	}

	fedRound++
	fmt.Printf("[INFO] Completed federated learning round %d of %d\n", fedRound, maxRounds)

	// If we haven't reached max rounds, schedule the next round
	if fedRound < maxRounds {
		go scheduleNextRound()
	} else {
		fmt.Println("[INFO] Completed all federated learning rounds!")
	}

	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, `{"message": "Model distributed to all nodes"}`)
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
	fmt.Println("update local model func called")
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

// Suzuki-Kasami algorithm functions
func requestToken() {
    // Increment request number for this node
    requestNum[myAddress]++
    
    // Send request to all other nodes
    for _, node := range nodes {
        if node != myAddress {
            sendTokenRequest(node, requestNum[myAddress])
        }
    }
}

func sendTokenRequest(node string, reqNum int) {
    reqData := map[string]interface{}{
        "source": myAddress,
        "reqNum": reqNum,
    }
    
    data, _ := json.Marshal(reqData)
    client := http.Client{Timeout: 10 * time.Second}
    _, err := client.Post("http://"+node+"/tokenRequest", "application/json", bytes.NewBuffer(data))
    if err != nil {
        fmt.Printf("[ERROR] Failed to send token request to %s: %v\n", node, err)
    }
}

func tokenRequestHandler(w http.ResponseWriter, r *http.Request) {
    var reqData map[string]interface{}
    body, _ := ioutil.ReadAll(r.Body)
    json.Unmarshal(body, &reqData)
    
    sourceNode := reqData["source"].(string)
    reqNum := int(reqData["reqNum"].(float64))
    
    // Update request number for this node
    if requestNum[sourceNode] < reqNum {
        requestNum[sourceNode] = reqNum
    }
    
    // If I have the token and I'm not using it, send it
    tokenMutex.Lock()
    if hasToken && !isTokenInUse() {
        hasToken = false
        sendToken(sourceNode)
    }
    tokenMutex.Unlock()
    
    w.WriteHeader(http.StatusOK)
}

func isTokenInUse() bool {
    // In this implementation, the token is considered in use if
    // this node is currently processing its own weights
    return false // Simplified for this example
}

func sendToken(node string) {
    // Add the recipient to the token queue
    if requestNum[node] > 0 {
        tokenQueue = append(tokenQueue, node)
    }
    
    // Create token message
    tokenMsg := TokenMessage{
        Queue: tokenQueue,
        RN:    requestNum,
    }
    
    data, _ := json.Marshal(tokenMsg)
    client := http.Client{Timeout: 10 * time.Second}
    _, err := client.Post("http://"+node+"/receiveToken", "application/json", bytes.NewBuffer(data))
    if err != nil {
        fmt.Printf("[ERROR] Failed to send token to %s: %v\n", node, err)
        
        // If failed to send token, keep it
        tokenMutex.Lock()
        hasToken = true
        tokenMutex.Unlock()
    } else {
        fmt.Printf("[INFO] Sent token to %s\n", node)
    }
}

func receiveTokenHandler(w http.ResponseWriter, r *http.Request) {
    var tokenMsg TokenMessage
    body, _ := ioutil.ReadAll(r.Body)
    json.Unmarshal(body, &tokenMsg)
    
    tokenMutex.Lock()
    hasToken = true
    tokenQueue = tokenMsg.Queue
    
    // Update request numbers
    for node, rn := range tokenMsg.RN {
        if requestNum[node] < rn {
            requestNum[node] = rn
        }
    }
    tokenMutex.Unlock()
    
    fmt.Println("[INFO] Received token")
    
    // If we were waiting for the token to update weights, do it now
    updateSharedWeights()
    
    w.WriteHeader(http.StatusOK)
}

func releaseToken() {
    tokenMutex.Lock()
    defer tokenMutex.Unlock()
    
    // Find the next node that needs the token
    if len(tokenQueue) > 0 {
        nextNode := tokenQueue[0]
        tokenQueue = tokenQueue[1:] // Remove from queue
        hasToken = false
        
        go sendToken(nextNode)
    }
}

func updateSharedWeights() {
    fmt.Println("[INFO] Updating shared weights list")
    
    // Get model weights from Python service
    res, err := http.Get("http://localhost:6002/get_model")
    if err != nil {
        fmt.Println("[ERROR] Failed to get model from Python service:", err)
        releaseToken()
        return
    }
    defer res.Body.Close()
    
    body, _ := ioutil.ReadAll(res.Body)
    var params ModelParams
    json.Unmarshal(body, &params)
    
    // Add to shared weights list
    receivedWeights = append(receivedWeights, params)
    weightsReceived++
    
    fmt.Printf("[INFO] Added weights to shared list (total: %d)\n", weightsReceived)
    
    // Check if we're the leader and all weights are received
    if myAddress == leader && !aggregatorInProgress && weightsReceived == len(nodes) {
        aggregatorInProgress = true
        fmt.Println("[INFO] All weights received. Starting aggregation...")
        go startAggregation()
    }
    
    // Release the token
    releaseToken()
    
    // Notify leader training is complete
    if myAddress != leader {
        notifyTrainingComplete()
    }
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
    http.HandleFunc("/trainingComplete", trainingCompleteHandler)
    
    // Add Suzuki-Kasami token handlers
    http.HandleFunc("/tokenRequest", tokenRequestHandler)
    http.HandleFunc("/receiveToken", receiveTokenHandler)

    go monitorLeader()

    // Extract port from myAddress
    parts := strings.Split(myAddress, ":")
    port := ":" + parts[1]

    fmt.Println("[INFO] Node started:", myAddress)
    log.Fatal(http.ListenAndServe(port, nil))
}
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

var (
	nodes           []string
	myAddress       string
	leader          string
	mu              sync.Mutex
	fedRound        = 0
	maxRounds       = 10
	trainingComplete = make(map[string]bool)
	nodeResponses   = 0
	trainingMutex   sync.Mutex
)

type ModelParams struct {
	Weights json.RawMessage `json:"weights"`
}

type AggregatedModel struct {
	Weights json.RawMessage `json:"weights"`
}

type Config struct {
	MyAddress string   `json:"myAddress"`
	Nodes     []string `json:"nodes"`
}

func loadConfig() {
	configFile, err := ioutil.ReadFile("config.json")
	if err != nil {
		log.Fatalf("[ERROR] Config read failed: %v", err)
	}

	var config Config
	if err := json.Unmarshal(configFile, &config); err != nil {
		log.Fatalf("[ERROR] Config parse failed: %v", err)
	}

	myAddress = config.MyAddress
	nodes = config.Nodes
}

func startElection() {
	mu.Lock()
	leader = ""
	mu.Unlock()

	higherNodes := make([]string, 0)
	for _, node := range nodes {
		if node > myAddress {
			higherNodes = append(higherNodes, node)
		}
	}

	if len(higherNodes) == 0 {
		declareLeader()
		return
	}

	responses := make(chan bool, len(higherNodes))
	for _, node := range higherNodes {
		go func(n string) {
			res, err := http.Get("http://" + n + "/election")
			responses <- (err == nil && res.StatusCode == http.StatusOK)
		}(node)
	}

	timeout := time.After(5 * time.Second)
	responded, higherAlive := 0, false
	for responded < len(higherNodes) {
		select {
		case alive := <-responses:
			responded++
			if alive {
				higherAlive = true
			}
		case <-timeout:
			break
		}
	}

	if !higherAlive {
		declareLeader()
	}
}

func declareLeader() {
	mu.Lock()
	leader = myAddress
	mu.Unlock()

	fmt.Println("New leader:", leader)
	for _, node := range nodes {
		if node != myAddress {
			http.Get("http://" + node + "/leader?address=" + leader)
		}
	}
}

func electionHandler(w http.ResponseWriter, _ *http.Request) {
	go startElection()
	w.WriteHeader(http.StatusOK)
}

func leaderHandler(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	leader = r.URL.Query().Get("address")
	mu.Unlock()
	w.WriteHeader(http.StatusOK)
}

func monitorLeader() {
	for {
		time.Sleep(5 * time.Second)

		mu.Lock()
		if leader == "" {
			mu.Unlock()
			startElection()
			continue
		}
		mu.Unlock()

		alive := false
		for i := 0; i < 3; i++ {
			res, err := http.Get("http://" + leader + "/ping")
			if err == nil && res.StatusCode == http.StatusOK {
				alive = true
				break
			}
			time.Sleep(500 * time.Millisecond)
		}

		if !alive {
			startElection()
		}
	}
}

func pingHandler(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func startTrainingHandler(w http.ResponseWriter, _ *http.Request) {
	mu.Lock()
	currentLeader := leader
	mu.Unlock()

	url := fmt.Sprintf("http://localhost:6002/train_local?leader=%s", currentLeader)
	res, err := http.Get(url)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintf(w, `{"error": "training failed"}`)
		return
	}

	body, _ := ioutil.ReadAll(res.Body)
	w.WriteHeader(http.StatusOK)
	w.Write(body)
}

func distributeModelHandler(w http.ResponseWriter, r *http.Request) {
	body, _ := ioutil.ReadAll(r.Body)
	var model AggregatedModel
	json.Unmarshal(body, &model)

	// Update local model
	data, _ := json.Marshal(model)
	http.Post("http://localhost:6002/update_model", "application/json", bytes.NewBuffer(data))

	// Distribute to other nodes
	for _, node := range nodes {
		if node != myAddress {
			http.Post("http://"+node+"/updateModel", "application/json", bytes.NewBuffer(body))
		}
	}

	fedRound++
	if fedRound < maxRounds {
		time.Sleep(5 * time.Second)
		trainingMutex.Lock()
		trainingComplete = make(map[string]bool)
		nodeResponses = 0
		trainingMutex.Unlock()
		for _, node := range nodes {
			go func(n string) { http.Get("http://" + n + "/startTraining") }(node)
		}
	} else {
		log.Println("[INFO] Training completed")
	}
	w.WriteHeader(http.StatusOK)
}

func main() {
	loadConfig()

	http.HandleFunc("/election", electionHandler)
	http.HandleFunc("/leader", leaderHandler)
	http.HandleFunc("/ping", pingHandler)
	http.HandleFunc("/startTraining", startTrainingHandler)
	http.HandleFunc("/distributeModel", distributeModelHandler)

	go monitorLeader()

	parts := strings.Split(myAddress, ":")
	fmt.Printf("[INFO] Node %s starting\n", myAddress)
	log.Fatal(http.ListenAndServe(":"+parts[1], nil))
}
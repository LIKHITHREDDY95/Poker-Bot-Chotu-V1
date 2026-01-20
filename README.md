# Poker Chotu V1: Hybrid AI & Simulation Environment
## 1. Abstract
**Poker Chotu V1** is a comprehensive research environment and application designed for the development, training, and simulation of autonomous poker agents. The system integrates Deep Reinforcement Learning (DRL), Game Theory Optimal (GTO) heuristics, and Case-Based Reasoning (CBR) within a unified framework. It features a custom gym-style environment (`PokerEnv`), a Deep Q-Network implementation (`DQN`), and a real-time web-based visualization interface for monitoring agent performance and 1v1 automated simulations.
## 2. System Architecture
The project is architected as a modular system comprising three primary layers:
### 2.1. The AI Layer (Agent Logic)
*   **Deep Q-Network (DQN)**: A PyTorch-based neural network trained via self-play. It maps a 19-dimensional state vector (representing hole cards, community cards, pot odds, and stack sizes) to a discrete action space (Fold, Call, Raise-Min, Raise-Big).
    *   **Architecture**: Multi-Layer Perceptron (MLP) with ReLU activations and Dropout for regularization.
    *   **Training**: Experience Replay Buffer with Epsilon-Greedy exploration and Target Network stabilization.
*   **GTO Proxy ("DeepStack-Lite")**: A deterministic algorithmic agent designed to approximate Nash Equilibrium strategies using real-time heuristic constraints.
    *   **Logic**: Implements Minimum Defense Frequency (MDF) and Equity calculations to balance ranges and prevent exploitability.
*   **Vector Memory Module**: A Case-Based Reasoning system utilizing `ChromaDB` (local vector store).
    *   **Function**: Encodes game states into high-dimensional vectors to enable semantic similarity search. This allows agents to query historical hands for strategic decision-making based on past outcomes.
### 2.2. The Environment Layer
*   **PokerEnv**: A custom Python environment compliant with OpenAI Gym interfaces. It manages the game state, rules engine, reward distribution, and state observation encoding.
*   **State Space**: The observation vector includes normalized card rankings, suit encodings, betting history, and game phase indicators.
### 2.3. The Visualization Layer (Ui-Lite)
*   **Frontend**: A responsive Web Application built with Vanilla JavaScript, HTML5, and CSS3.
*   **Simulation Engine**: An asynchronous event loop capable of executing high-frequency 1v1 matches between agents ("Skynet" vs. "DeepStack") without blocking the main thread.
*   **God-View Telemetry**: Real-time rendering of hidden information (hole cards) during simulation mode for analysis.
---
## 3. Technical Specifications
### 3.1. Deep Q-Network Config
*   **Input Layer**: 19 Neurons (State Vector)
*   **Hidden Layers**: 128 -> 64 -> 32 (ReLU)
*   **Output Layer**: 4 Neurons (Action Q-Values)
*   **Optimizer**: Adam (`lr=0.001`)
*   **Loss Function**: MSE (Mean Squared Error)
### 3.2. Vector Memory Embeddings
The `PokerMemory` module utilizes a heuristic embedding strategy to map poker states to vector space:
$$ V_{state} = [ H_{strength}, P_{odds}, S_{street}, A_{aggression}, R_{stack} ] $$
*   **Storage**: ChromaDB (Persistent Local Storage)
*   **Distance Metric**: Cosine Similarity
---
## 4. Repository Structure
```
/Poker_Chotu_V1
├── /poker_ui_lite          # Visualization & Frontend Interface
│   ├── poker_game.js       # Core Game Engine (JavaScript Implementation)
│   ├── script.js           # Simulation Loop & UI Controller
│   └── index.html          # Application Entry Point
│
├── /poker_memory_db        # Vector Database Storage (Auto-Generated)
├── vector_memory.py        # Vector Database Interface (ChromaDB)
├── train_poker_ai.py       # Training Pipeline (RL + Memory Integration)
├── poker_dqn_trainer.py    # PyTorch Model Definitions
├── poker_env.py            # Simulation Environment (Python)
└── requirements.txt        # Dependency Manifest
```
---
## 5. Installation & Setup
### Prerequisites
*   **Python 3.8+**
*   **Node.js** (Optional, for advanced development)
### Dependencies
Install the required Python packages:
```bash
pip install torch numpy chromadb
```
### Execution
#### Training the Agent
To initiate the training loop for the DQN agent and populate the Vector Memory:
```bash
python train_poker_ai.py
```
#### Running the Interface
Launch the visualization interface by opening `poker_ui_lite/index.html` in any modern web browser.
*   **Human vs AI**: Select "New Hand" to test the model interactively.
*   **Simulation Mode**: Select **"⚔️ Sim 1v1"** to initiate the high-frequency evaluation loop between the Adaptive Agent and GTO Proxy.
---
## 6. Future Development
*   **Counterfactual Regret Minimization (CFR)**: Transitioning from heuristic GTO proxies to real-time solver-based strategies.
*   **Transformer-Based Embeddings**: Replacing heuristic state vectors with learned embeddings for improved semantic search in the Vector Memory.
*   **Multi-Agent Reinforcement Learning (MARL)**: expanding the environment to support >2 agents for ring-game solving.
---
## Author
likhith
*   Research & Development

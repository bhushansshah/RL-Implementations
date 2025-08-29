# Multi-Armed Bandit – ε-Greedy Exploration

This project contains an implementation of the **k-Armed Bandit problem** with different strategies for estimating action values and exploring optimal actions. The notebook (`main.ipynb`) demonstrates how the **ε-greedy method** performs under **stationary** and **non-stationary reward distributions**.

## 📘 Overview
The multi-armed bandit problem is a classic reinforcement learning setup that models the trade-off between **exploration** (trying new actions) and **exploitation** (choosing the best-known action).

In this notebook, we:
- Implement the ε-greedy algorithm.
- Compare **sample average method** vs. **constant step-size method** for action-value estimation.
- Analyze performance on both **stationary** and **non-stationary** reward distributions.
- Visualize results through reward and action-value plots.

## 🚀 Key Concepts
- **ε-Greedy Strategy**: With probability ε, a random action is selected (exploration), and with probability 1-ε, the best-known action is chosen (exploitation).  
- **Stationary Distribution**: Reward probabilities remain fixed over time.  
- **Non-Stationary Distribution**: Reward probabilities change over time, requiring adaptive methods.  
- **Action-Value Estimation Methods**:  
  - **Sample Average** – averages all past rewards.  
  - **Constant Step-Size (α)** – gives higher weight to recent rewards, adapting to changes.  

## 📊 Results
- **Stationary Case**: The ε-greedy approach successfully converges to optimal action estimates and maximizes rewards.  
- **Non-Stationary Case**: The **constant step-size method** adapts better than sample averages, leading to improved performance in dynamic environments.  

## 🛠️ Requirements
Install dependencies with:
```bash
pip install numpy matplotlib
```

## ▶️ Usage
Run the notebook:
```bash
jupyter notebook main.ipynb
```

## 📂 Files

main.ipynb – Implementation of k-Armed Bandit experiments with visualizations.

README.md – Project overview and usage instructions.

# 🎰 Multi-Armed Bandit A/B Testing: Epsilon-Greedy vs. Thompson Sampling

This project implements and compares two classic multi-armed bandit strategies—**Epsilon-Greedy** and **Thompson Sampling**—on a simulated 4-arm bandit problem using Python.

---

## 📌 Problem Setup

- **Number of Arms (Bandits):** 4  
- **True Mean Rewards:** `[1, 2, 3, 4]`  
- **Number of Trials:** 20,000  
- **Reward Noise:** Normally distributed (mean-centered)

---

## 🧠 Algorithms Implemented

### 🔹 Epsilon-Greedy
- Decaying ε: `ε = 0.1 / t`
- Balances exploration and exploitation

### 🔹 Thompson Sampling
- Bayesian approach using Beta priors
- Explores based on posterior sampling

---

## 📊 Output & Files

The experiment generates the following CSV files:

- `bandit_rewards.csv`: Log of all actions `{Bandit, Reward, Algorithm}`
- `epsilon_greedy_rewards.csv`: Cumulative reward & regret per trial
- `thompson_sampling_rewards.csv`: Same as above for Thompson Sampling

Visualization includes:
- Individual algorithm reward trends
- Comparison of cumulative reward and regret

---

## 🚀 How to Run

```bash
python Bandit.py


pip install numpy pandas matplotlib loguru

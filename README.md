# Contextual Multi-Armed Bandits Library

## Overview

This Python library provides implementations of contextual multi-armed bandit algorithms, including LinUCB and Epsilon-Greedy strategies. It is designed for easy integration into reinforcement learning systems that require decision-making under uncertainty with contextual information.

<p align="center">
<a href="https://github.com/singhsidhukuldeep/contextual-bandits"><img src="./Contextual Bandit Algorithms.png" alt="contextual-bandits" width ="75%" /></a>
</p>

## Features

-   **Contextual Algorithms**:
    -   **LinUCB**: Balances exploration and exploitation using linear regression with upper confidence bounds.
    -   **Epsilon-Greedy**: Explores randomly with probability epsilon and exploits the best-known option otherwise.
    -   **KernelUCB**: Uses kernel methods to capture non-linear relationships in the context space.
    -   **NeuralLinearBandit**: Combines neural networks for feature extraction with linear models for prediction.
    -   **DecisionTreeBandit**: Employs decision trees to model complex relationships between context and rewards.

-   **Non-Contextual Algorithms**:
    -   **Upper Confidence Bound (UCB)**: Selects arms based on upper confidence bounds of estimated rewards.
    -   **Thompson Sampling**: Uses Bayesian methods to balance exploration and exploitation.


## Installation

```bash
pip install contextual-bandits-algos
```

### Instructions to Use the Updated Library


1.  **Clone the Repository**

    ```bash
    git clone https://github.com/singhsidhukuldeep/contextual-bandits.git
    cd contextual-bandits
    ```

2.  **Install the Requirements**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the Package**

    ```bash
    pip install .
    ```

4.  **Run the Examples**

    ```bash
    python examples/example_linucb.py
    python examples/example_epsilon_greedy.py
    python examples/example_ucb.py
    python examples/example_thompson_sampling.py
    python examples/example_kernelucb.py
    python examples/example_neurallinear.py
    python examples/example_tree_bandit.py
    ```

    To run both algorithms in a single script:

    ```bash
    python examples/example_usage.py
    ```

    Run all examples *.py files
    
    ```bash
    cd examples
    for f in *.py; do echo "$f" & python "$f"; done
    ```

## Algorithms and Examples


Below are detailed descriptions of each algorithm, indicating whether they are contextual or non-contextual, how they work, and examples demonstrating their usage.

### 1. LinUCB

-   **Type**: Contextual
-   **Description**: The LinUCB algorithm is a contextual bandit algorithm that uses linear regression to predict the expected reward for each arm given the current context. It balances exploration and exploitation by adding an upper confidence bound to the estimated rewards.

#### How It Works

-   **Model**: Assumes that the reward is a linear function of the context features.
-   **Exploration**: Incorporates uncertainty in the estimation by adding a confidence interval (scaled by `alpha`).
-   **Exploitation**: Chooses the arm with the highest upper confidence bound.



### 2. Epsilon-Greedy

-   **Type**: Contextual
-   **Description**: The Epsilon-Greedy algorithm selects a random arm with probability `epsilon` (exploration) and the best-known arm with probability `1 - epsilon` (exploitation). It uses the context to predict rewards for each arm.

#### How It Works

-   **Model**: Uses linear models or other estimators to predict rewards based on context.
-   **Exploration**: With probability `epsilon`, selects a random arm.
-   **Exploitation**: With probability `1 - epsilon`, selects the arm with the highest predicted reward.



### 3. Upper Confidence Bound (UCB)

-   **Type**: Non-Contextual
-   **Description**: The UCB algorithm selects arms based on upper confidence bounds of the estimated rewards, without considering any context. It is suitable when no contextual information is available.

#### How It Works

-   **Model**: Estimates the average reward for each arm.
-   **Exploration**: Adds a confidence term to the average reward to explore less-tried arms.
-   **Exploitation**: Chooses the arm with the highest upper confidence bound.



### 4. Thompson Sampling

-   **Type**: Non-Contextual
-   **Description**: Thompson Sampling is a Bayesian algorithm that selects arms based on samples drawn from the posterior distributions of the arm's reward probabilities.

#### How It Works

-   **Model**: Assumes Bernoulli-distributed rewards for each arm.
-   **Exploration & Exploitation**: Balances both by sampling from the posterior distributions.
-   **Updates**: Updates the posterior distributions based on observed rewards.



### 5. KernelUCB

-   **Type**: Contextual
-   **Description**: KernelUCB uses kernel methods to capture non-linear relationships between contexts and rewards. It extends the UCB algorithm to a kernelized context space.

#### How It Works

-   **Model**: Uses a kernel function (e.g., RBF kernel) to compute similarity between contexts.
-   **Exploration**: Adds an exploration term based on the uncertainty in the kernel space.
-   **Exploitation**: Predicts the expected reward using kernel regression.



### 6. NeuralLinearBandit

-   **Type**: Contextual
-   **Description**: NeuralLinearBandit uses a neural network to learn a representation of the context and then applies linear regression on the learned features.

#### How It Works

-   **Model**: Combines a neural network for feature extraction with a linear model for reward prediction.
-   **Exploration**: Adds an exploration bonus based on the uncertainty of the linear model.
-   **Exploitation**: Uses the predicted rewards from the linear model.


### 7. DecisionTreeBandit

-   **Type**: Contextual
-   **Description**: The DecisionTreeBandit algorithm uses decision trees to model the relationship between context and rewards, allowing it to capture non-linear patterns.

#### How It Works

-   **Model**: Fits a decision tree regressor for each arm based on the observed contexts and rewards.
-   **Exploration**: Relies on the decision tree's predictions; may require additional mechanisms for exploration.
-   **Exploitation**: Selects the arm with the highest predicted reward.
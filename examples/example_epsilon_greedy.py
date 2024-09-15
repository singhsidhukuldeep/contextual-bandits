from contextual_bandits import EpsilonGreedy
import numpy as np


def main():
    n_arms = 5
    n_features = 10
    n_rounds = 1000
    epsilon = 0.1
    learning_rate = 0.01  # For updating weights

    # True parameters for simulation
    true_theta = np.random.randn(n_arms, n_features)

    # Initialize Epsilon-Greedy algorithm
    alg = EpsilonGreedy(n_arms, n_features, epsilon=epsilon)

    total_reward = 0

    for t in range(n_rounds):
        # Generate random context
        context = np.random.randn(n_features)

        # Select an arm
        chosen_arm = alg.select_arm(context)

        # Simulate reward
        noise = np.random.randn() * 0.1
        reward = np.dot(context, true_theta[chosen_arm]) + noise

        # Update algorithm
        alg.update(chosen_arm, context, reward, learning_rate)

        total_reward += reward

        if (t + 1) % 100 == 0:
            avg_reward = total_reward / (t + 1)
            print(f"Round {t + 1}: Average Reward = {avg_reward:.4f}")


if __name__ == "__main__":
    main()

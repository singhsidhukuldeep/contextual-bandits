from contextual_bandits import LinUCB, EpsilonGreedy
import numpy as np


def simulate_bandit(algorithm, n_arms, n_features, n_rounds, true_theta, alg_name):
    total_reward = 0

    for t in range(n_rounds):
        # Generate random context
        context = np.random.randn(n_features)

        # Select an arm
        chosen_arm = algorithm.select_arm(context)

        # Simulate reward
        noise = np.random.randn() * 0.1
        reward = np.dot(context, true_theta[chosen_arm]) + noise

        # Update algorithm
        if isinstance(algorithm, EpsilonGreedy):
            # Provide a learning rate for EpsilonGreedy
            algorithm.update(chosen_arm, context, reward, learning_rate=0.01)
        else:
            algorithm.update(chosen_arm, context, reward)

        total_reward += reward

        if (t + 1) % 100 == 0:
            avg_reward = total_reward / (t + 1)
            print(f"{alg_name} - Round {t + 1}: Average Reward = {avg_reward:.4f}")


def main():
    n_arms = 5
    n_features = 10
    n_rounds = 1000
    alpha = 0.1
    epsilon = 0.1

    # True parameters for simulation
    true_theta = np.random.randn(n_arms, n_features)

    # Initialize algorithms
    linucb_alg = LinUCB(n_arms, n_features, alpha=alpha)
    epsilon_greedy_alg = EpsilonGreedy(n_arms, n_features, epsilon=epsilon)

    print("Running LinUCB Algorithm:")
    simulate_bandit(linucb_alg, n_arms, n_features, n_rounds, true_theta, "LinUCB")

    print("\nRunning Epsilon-Greedy Algorithm:")
    simulate_bandit(
        epsilon_greedy_alg, n_arms, n_features, n_rounds, true_theta, "Epsilon-Greedy"
    )


if __name__ == "__main__":
    main()

from contextual_bandits import (
    LinUCB,
    EpsilonGreedy,
    UCB,
    ThompsonSampling,
    KernelUCB,
    NeuralLinearBandit,
    DecisionTreeBandit,
)
import numpy as np


def simulate_bandit(algorithm, n_arms, n_features, n_rounds, true_theta, alg_name):
    total_reward = 0
    rewards = []

    for t in range(n_rounds):
        # Generate random context
        context = np.random.randn(n_features)

        # Select an arm
        if isinstance(algorithm, (UCB, ThompsonSampling)):
            chosen_arm = algorithm.select_arm()
        else:
            chosen_arm = algorithm.select_arm(context)

        # Simulate reward
        if isinstance(algorithm, (UCB, ThompsonSampling)):
            # For non-contextual algorithms, use Bernoulli rewards
            reward_prob = 1 / (1 + np.exp(-np.dot(true_theta[chosen_arm], context)))
            reward = np.random.rand() < reward_prob
        else:
            noise = np.random.randn() * 0.1
            reward = np.dot(context, true_theta[chosen_arm]) + noise

        # Update algorithm
        if isinstance(algorithm, EpsilonGreedy):
            # Provide a learning rate for EpsilonGreedy
            algorithm.update(chosen_arm, context, reward, learning_rate=0.01)
        elif isinstance(algorithm, (UCB, ThompsonSampling)):
            algorithm.update(chosen_arm, reward=reward)
        else:
            algorithm.update(chosen_arm, context, reward)

        total_reward += reward
        rewards.append(total_reward / (t + 1))

        if (t + 1) % (n_rounds // 10) == 0:
            avg_reward = total_reward / (t + 1)
            print(f"{alg_name} - Round {t + 1}: Average Reward = {avg_reward:.4f}")

    return rewards


def main():
    n_arms = 5
    n_features = 10
    n_rounds = 500

    # True parameters for simulation
    true_theta = np.random.randn(n_arms, n_features)

    # Initialize algorithms
    algorithms = [
        ("LinUCB", LinUCB(n_arms, n_features, alpha=0.1)),
        ("Epsilon-Greedy", EpsilonGreedy(n_arms, n_features, epsilon=0.1)),
        ("UCB", UCB(n_arms)),
        ("Thompson Sampling", ThompsonSampling(n_arms)),
        (
            "KernelUCB",
            KernelUCB(n_arms, n_features, alpha=1.0, kernel="rbf", gamma=0.5),
        ),
        (
            "NeuralLinear",
            NeuralLinearBandit(n_arms, n_features, hidden_size=50, alpha=0.1),
        ),
        ("DecisionTreeBandit", DecisionTreeBandit(n_arms)),
    ]

    # Store rewards for plotting or analysis
    all_rewards = {}

    for alg_name, alg in algorithms:
        print(f"\nRunning {alg_name} Algorithm:")
        rewards = simulate_bandit(
            alg, n_arms, n_features, n_rounds, true_theta, alg_name
        )
        all_rewards[alg_name] = rewards

    # Optional: Plot the average rewards over time
    try:
        import matplotlib.pyplot as plt

        for alg_name, rewards in all_rewards.items():
            plt.plot(rewards, label=alg_name)

        plt.xlabel("Rounds")
        plt.ylabel("Average Reward")
        plt.title("Performance of Contextual Bandit Algorithms")
        plt.legend(loc="lower right")
        plt.savefig("Contextual Bandit Algorithms.png")
        print("plt saved: plt.savefig('Contextual Bandit Algorithms.png')")
        plt.show()
    except ImportError:
        print("Matplotlib not installed. Skipping the plot.")


if __name__ == "__main__":
    main()

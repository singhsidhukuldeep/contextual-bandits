from contextual_bandits import ThompsonSampling
import numpy as np


def main():
    n_arms = 5
    n_rounds = 1000

    # True reward probabilities for simulation
    true_means = np.random.rand(n_arms)

    # Initialize Thompson Sampling algorithm
    alg = ThompsonSampling(n_arms)

    total_reward = 0

    for t in range(n_rounds):
        # Select an arm
        chosen_arm = alg.select_arm()

        # Simulate reward (Bernoulli distribution)
        reward = np.random.rand() < true_means[chosen_arm]

        # Update algorithm
        alg.update(chosen_arm, reward=reward)

        total_reward += reward

        if (t + 1) % 100 == 0:
            avg_reward = total_reward / (t + 1)
            print(f"Round {t + 1}: Average Reward = {avg_reward:.4f}")


if __name__ == "__main__":
    main()

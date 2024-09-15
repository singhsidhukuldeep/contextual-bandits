import numpy as np


class ContextualBanditAlgorithm:
    """
    Base class for contextual bandit algorithms.

    Attributes:
        n_arms (int): Number of arms.
        n_features (int): Number of contextual features.
    """

    def __init__(self, n_arms: int, n_features: int):
        """
        Initialize the contextual bandit algorithm.

        Args:
            n_arms (int): Number of arms.
            n_features (int): Number of contextual features.
        """
        self.n_arms = n_arms
        self.n_features = n_features

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm based on the provided context.

        Args:
            context (np.ndarray): Contextual feature vector.

        Returns:
            int: The index of the selected arm.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update(self, arm: int, context: np.ndarray, reward: float):
        """
        Update the algorithm's parameters based on the observed reward.

        Args:
            arm (int): The index of the arm that was played.
            context (np.ndarray): Contextual feature vector.
            reward (float): Observed reward.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class LinUCB(ContextualBanditAlgorithm):
    """
    LinUCB algorithm implementation.

    Attributes:
        alpha (float): Exploration parameter.
        A (list): List of A matrices for each arm.
        b (list): List of b vectors for each arm.
    """

    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        """
        Initialize the LinUCB algorithm.

        Args:
            n_arms (int): Number of arms.
            n_features (int): Number of contextual features.
            alpha (float, optional): Exploration parameter. Defaults to 1.0.
        """
        super().__init__(n_arms, n_features)
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm using the LinUCB algorithm.

        Args:
            context (np.ndarray): Contextual feature vector.

        Returns:
            int: The index of the selected arm.
        """
        context = context.reshape(-1, 1)
        p = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            p[arm] = (
                theta.T @ context + self.alpha * np.sqrt(context.T @ A_inv @ context)
            ).item()
        return int(np.argmax(p))

    def update(self, arm: int, context: np.ndarray, reward: float):
        """
        Update the parameters of the selected arm.

        Args:
            arm (int): The index of the arm that was played.
            context (np.ndarray): Contextual feature vector.
            reward (float): Observed reward.
        """
        context = context.reshape(-1, 1)
        self.A[arm] += context @ context.T
        self.b[arm] += reward * context


class EpsilonGreedy(ContextualBanditAlgorithm):
    """
    Epsilon-Greedy algorithm implementation.

    Attributes:
        epsilon (float): Probability of exploring.
        weights (np.ndarray): Weight matrix for each arm.
    """

    def __init__(self, n_arms: int, n_features: int, epsilon: float = 0.1):
        """
        Initialize the Epsilon-Greedy algorithm.

        Args:
            n_arms (int): Number of arms.
            n_features (int): Number of contextual features.
            epsilon (float, optional): Probability of exploring. Defaults to 0.1.
        """
        super().__init__(n_arms, n_features)
        self.epsilon = epsilon
        self.weights = np.zeros((n_arms, n_features))

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm using the Epsilon-Greedy algorithm.

        Args:
            context (np.ndarray): Contextual feature vector.

        Returns:
            int: The index of the selected arm.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            predictions = self.weights @ context
            return int(np.argmax(predictions))

    def update(
        self, arm: int, context: np.ndarray, reward: float, learning_rate: float = 0.01
    ):
        """
        Update the weights of the selected arm.

        Args:
            arm (int): The index of the arm that was played.
            context (np.ndarray): Contextual feature vector.
            reward (float): Observed reward.
            learning_rate (float): Learning rate for the weight update.
        """
        prediction = self.weights[arm] @ context
        error = reward - prediction
        self.weights[arm] += learning_rate * error * context

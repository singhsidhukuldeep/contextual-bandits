import numpy as np
from scipy.stats import beta
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeRegressor
import torch
import torch.nn as nn
import torch.optim as optim


class ContextualBanditAlgorithm:
    """
    Base class for contextual bandit algorithms.

    Attributes:
        n_arms (int): Number of arms.
        n_features (int): Number of contextual features.
    """

    def __init__(self, n_arms: int, n_features: int = None):
        """
        Initialize the contextual bandit algorithm.

        Args:
            n_arms (int): Number of arms.
            n_features (int, optional): Number of contextual features.
        """
        self.n_arms = n_arms
        self.n_features = n_features

    def select_arm(self, context: np.ndarray = None) -> int:
        """
        Select an arm based on the provided context.

        Args:
            context (np.ndarray, optional): Contextual feature vector.

        Returns:
            int: The index of the selected arm.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update(self, arm: int, context: np.ndarray = None, reward: float = None):
        """
        Update the algorithm's parameters based on the observed reward.

        Args:
            arm (int): The index of the arm that was played.
            context (np.ndarray, optional): Contextual feature vector.
            reward (float, optional): Observed reward.
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


class UCB(ContextualBanditAlgorithm):
    """
    Upper Confidence Bound (UCB) algorithm implementation for non-contextual bandits.

    Attributes:
        counts (np.ndarray): Number of times each arm has been selected.
        values (np.ndarray): Average reward for each arm.
    """

    def __init__(self, n_arms: int):
        """
        Initialize the UCB algorithm.

        Args:
            n_arms (int): Number of arms.
        """
        super().__init__(n_arms)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0

    def select_arm(self, context: np.ndarray = None) -> int:
        """
        Select an arm using the UCB algorithm.

        Args:
            context (np.ndarray, optional): Contextual feature vector (ignored).

        Returns:
            int: The index of the selected arm.
        """
        if self.total_counts < self.n_arms:
            return self.total_counts  # Play each arm once to initialize
        else:
            ucb_values = self.values + np.sqrt(
                (2 * np.log(self.total_counts)) / self.counts
            )
            return int(np.argmax(ucb_values))

    def update(self, arm: int, context: np.ndarray = None, reward: float = None):
        """
        Update the estimated values for the selected arm.

        Args:
            arm (int): The index of the arm that was played.
            context (np.ndarray, optional): Contextual feature vector (ignored).
            reward (float): Observed reward.
        """
        self.counts[arm] += 1
        self.total_counts += 1
        n = self.counts[arm]
        value = self.values[arm]
        # Update the average reward
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward


class ThompsonSampling(ContextualBanditAlgorithm):
    """
    Thompson Sampling algorithm implementation for non-contextual bandits.

    Attributes:
        alpha (np.ndarray): Success counts for Beta distribution.
        beta (np.ndarray): Failure counts for Beta distribution.
    """

    def __init__(self, n_arms: int):
        """
        Initialize the Thompson Sampling algorithm.

        Args:
            n_arms (int): Number of arms.
        """
        super().__init__(n_arms)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self, context: np.ndarray = None) -> int:
        """
        Select an arm using Thompson Sampling.

        Args:
            context (np.ndarray, optional): Contextual feature vector (ignored).

        Returns:
            int: The index of the selected arm.
        """
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, context: np.ndarray = None, reward: float = None):
        """
        Update the alpha and beta parameters for the selected arm.

        Args:
            arm (int): The index of the arm that was played.
            context (np.ndarray, optional): Contextual feature vector (ignored).
            reward (float): Observed reward (assumed to be 0 or 1).
        """
        # Assuming reward is between 0 and 1
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward


class KernelUCB(ContextualBanditAlgorithm):
    """
    KernelUCB algorithm implementation.

    Attributes:
        alpha (float): Exploration parameter.
        contexts (list): List of observed contexts.
        rewards (list): List of observed rewards.
        kernel (callable): Kernel function.
    """

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        alpha: float = 1.0,
        kernel: str = "rbf",
        gamma: float = 1.0,
    ):
        """
        Initialize the KernelUCB algorithm.

        Args:
            n_arms (int): Number of arms.
            n_features (int): Number of contextual features.
            alpha (float, optional): Exploration parameter.
            kernel (str, optional): Type of kernel ('rbf', 'linear').
            gamma (float, optional): Kernel coefficient for 'rbf'.
        """
        super().__init__(n_arms, n_features)
        self.alpha = alpha
        self.contexts = [[] for _ in range(n_arms)]
        self.rewards = [[] for _ in range(n_arms)]
        if kernel == "rbf":
            self.kernel = lambda x, y: np.exp(-gamma * np.linalg.norm(x - y) ** 2)
        elif kernel == "linear":
            self.kernel = lambda x, y: np.dot(x, y)
        else:
            raise ValueError("Unsupported kernel type")

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm using the KernelUCB algorithm.

        Args:
            context (np.ndarray): Contextual feature vector.

        Returns:
            int: The index of the selected arm.
        """
        p = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if self.contexts[arm]:
                K = np.array([self.kernel(context, x) for x in self.contexts[arm]])
                K_inv = np.linalg.inv(np.outer(K, K) + np.eye(len(K)) * 1e-5)
                rewards = np.array(self.rewards[arm])
                mu = K @ K_inv @ rewards
                sigma = np.sqrt(self.kernel(context, context) - K @ K_inv @ K)
                p[arm] = mu + self.alpha * sigma
            else:
                p[arm] = self.alpha * np.sqrt(self.kernel(context, context))
        return int(np.argmax(p))

    def update(self, arm: int, context: np.ndarray, reward: float):
        """
        Update the observations for the selected arm.

        Args:
            arm (int): The index of the arm that was played.
            context (np.ndarray): Contextual feature vector.
            reward (float): Observed reward.
        """
        self.contexts[arm].append(context)
        self.rewards[arm].append(reward)


class ArmModel(nn.Module):
    def __init__(self, n_features, hidden_size):
        super(ArmModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.lin_weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        features = self.feature_extractor(x)
        output = features @ self.lin_weight
        return output, features


class NeuralLinearBandit(ContextualBanditAlgorithm):
    """
    NeuralLinearBandit algorithm implementation.

    Attributes:
        models (list): List of neural network models for each arm.
        optimizers (list): Optimizers for each arm.
    """

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        hidden_size: int = 50,
        alpha: float = 1.0,
    ):
        """
        Initialize the NeuralLinearBandit algorithm.

        Args:
            n_arms (int): Number of arms.
            n_features (int): Number of contextual features.
            hidden_size (int, optional): Size of hidden layer.
            alpha (float, optional): Exploration parameter.
        """
        super().__init__(n_arms, n_features)
        self.alpha = alpha
        self.models = []
        self.optimizers = []
        for _ in range(n_arms):
            model = ArmModel(n_features, hidden_size)
            self.models.append(model)
            optimizer = optim.Adam(model.parameters())
            self.optimizers.append(optimizer)

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm using the NeuralLinearBandit algorithm.

        Args:
            context (np.ndarray): Contextual feature vector.

        Returns:
            int: The index of the selected arm.
        """
        p = np.zeros(self.n_arms)
        context_tensor = torch.from_numpy(context.astype(np.float32))
        for i, model in enumerate(self.models):
            with torch.no_grad():
                output, features = model(context_tensor)
                mu = output.item()
                sigma = self.alpha * torch.norm(features).item()
                p[i] = mu + sigma
        return int(np.argmax(p))

    def update(self, arm: int, context: np.ndarray, reward: float):
        """
        Update the model parameters for the selected arm.

        Args:
            arm (int): The index of the arm that was played.
            context (np.ndarray): Contextual feature vector.
            reward (float): Observed reward.
        """
        model = self.models[arm]
        optimizer = self.optimizers[arm]
        context_tensor = torch.from_numpy(context.astype(np.float32))
        target = torch.tensor([reward], dtype=torch.float32)

        # Forward pass
        prediction, _ = model(context_tensor)

        # Compute loss
        loss = nn.MSELoss()(prediction, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class DecisionTreeBandit(ContextualBanditAlgorithm):
    """
    Decision Tree Bandit algorithm implementation.

    Attributes:
        models (list): List of decision tree models for each arm.
    """

    def __init__(self, n_arms: int):
        """
        Initialize the Decision Tree Bandit algorithm.

        Args:
            n_arms (int): Number of arms.
        """
        super().__init__(n_arms)
        self.models = [DecisionTreeRegressor() for _ in range(n_arms)]
        self.contexts = [[] for _ in range(n_arms)]
        self.rewards = [[] for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm using the Decision Tree Bandit algorithm.

        Args:
            context (np.ndarray): Contextual feature vector.

        Returns:
            int: The index of the selected arm.
        """
        predictions = np.zeros(self.n_arms)
        for i, model in enumerate(self.models):
            if self.contexts[i]:
                predictions[i] = model.predict([context])[0]
            else:
                predictions[i] = 0  # Default prediction
        return int(np.argmax(predictions))

    def update(self, arm: int, context: np.ndarray, reward: float):
        """
        Update the decision tree model for the selected arm.

        Args:
            arm (int): The index of the arm that was played.
            context (np.ndarray): Contextual feature vector.
            reward (float): Observed reward.
        """
        self.contexts[arm].append(context)
        self.rewards[arm].append(reward)
        # Retrain the model
        self.models[arm].fit(self.contexts[arm], self.rewards[arm])

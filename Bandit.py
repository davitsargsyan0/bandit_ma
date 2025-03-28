from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import os


class Bandit(ABC):
    """
    Abstract Bandit class for multi-armed bandit algorithms.
    """

    def __init__(self, k, rewards):
        """
        Initialize the Bandit with number of arms and reward means.

        :param k: Number of bandit arms.
        :param rewards: List of expected rewards for each arm.
        """
        self.k = k
        self.rewards = rewards
        self.action_count = np.zeros(k)
        self.action_rewards = np.zeros(k)
        self.cumulative_rewards = [0]
        self.cumulative_regrets = [0]
        self.logs = []

    @abstractmethod
    def pull(self):
        """
        Abstract method to select an arm. Must be implemented in subclass.
        """
        pass

    def update(self, action, reward, algorithm_name):
        """
        Update internal stats and log reward and regret.

        :param action: Chosen arm index.
        :param reward: Observed reward.
        :param algorithm_name: Name of the algorithm used.
        """
        self.action_count[action] += 1
        self.action_rewards[action] += reward
        self.cumulative_rewards.append(self.cumulative_rewards[-1] + reward)
        regret = max(self.rewards) - reward
        self.cumulative_regrets.append(self.cumulative_regrets[-1] + regret)
        self.logs.append({
            "Bandit": action,
            "Reward": reward,
            "Algorithm": algorithm_name
        })

    def get_reward(self, action):
        """
        Simulate reward using normal distribution around true mean.

        :param action: Chosen arm index.
        :return: Simulated reward.
        """
        return np.random.normal(self.rewards[action], 1)

    @abstractmethod
    def experiment(self, n_trials):
        """
        Run n_trials of the bandit experiment.
        
        :param n_trials: Number of trials to simulate.
        """
        pass

    def report(self, algorithm_name):
        """
        Save experiment logs and cumulative stats to CSV files.

        :param algorithm_name: Name of the algorithm used.
        """
        average_reward = np.sum(self.action_rewards) / np.sum(self.action_count)
        total_regret = self.cumulative_regrets[-1]
        logger.info(f"{algorithm_name} - Average Reward: {average_reward}")
        logger.info(f"{algorithm_name} - Total Regret: {total_regret}")

        # Append trial log to common file
        pd.DataFrame(self.logs).to_csv('bandit_rewards.csv', mode='a', index=False, header=not os.path.exists('bandit_rewards.csv'))

        # Save summary of cumulative reward and regret
        df_summary = pd.DataFrame({
            "Trial": range(len(self.cumulative_rewards)),
            "Cumulative Reward": self.cumulative_rewards,
            "Cumulative Regret": self.cumulative_regrets
        })
        filename = f"{algorithm_name.lower().replace(' ', '_')}_rewards.csv"
        df_summary.to_csv(filename, index=False)


class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy strategy for the multi-armed bandit problem.
    """

    def __init__(self, k, rewards, epsilon=0.1):
        """
        Initialize Epsilon-Greedy with decayable epsilon.

        :param k: Number of bandit arms.
        :param rewards: True reward means for each arm.
        :param epsilon: Initial exploration rate.
        """
        super().__init__(k, rewards)
        self.initial_epsilon = epsilon
        self.t = 1

    def pull(self):
        """
        Choose arm using epsilon-greedy strategy with decaying epsilon.

        :return: Selected arm index.
        """
        epsilon = self.initial_epsilon / self.t
        self.t += 1
        if np.random.rand() < epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.action_rewards / (self.action_count + 1e-10))

    def experiment(self, n_trials):
        """
        Run epsilon-greedy for n_trials.

        :param n_trials: Number of trials.
        """
        for _ in range(n_trials):
            action = self.pull()
            reward = self.get_reward(action)
            self.update(action, reward, "Epsilon-Greedy")


class ThompsonSampling(Bandit):
    """
    Thompson Sampling strategy for the multi-armed bandit problem.
    """

    def __init__(self, k, rewards):
        """
        Initialize Thompson Sampling with Beta priors.

        :param k: Number of bandit arms.
        :param rewards: True reward means for each arm.
        """
        super().__init__(k, rewards)
        self.alpha = np.ones(k)
        self.beta = np.ones(k)

    def pull(self):
        """
        Choose arm based on Beta sampling.

        :return: Selected arm index.
        """
        sampled_means = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_means)

    def experiment(self, n_trials):
        """
        Run Thompson Sampling for n_trials.

        :param n_trials: Number of trials.
        """
        for _ in range(n_trials):
            action = self.pull()
            reward = self.get_reward(action)
            self.update(action, reward, "Thompson Sampling")
            self.alpha[action] += reward
            self.beta[action] += 1


class Visualization:
    """
    Visualization utility class.
    """

    def plot1(self, data):
        """
        Plot cumulative reward over time.

        :param data: Cumulative reward list.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(data, label='Rewards over time')
        plt.title('Performance of Bandits')
        plt.legend()
        plt.show()

    def plot2(self, data1, data2):
        """
        Compare cumulative rewards from two algorithms.

        :param data1: First algorithm's reward data.
        :param data2: Second algorithm's reward data.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(data1, label='E-Greedy Cumulative Rewards')
        plt.plot(data2, label='Thompson Sampling Cumulative Rewards')
        plt.title('Comparison of E-Greedy and Thompson Sampling')
        plt.legend()
        plt.show()


def comparison(egreedy_rewards, tsampling_rewards, egreedy_regrets, tsampling_regrets):
    """
    Side-by-side comparison plots for reward and regret.

    :param egreedy_rewards: Epsilon-Greedy cumulative rewards.
    :param tsampling_rewards: Thompson Sampling cumulative rewards.
    :param egreedy_regrets: Epsilon-Greedy cumulative regrets.
    :param tsampling_regrets: Thompson Sampling cumulative regrets.
    """
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(egreedy_rewards, label='Epsilon-Greedy Rewards')
    plt.plot(tsampling_rewards, label='Thompson Sampling Rewards')
    plt.title('Cumulative Rewards Comparison')
    plt.xlabel('Trials')
    plt.ylabel('Cumulative Rewards')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(egreedy_regrets, label='Epsilon-Greedy Regrets')
    plt.plot(tsampling_regrets, label='Thompson Sampling Regrets')
    plt.title('Cumulative Regrets Comparison')
    plt.xlabel('Trials')
    plt.ylabel('Cumulative Regrets')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    k = 4
    rewards = [1, 2, 3, 4]
    n_trials = 20000

    logger.info("Starting experiments")

    e_greedy = EpsilonGreedy(k, rewards)
    thompson = ThompsonSampling(k, rewards)

    e_greedy.experiment(n_trials)
    thompson.experiment(n_trials)

    e_greedy.report("Epsilon-Greedy")
    thompson.report("Thompson Sampling")

    vis = Visualization()
    vis.plot1(e_greedy.cumulative_rewards)
    vis.plot2(e_greedy.cumulative_rewards, thompson.cumulative_rewards)

    comparison(
        e_greedy.cumulative_rewards, thompson.cumulative_rewards,
        e_greedy.cumulative_regrets, thompson.cumulative_regrets
    )

    logger.info(f"Saving CSV to: {os.getcwd()}")
    logger.info("Experiments completed")

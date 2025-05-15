import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sklearn.preprocessing import RobustScaler
from utils import calculate_sma, calculate_rsi, calculate_macd
from datetime import datetime

class TradingEnv(gym.Env):
    def __init__(self, data, window_size=100, initial_balance=10000):
        super(TradingEnv, self).__init__()

        self.base_data = data.copy()
        self.data = self._add_indicators(data)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.action_space = gym.spaces.Discrete(6)
        num_features = len(self.data.drop(columns=['timestamp']).columns)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, num_features), dtype=np.float32)

        self.reset()

    def _add_indicators(self, df):
        df['SMA'] = calculate_sma(df, 14)
        df['RSI'] = calculate_rsi(df, 14)
        df['MACD'], df['MACD_signal'] = calculate_macd(df, fastperiod=12, slowperiod=26, signalperiod=9)
        scaler = RobustScaler()
        cols = ['open', 'high', 'low', 'close', 'volume', 'SMA', 'RSI', 'MACD', 'MACD_signal']
        df[cols] = scaler.fit_transform(df[cols])
        return df.dropna().fillna(0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.net_worth = self.initial_balance
        self.peak_net_worth = self.initial_balance
        self.open_position = False
        self.last_action_step = 0
        return self._next_observation(), {}

    def _next_observation(self):
        start = max(0, self.current_step - self.window_size + 1)
        end = self.current_step + 1
        obs = self.data.drop(columns=['timestamp']).iloc[start:end]
        if len(obs) < self.window_size:
            padding = self.data.drop(columns=['timestamp']).iloc[:self.window_size - len(obs)].values
            obs = np.vstack((padding, obs))
        return obs.astype(np.float32)

    def step(self, action):
        terminated = False
        truncated = False
        penalty = 0
        price = self.base_data.iloc[self.current_step + self.window_size]['close']

        if self.open_position and action in [1, 2, 3, 4]:
            penalty -= 0.1

        if action == 1 and not self.open_position:  # Buy full
            self.position = self.balance / price
            self.balance = 0
            self.open_position = True

        elif action == 2 and not self.open_position:  # Buy partial
            self.position = (self.balance / 2) / price
            self.balance /= 2
            self.open_position = True

        elif action == 3 and not self.open_position:  # Sell short full
            self.position = -self.balance / price
            self.balance += abs(self.position) * price
            self.open_position = True

        elif action == 4 and not self.open_position:  # Sell short partial
            self.position = -(self.balance / 2) / price
            self.balance += abs(self.position) * price
            self.open_position = True

        elif action == 5 and self.open_position:  # Close position
            self.balance += self.position * price
            self.position = 0
            self.open_position = False
        elif action == 5:
            penalty -= 0.1

        prev_worth = self.net_worth
        self.net_worth = self.balance + self.position * price
        self.peak_net_worth = max(self.peak_net_worth, self.net_worth)
        profit = self.net_worth - prev_worth
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth

        reward = profit / self.initial_balance - drawdown * 0.05 + penalty

        self.current_step += 1
        if self.current_step + self.window_size >= len(self.data):
            truncated = True

        return self._next_observation(), reward, terminated, truncated, {}

def evaluate_agent(model, env, episodes=1):
    all_rewards = []
    all_net_worths = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        rewards = []
        net_worths = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            net_worths.append(env.net_worth)
        all_rewards.append(np.sum(rewards))
        all_net_worths.append(net_worths)
    return all_rewards, all_net_worths

def plot_performance(net_worths):
    plt.figure(figsize=(12, 6))
    for nw in net_worths:
        plt.plot(nw)
    plt.title("Evolução do Patrimônio Líquido por Episódio")
    plt.xlabel("Passos")
    plt.ylabel("Patrimônio Líquido")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('BTC_USDT_1m_historical_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    env = TradingEnv(df, window_size=100)
    vec_env = DummyVecEnv([lambda: Monitor(env)])

    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, n_steps=512, batch_size=64, gae_lambda=0.95, gamma=0.99)
    model.learn(total_timesteps=50000)
    model.save("ppo_trading_agent")

    print("Avaliando o agente PPO...")
    rewards, net_worths = evaluate_agent(model, env, episodes=1)
    print(f"Reward total do episódio: {rewards[0]:.2f}")
    plot_performance(net_worths)

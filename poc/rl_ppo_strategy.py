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
    def __init__(self, data, window_size=100, initial_balance=1000, leverage=5, fee=0.0004, stop_loss=0.05, take_profit=0.1):
        super(TradingEnv, self).__init__()

        self.base_data = data.copy()
        self.data = self._add_indicators(data)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.fee = fee
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        self.action_space = gym.spaces.Discrete(4)  # 0: Hold, 1: Long, 2: Short, 3: Close
        num_features = len(self.data.drop(columns=['timestamp']).columns) + 4 # 4 features: balance, position, entry_price, open_flag
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, num_features), dtype=np.float32)

        self.reset()

    def _add_indicators(self, df):
        df['SMA'] = calculate_sma(df, 14)
        df['RSI'] = calculate_rsi(df, 14)
        df['MACD'], df['MACD_signal'] = calculate_macd(df)
        scaler = RobustScaler()
        cols = ['open', 'high', 'low', 'close', 'volume', 'SMA', 'RSI', 'MACD', 'MACD_signal']
        df[cols] = scaler.fit_transform(df[cols])
        return df.fillna(0).dropna()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.net_worth = self.initial_balance
        self.peak_net_worth = self.initial_balance
        self.open_position = False
        self.trade_history = []
        return self._next_observation(), {}

    def _next_observation(self):
        start = max(0, self.current_step - self.window_size + 1)
        end = self.current_step + 1
        obs = self.data.drop(columns=['timestamp']).iloc[start:end]
        
        if len(obs) < self.window_size:
            padding = self.data.drop(columns=['timestamp']).iloc[start+1:self.window_size - len(obs) + 1].values
            # padding = self.data.drop(columns=['timestamp']).iloc[:self.window_size - len(obs)].values
            obs = np.vstack((obs, padding))
        else:
            obs = obs.values
        
        # Variáveis adicionais normalizadas
        balance_norm = self.balance / 100_000 # supondo máx 100k
        position_norm = self.position / 1 # supondo máx ~1 BTC/USDT
        entry_price_norm = self.entry_price / 100_000 # supondo máx 100k
        open_flag = float(self.open_position)
        
        extras = np.array([[balance_norm, position_norm, entry_price_norm, open_flag]] * self.window_size)
        obs = np.hstack([obs, extras])
        
        return obs.astype(np.float32)

    def _calculate_liquidation_price(self, entry_price, position):
        # Para margem isolada com Tiered Margin (até $100k com alavancagem de até 5x)
        # Usando margem de manutenção de 2.5%
        maintenance_margin_rate = 0.025
        initial_margin_rate = 1 / self.leverage

        if position > 0:  # Long
            return entry_price * (1 - (initial_margin_rate - maintenance_margin_rate))
        elif position < 0:  # Short
            return entry_price * (1 + (initial_margin_rate - maintenance_margin_rate))
        return 0

    def step(self, action):
        terminated = False
        truncated = False
        penalty = 0
        price = self.base_data.iloc[self.current_step + self.window_size]['close']

        def apply_fee(value):
            return abs(value) * self.fee

        if action == 1 and not self.open_position:  # Long
            self.position = (self.balance * self.leverage) / price
            self.entry_price = price
            fee = apply_fee(self.position * price)
            self.balance -= fee
            self.open_position = True
            self.trade_history.append((self.current_step, 'long', price))

        elif action == 2 and not self.open_position:  # Short
            self.position = -(self.balance * self.leverage) / price
            self.entry_price = price
            fee = apply_fee(abs(self.position) * price)
            self.balance -= fee
            self.open_position = True
            self.trade_history.append((self.current_step, 'short', price))

        elif action == 3 and self.open_position:  # Close
            profit = self.position * (price - self.entry_price)
            fee = apply_fee(abs(self.position) * price)
            self.balance += profit - fee
            self.trade_history.append((self.current_step, 'close', price))
            self.position = 0
            self.open_position = False

        elif (action == 3 and not self.open_position) or (action in [1, 2] and self.open_position):
            penalty -= 0.02

        # Liquidation
        liquidation_price = self._calculate_liquidation_price(self.entry_price, self.position)
        if self.open_position:
            if (self.position > 0 and price <= liquidation_price) or (self.position < 0 and price >= liquidation_price):
                self.trade_history.append((self.current_step, 'liquidated', price))
                self.position = 0
                self.balance = 0
                terminated = True

            # Stop-loss / Take-profit
            pct_change = (price - self.entry_price) / self.entry_price if self.position > 0 else (self.entry_price - price) / self.entry_price
            if pct_change <= -self.stop_loss or pct_change >= self.take_profit:
                profit = self.position * (price - self.entry_price)
                fee = apply_fee(abs(self.position) * price)
                self.balance += profit - fee
                self.trade_history.append((self.current_step, 'sl/tp', price))
                self.position = 0
                self.open_position = False

        self.net_worth = self.balance + self.position * (price - self.entry_price if self.open_position else 0)
        self.peak_net_worth = max(self.peak_net_worth, self.net_worth)

        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth
        
        # 1. Lucro flutuante por passo
        if self.open_position:
            unrealized_profit = self.position * (price - self.entry_price)
        else:
            unrealized_profit = 0

        reward = unrealized_profit / (self.initial_balance * 0.1)  # normalizado (10% da carteira)

        # Ficar muito tempo em uma posição pode ser ruim
        last_action = self.trade_history[-1][0] if len(self.trade_history) > 0 else 0
        if self.current_step - last_action > 10:
            penalty -= min(0.01, (self.current_step - last_action - 10) * 0.001)

        # 2. Recompensa ao fechar a operação
        if action == 3 and not terminated:
            realized_profit = self.balance - self.initial_balance
            reward += realized_profit / (self.initial_balance * 0.1)  # bônus proporcional

        # 3. Penalização por liquidação
        if terminated:
            reward = -1  # punição clara

        # 4. Penalidades por ações inválidas (mantido)
        reward += penalty

        # 5. Penalização leve por drawdown exagerado
        reward -= drawdown * 0.02  # peso menor

        self.current_step += 1
        if self.current_step + self.window_size >= len(self.data):
            truncated = True

        return self._next_observation(), reward, terminated, truncated, {}

def evaluate_agent(model: PPO, env: TradingEnv, episodes=1):
    all_rewards = []
    all_net_worths = []
    all_histories = []
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
            if env.current_step % 100 == 0:
                print(f"Passo: {env.current_step}, Patrimônio Líquido: {env.net_worth:.2f}, Ação: {action}, Recompensa: {reward:.2f}")
        all_rewards.append(np.sum(rewards))
        all_net_worths.append(net_worths)
        all_histories.append(env.trade_history)
    return all_rewards, all_net_worths, all_histories

def plot_performance(net_worths):
    plt.figure(figsize=(12, 6))
    for nw in net_worths:
        plt.plot(nw)
    plt.title("Evolução do Patrimônio Líquido por Episódio")
    plt.xlabel("Passos")
    plt.ylabel("Patrimônio Líquido")
    plt.grid()
    plt.show()

def plot_trades(base_data, trade_history, window_size):
    closes = base_data['close'].iloc[window_size:].reset_index(drop=True)
    plt.figure(figsize=(14, 6))
    plt.plot(closes, label="Preço", color='blue')
    
    for step, action, price in trade_history:
        if action == 'long':
            plt.scatter(step, price, marker='^', color='green', label='Long')
        elif action == 'short':
            plt.scatter(step, price, marker='v', color='red', label='Short')
        elif action == 'close':
            plt.scatter(step, price, marker='x', color='black', label='Close')
        elif action == 'liquidated':
            plt.scatter(step, price, marker='o', color='orange', label='Liquidated')
        elif action == 'sl/tp':
            plt.scatter(step, price, marker='*', color='purple', label='SL/TP')

    plt.title("Operações Executadas")
    plt.xlabel("Passos")
    plt.ylabel("Preço")
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('BTC_USDT_1m_historical_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    env = TradingEnv(df, window_size=100, )
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    
    model_file = "ppo_trading_agent.zip"
    if os.path.exists(model_file):
        print("Carregando o modelo PPO existente...")
        model = PPO.load(model_file, env=vec_env)
    else:
        model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, gae_lambda=0.95, gamma=0.95, ent_coef=0.01, clip_range=0.2, policy_kwargs={"net_arch": [512, 128, 32]})
    model.learn(total_timesteps=50000)
    model.save("ppo_trading_agent")

    print("Avaliando o agente PPO...")
    rewards, net_worths, histories = evaluate_agent(model, env, episodes=1)
    print(f"Reward total do episódio: {rewards[0]:.2f}")
    plot_performance(net_worths)
    plot_trades(env.base_data, histories[0], env.window_size)
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import pandas as pd
import logging
from sklearn.preprocessing import RobustScaler
from utils import calculate_sma, calculate_rsi, calculate_macd
import matplotlib.dates as mdates
from torch import nn

# Configurar logging detalhado
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')#, filename='trading.log', filemode='w')

class TradingEnvironment(gym.Env):
    """Ambiente de aprendizado por reforço para simular o mercado financeiro."""
    def __init__(self, data, window_size=100, initial_balance=10000):
        """
        Inicializa o ambiente de negociação.
        :param data: DataFrame com os dados históricos de preços.
        :param window_size: Tamanho da janela deslizante para observação.
        :param initial_balance: Saldo inicial para o agente.
        """

        super(TradingEnvironment, self).__init__()
        self.base_data = data.copy()
        self.data = self._add_technical_indicators(data)
        self.window_size = window_size  # Tamanho da janela deslizante
        self.current_step = 0
        self.initial_balance = initial_balance  # Saldo inicial
        self.balance = initial_balance # Saldo inicial
        self.position = 0  # Posição atual (quantidade de ativos)
        self.net_worth = self.balance
        self.open_position = False # Indica se há uma posição aberta
        self.last_buy_step = None  # Último passo em que houve uma compra

        # Espaço de ações expandido: 0 = manter, 1 = comprar integral, 2 = comprar parcial, 3 = vender integral, 4 = vender parcial, 5 = encerrar operação
        self.action_space = gym.spaces.Discrete(6)

        # Ajustar o espaço de observação para refletir o número correto de colunas
        num_features = len(self.data.drop(columns=['timestamp']).columns)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, num_features), dtype=np.float32
        )

    def _add_technical_indicators(self, data):
        """Adiciona indicadores técnicos aos dados."""
        data['SMA'] = calculate_sma(data, window=14)  # Média móvel simples
        data['RSI'] = calculate_rsi(data, window=14)  # Índice de Força Relativa
        data['MACD'], data['MACD_signal'] = calculate_macd(data, fastperiod=12, slowperiod=26, signalperiod=9)  # MACD

        # Normalizar os dados usando RobustScaler
        scaler = RobustScaler()
        data[['open', 'high', 'low', 'close', 'volume', 'SMA', 'RSI', 'MACD', 'MACD_signal']] = scaler.fit_transform(
            data[['open', 'high', 'low', 'close', 'volume', 'SMA', 'RSI', 'MACD', 'MACD_signal']]
        )
        
        # Preencher valores NaN com 0
        data['SMA'] = data['SMA'].fillna(0)
        data['RSI'] = data['RSI'].fillna(0)
        data['MACD'] = data['MACD'].fillna(0)
        data['MACD_signal'] = data['MACD_signal'].fillna(0)

        return data.dropna()

    def reset(self):
        """Reinicia o ambiente para um novo episódio."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.net_worth = 0
        self.open_position = False
        self.last_buy_step = None
        self.peak_balance = self.initial_balance  # Patrimônio líquido máximo alcançado
        return self._next_observation()

    def _next_observation(self):
        """Obtém o próximo estado do ambiente."""
        start_index = max(0, self.current_step - self.window_size + 1)
        end_index = self.current_step + 1
        window_data = self.data.drop(columns=['timestamp']).iloc[start_index:end_index]

        # Preencher com os dados mais antigos disponíveis se a janela for menor que o tamanho esperado
        if len(window_data) < self.window_size:
            padding = self.data.drop(columns=['timestamp']).iloc[:self.window_size - len(window_data)].values
            window_data = np.vstack((padding, window_data))

        return window_data.astype(np.float32)

    def step(self, action):
        """Executa uma ação no ambiente."""
        current_price = self.base_data.iloc[self.current_step + self.window_size]['close'].item()
        
        # logging.info(f"Ação tomada: {action} na etapa {self.current_step}")
        
        penalty = 0  # Penalização inicial
        
        # Penalização por tentar iniciar nova operação sem encerrar a anterior
        if self.open_position and action in [1, 2, 3, 4]:
            penalty -= 0.1

        if action == 1:  # Comprar integral
            if not self.open_position and self.balance > 0:
                self.position += self.balance / current_price  # Compra integral
                self.balance = 0
                self.open_position = True
                self.last_buy_step = self.current_step
                # logging.info(f"Compra integral realizada no preço {current_price:.2f}, spep {self.current_step}")

        elif action == 2:  # Comprar parcial
            if not self.open_position and self.balance > 0:
                self.position += (self.balance / 2) / current_price  # Compra parcial (50% do saldo)
                self.balance /= 2
                self.open_position = True
                self.last_buy_step = self.current_step
                # logging.info(f"Compra parcial realizada no preço {current_price:.2f}, step {self.current_step}")

        elif action == 3:  # Vender integral (Short Selling)
            if not self.open_position and self.balance > 0:
                self.position -= self.balance / current_price  # Abrir posição vendida integral
                self.balance -= self.position * current_price  # Vender a posição
                self.open_position = True
                self.last_buy_step = self.current_step
                # logging.info(f"Venda integral (short) realizada no preço {current_price:.2f}, step {self.current_step}")

        elif action == 4:  # Vender parcial (Short Selling)
            if not self.open_position and self.balance > 0:
                self.position -= (self.balance / 2) / current_price  # Abrir posição vendida parcial (50% do saldo)
                self.balance -= self.position * current_price  # Vender a posição
                self.open_position = True
                self.last_buy_step = self.current_step
                # logging.info(f"Venda parcial (short) realizada no preço {current_price:.2f}, step {self.current_step}")

        elif action == 5:  # Encerrar operação
            if self.open_position:
                # Penalização por encerrar operação muito cedo
                if (self.current_step - self.last_buy_step) < 2:
                    penalty -= 0.1
                
                operation = 'buy ' if self.position > 0 else 'sell'
                self.balance += (self.position * current_price)  # Encerrar posição
                self.position = 0
                self.open_position = False
                self.last_buy_step = self.current_step
                logging.info(f"Operação ({operation}) encerrada, step {self.current_step}, balance {self.balance:.2f}")
            else:
                penalty -= 0.1  # Penalização por tentar encerrar sem operação aberta

        # Penalização dinâmica baseada no tempo sem realizar ações
        steps_since_last_action = self.current_step - (self.last_buy_step or 0)
        if steps_since_last_action > 100:
            penalty -= min(steps_since_last_action * 0.00001, 0.05)  # Penalidade máxima reduzida para 0.05

        previous_net_worth = self.net_worth
        self.net_worth = self.balance - self.initial_balance + (self.position * current_price)
        profit = self.net_worth - previous_net_worth

        done = self.current_step >= len(self.data) - 1

        reward = 0
        if done and self.net_worth > self.initial_balance:
            reward = 1  # lucro
        elif done:
            reward = -1  # prejuízo
        else:
            reward = profit / self.initial_balance
        reward += penalty

        # Penalizar por risco excessivo
        self.peak_balance = max(self.peak_balance, (self.balance + (self.position * current_price)))
        drawdown = ((self.peak_balance - self.net_worth) / self.peak_balance) if self.peak_balance > 0 else 1
        reward -= drawdown * 0.05

        # logging.info(f"Passo {self.current_step}: Ação {action}, Preço {current_price:.2f}, Saldo {self.balance:.2f}, Posição {self.position:.4f}, Patrimônio Líquido {self.net_worth:.2f}")

        # Verificar se o patrimônio líquido é válido
        if not np.isfinite(self.net_worth):
            logging.error(f"Patrimônio líquido inválido detectado no passo {self.current_step}: {self.net_worth}")
            return self._next_observation(), -100, True, {}  # Penalizar e encerrar o episódio

        self.current_step += 1

        return self._next_observation(), reward, done, {}

def plot_training_results(data, rewards, net_worths):
    """Plota os resultados do treinamento para avaliação."""
    plt.figure(figsize=(14, 7))

    # Gráfico de recompensas
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label='Recompensas Acumuladas', color='blue')
    plt.title('Recompensas Acumuladas Durante o Treinamento')
    plt.xlabel('Passos')
    plt.ylabel('Recompensa')
    plt.legend()

    # Gráfico de patrimônio líquido
    plt.subplot(2, 1, 2)
    plt.plot(net_worths, label='Patrimônio Líquido', color='green')
    plt.title('Evolução do Patrimônio Líquido')
    plt.xlabel('Passos')
    plt.ylabel('Patrimônio Líquido')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_operations(data, operations: list):
    """Plota os dados históricos junto com as operações realizadas."""
    plt.figure(figsize=(14, 7))

    # Plotar os preços de fechamento
    plt.plot(data['timestamp'], data['close'], label='Preço de Fechamento', color='blue')
    
    operations = pd.DataFrame(operations, columns=['timestamp', 'action', 'price'])
    operations['timestamp'] = pd.to_datetime(operations['timestamp'])
    operations['action'] = operations['action'].astype('category')
    operations['price'] = operations['price'].astype(float)

    # Adicionar operações ao gráfico
    buy_operations = operations[operations['action'] == 'buy']
    sell_operations = operations[operations['action'] == 'sell']
    close_operations = operations[operations['action'] == 'close']
    plt.scatter(buy_operations['timestamp'], buy_operations['price'], color='green', label='Compra', marker='^')
    plt.scatter(sell_operations['timestamp'], sell_operations['price'], color='red', label='Venda', marker='v')
    plt.scatter(close_operations['timestamp'], close_operations['price'], color='gray', label='Encer', marker='o')

    # Formatação do gráfico
    plt.title('Dados Históricos e Operações Realizadas')
    plt.xlabel('Data')
    plt.ylabel('Preço de Fechamento')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def train_rl_agent_with_logging(env: TradingEnvironment, model_path='rl_agent.zip', train_steps=10000):
    """Treina um agente de RL usando DQN e registra recompensas e ações."""
    policy_kwargs = dict(
        net_arch=[512, 128, 64, 32],
        activation_fn=nn.Tanh
    )

    if os.path.exists(model_path):
        print(f"Modelo encontrado em {model_path}. Carregando...")
        model = DQN.load(model_path, env=env)  # Carregar modelo salvo, se existir
    else:
        model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0, learning_rate=0.0001, 
                    buffer_size=100000, learning_starts=1000, batch_size=64, gamma=0.99, 
                    train_freq=4, target_update_interval=1000)

    model.learning_rate = 0.0001
    model.train_freq = 10
    model._setup_model()

    i = np.random.randint(0, len(env.data) - env.window_size - train_steps)
    env.base_data = env.base_data.iloc[i:i + (train_steps + env.window_size)]
    env.data = env.data.iloc[i:i + (train_steps + env.window_size)]
    env.reset()

    rewards = []
    net_worths = []
    operations =  []

    def callback(_locals, _globals):
        """Callback para registrar recompensas e ações durante o treinamento."""
        if 'rewards' in _locals:
            rewards.append(_locals['rewards'])
            net_worths.append(env.net_worth)
        
        if 'actions' in _locals:
            action = _locals['actions']
            # Registrar operações
            if (action == 1 or action == 2) and (len(operations) == 0 or operations[-1][1] == 'close'):  # Compra
                operations.append((env.base_data.iloc[env.current_step]['timestamp'], 'buy', env.base_data.iloc[env.current_step]['close']))
            elif (action == 3 or action == 4) and (len(operations) == 0 or operations[-1][1] == 'close'):  # Venda
                operations.append((env.base_data.iloc[env.current_step]['timestamp'], 'sell', env.base_data.iloc[env.current_step]['close']))
            elif action == 5 and len(operations) > 0 and operations[-1][1] != 'close':  # Encerrar operação
                operations.append((env.base_data.iloc[env.current_step]['timestamp'], 'close', env.base_data.iloc[env.current_step]['close']))
        return True

    # Aprendizado contínuo
    print("Iniciando o aprendizado contínuo...")
    model = model.learn(total_timesteps=train_steps, callback=callback)
    print("Aprendizado contínuo concluído. Iniciando o treinamento...")

    # Salvar o modelo treinado
    print(f"Salvando o modelo treinado em {model_path}...")
    model.save(model_path)

    # Plotar os resultados
    plot_training_results(env.data, rewards, net_worths)
    plot_operations(env.base_data[:train_steps], operations)

    return model

def test_rl_agent(env: TradingEnvironment, model_path='rl_agent.zip'):
    """Testa um agente de RL treinado no ambiente."""
    # Carregar o modelo treinado
    print(f"Carregando o modelo treinado de {model_path}...")
    model = DQN.load(model_path, env=env)

    # Testar o agente com uma amostragem de 10% dos dados com posição inicial aleatoria
    i = np.random.randint(0, len(env.data) - (env.window_size * 100))
    env.base_data = env.base_data.iloc[i:i + (env.window_size * 100)]
    env.data = env.data.iloc[i:i + (env.window_size * 100)]
    obs = env.reset()

    rewards = []
    net_worths = []
    operations = []  # Para armazenar as operações realizadas
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        net_worths.append(env.net_worth)

        # Registrar operações
        if (action == 1 or action == 2) and (len(operations) == 0 or operations[-1][1] == 'close'):  # Compra
            operations.append((env.current_step + env.window_size, 'buy', env.base_data.iloc[env.current_step]['close']))
        elif (action == 3 or action == 4) and (len(operations) == 0 or operations[-1][1] == 'close'):  # Venda
            operations.append((env.current_step + env.window_size, 'sell', env.base_data.iloc[env.current_step]['close']))
        elif action == 5 and len(operations) > 0 and operations[-1][1] != 'close':  # Encerrar operação
            operations.append((env.current_step + env.window_size, 'close', env.base_data.iloc[env.current_step]['close']))

    # Plotar os resultados
    plot_training_results(env.data, rewards, net_worths)
    plot_operations(env.base_data, operations)

if __name__ == "__main__":
    # Carregar os dados históricos
    data = pd.read_csv('BTC_USDT_1m_historical_data.csv')
    
    # Converter timestamps para formato de data
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Criar o ambiente
    env = TradingEnvironment(data, window_size=300)

    # Treinar o agente com logging
    train_rl_agent_with_logging(env, train_steps=30000)
    # Testar o agente treinado
    # test_rl_agent(env)

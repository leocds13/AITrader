import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import logging
from sklearn.preprocessing import RobustScaler
from utils import calculate_sma, calculate_rsi, calculate_macd
import matplotlib.dates as mdates

# Configurar logging detalhado
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class TradingEnvironment(gym.Env):
    """Ambiente de aprendizado por reforço para simular o mercado financeiro."""
    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        self.data = self._add_technical_indicators(data)
        self.current_step = 0
        self.balance = 10000  # Saldo inicial
        self.position = 0  # Posição atual (quantidade de ativos)
        self.net_worth = self.balance
        self.open_position = False  # Indica se há uma operação aberta
        self.last_buy_step = None  # Último passo em que houve uma compra

        # Espaço de ações expandido: 0 = manter, 1 = comprar integral, 2 = comprar parcial, 3 = vender integral, 4 = vender parcial, 5 = encerrar operação
        self.action_space = gym.spaces.Discrete(6)

        # Ajustar o espaço de observação para refletir o número correto de colunas
        num_features = len(self.data.drop(columns=['timestamp']).columns)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )

    def _add_technical_indicators(self, data):
        """Adiciona indicadores técnicos aos dados."""
        data['SMA'] = calculate_sma(data, window=14)  # Média móvel simples
        data['RSI'] = calculate_rsi(data, window=14)  # Índice de Força Relativa
        data['MACD'], data['MACD_signal'] = calculate_macd(data, fastperiod=12, slowperiod=26, signalperiod=9)  # MACD

        # Garantir que os dados não contenham valores inválidos antes da normalização
        data = data.dropna(subset=['close', 'SMA', 'RSI', 'MACD', 'MACD_signal'])
        data = data[data['close'] > 0]

        # Normalizar os dados usando RobustScaler
        scaler = RobustScaler()
        data[['close', 'SMA', 'RSI', 'MACD', 'MACD_signal']] = scaler.fit_transform(
            data[['close', 'SMA', 'RSI', 'MACD', 'MACD_signal']]
        )

        return data.dropna()

    def reset(self):
        """Reinicia o ambiente para um novo episódio."""
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.net_worth = self.balance
        self.open_position = False
        self.last_buy_step = None
        return self._next_observation()

    def _next_observation(self):
        """Obtém o próximo estado do ambiente."""
        # Excluir a coluna 'timestamp' e garantir que os dados sejam numéricos
        return self.data.drop(columns=['timestamp']).iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        """Executa uma ação no ambiente."""
        current_price = self.data.iloc[self.current_step]['close']

        penalty = 0  # Penalização inicial

        if action == 1:  # Comprar integral
            if not self.open_position and self.balance > 0:
                self.position += self.balance / current_price  # Compra integral
                self.balance = 0
                self.open_position = True
                self.last_buy_step = self.current_step
                logging.info(f"Compra integral realizada no preço {current_price:.2f}")
            else:
                penalty -= 10  # Penalização por tentar comprar sem encerrar operação anterior

        elif action == 2:  # Comprar parcial
            if not self.open_position and self.balance > 0:
                self.position += (self.balance / 2) / current_price  # Compra parcial (50% do saldo)
                self.balance /= 2
                self.open_position = True
                self.last_buy_step = self.current_step
                logging.info(f"Compra parcial realizada no preço {current_price:.2f}")
            else:
                penalty -= 10  # Penalização por tentar comprar sem encerrar operação anterior

        elif action == 3:  # Vender integral (Short Selling)
            if not self.open_position:  # Permitir abrir uma operação de venda se não houver operação aberta
                if self.balance > 0:
                    self.position -= self.balance / current_price  # Abrir posição vendida integral
                    self.open_position = True
                    self.last_buy_step = self.current_step
                    logging.info(f"Venda integral (short) realizada no preço {current_price:.2f}")
                else:
                    penalty -= 10  # Penalização por tentar vender sem saldo suficiente
                    # logging.warning("Tentativa de venda integral sem saldo suficiente.")
            else:
                penalty -= 10  # Penalização por tentar iniciar uma venda sem encerrar a operação anterior

        elif action == 4:  # Vender parcial (Short Selling)
            if not self.open_position:  # Permitir abrir uma operação de venda parcial se não houver operação aberta
                if self.balance > 0:
                    self.position -= (self.balance / 2) / current_price  # Abrir posição vendida parcial (50% do saldo)
                    self.balance /= 2
                    self.open_position = True
                    self.last_buy_step = self.current_step
                    logging.info(f"Venda parcial (short) realizada no preço {current_price:.2f}")
                else:
                    penalty -= 10  # Penalização por tentar vender sem saldo suficiente
                    # logging.warning("Tentativa de venda parcial sem saldo suficiente.")
            else:
                penalty -= 10  # Penalização por tentar iniciar uma venda sem encerrar a operação anterior

        elif action == 5:  # Encerrar operação
            if self.open_position:
                self.balance += self.position * current_price
                self.position = 0
                self.open_position = False
                logging.info(f"Operação encerrada no preço {current_price:.2f}")
            else:
                penalty -= 5  # Penalização por tentar encerrar sem operação aberta

        # Penalização dinâmica baseada no tempo sem lucro ou ação
        if self.open_position:
            steps_since_last_action = self.current_step - self.last_buy_step
            if steps_since_last_action > 10:
                penalty -= min(steps_since_last_action * 0.1, 10)  # Limitar penalidade máxima a 10

        # Penalização por inatividade geral (sem abrir ou fechar posições por muito tempo)
        if not self.open_position and self.current_step > 10:
            penalty -= min((self.current_step - (self.last_buy_step or 0)) * 0.05, 5)  # Penalidade máxima limitada a 5

        # Garantir que operações sejam encerradas antes de iniciar novas
        if self.open_position and action in [1, 2, 3, 4]:
            penalty -= 10  # Penalização por tentar iniciar nova operação sem encerrar a anterior
            # logging.warning("Tentativa de iniciar nova operação sem encerrar a anterior.")

        self.net_worth = self.balance + self.position * current_price

        # Verificar se o patrimônio líquido é válido
        if not np.isfinite(self.net_worth):
            logging.error(f"Patrimônio líquido inválido detectado no passo {self.current_step}: {self.net_worth}")
            return self._next_observation(), -100, True, {}  # Penalizar e encerrar o episódio

        self.current_step += 1

        done = self.current_step >= len(self.data) - 1

        # Recompensa ajustada para incluir retorno percentual e redução de risco
        reward = (self.net_worth - 10000) / 10000 + penalty  # Normalizar o lucro

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

def plot_operations(data, operations):
    """Plota os dados históricos junto com as operações realizadas."""
    plt.figure(figsize=(14, 7))

    # Converter timestamps para formato de data
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Plotar os preços de fechamento
    plt.plot(data['timestamp'], data['close'], label='Preço de Fechamento', color='blue')

    # Adicionar operações ao gráfico
    for op in operations:
        step, action, price = op
        if action == 'buy':
            plt.scatter(data['timestamp'].iloc[step], price, color='green', label='Compra', marker='^')
        elif action == 'sell':
            plt.scatter(data['timestamp'].iloc[step], price, color='red', label='Venda', marker='v')

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

def train_rl_agent_with_logging(env, model_path='rl_agent.zip'):
    """Treina um agente de RL usando DQN e registra recompensas e ações."""
    model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.0001, buffer_size=50000, learning_starts=1000, batch_size=32, gamma=0.99, train_freq=4, target_update_interval=1000)

    rewards = []
    net_worths = []

    obs = env.reset()
    for step in range(10000):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)

        rewards.append(reward)
        net_worths.append(env.net_worth)

        if done:
            obs = env.reset()

    # Salvar o modelo treinado
    print(f"Salvando o modelo treinado em {model_path}...")
    model.save(model_path)

    # Plotar os resultados
    plot_training_results(env.data, rewards, net_worths)

    return model

def test_rl_agent(env, model_path='rl_agent.zip'):
    """Testa um agente de RL treinado no ambiente."""
    # Carregar o modelo treinado
    print(f"Carregando o modelo treinado de {model_path}...")
    model = DQN.load(model_path, env=env)

    # Testar o agente
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
        if action == 1 or action == 2:  # Compra
            operations.append((env.current_step, 'buy', env.data.iloc[env.current_step]['close']))
        elif action == 3 or action == 4:  # Venda
            operations.append((env.current_step, 'sell', env.data.iloc[env.current_step]['close']))

    # Plotar os resultados
    plot_training_results(env.data, rewards, net_worths)
    plot_operations(env.data, operations)

if __name__ == "__main__":
    import pandas as pd

    # Carregar os dados históricos
    data = pd.read_csv('BTC_USDT_1m_historical_data.csv')

    # Criar o ambiente
    env = TradingEnvironment(data)

    # Treinar o agente com logging
    model = train_rl_agent_with_logging(env)

    # Testar o agente
    # test_rl_agent(env)
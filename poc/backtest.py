import pandas as pd
import matplotlib.pyplot as plt
from ai_strategy import AIStrategy

class Backtest:
    def __init__(self, data):
        self.data = data
        self.data['signal'] = 0  # Inicializa os sinais
        self.trades = []  # Lista para armazenar operações realizadas

    def set_strategy(self, strategy):
        """Define a estratégia a ser usada no backtest."""
        self.strategy = strategy

    def apply_strategy(self, **kwargs):
        """Aplica a estratégia definida aos dados históricos."""
        if not hasattr(self, 'strategy'):
            raise ValueError("Nenhuma estratégia foi definida.")
        self.data = self.strategy(self.data, **kwargs)

    def simulate_trades(self):
        """Simula operações de compra e venda com base nos sinais gerados."""
        position = 0  # 1 para comprado, -1 para vendido, 0 para neutro
        for i in range(len(self.data)):
            signal = self.data['signal'].iloc[i]
            price = self.data['close'].iloc[i]
            if signal == 1 and position <= 0:  # Compra
                self.trades.append({'action': 'buy', 'price': price, 'index': i})
                position = 1
            elif signal == -1 and position >= 0:  # Venda
                self.trades.append({'action': 'sell', 'price': price, 'index': i})
                position = -1

    def calculate_performance(self):
        """Calcula métricas de desempenho do backtest."""
        if not self.trades:
            raise ValueError("Nenhuma operação foi realizada.")

        total_profit = 0
        for i in range(1, len(self.trades), 2):  # Avalia pares de compra e venda
            buy = self.trades[i - 1]
            sell = self.trades[i]
            if buy['action'] == 'buy' and sell['action'] == 'sell':
                total_profit += sell['price'] - buy['price']

        return {
            'total_trades': len(self.trades) // 2,
            'total_profit': total_profit,
            'average_profit_per_trade': total_profit / (len(self.trades) // 2) if len(self.trades) >= 2 else 0
        }

    def plot_results(self):
        """Plota os resultados do backtest."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['timestamp'], self.data['close'], label='Close Price')
        buy_signals = [trade for trade in self.trades if trade['action'] == 'buy']
        sell_signals = [trade for trade in self.trades if trade['action'] == 'sell']
        plt.scatter(
            [self.data['timestamp'].iloc[trade['index']] for trade in buy_signals],
            [trade['price'] for trade in buy_signals],
            marker='^', color='green', label='Buy Signal'
        )
        plt.scatter(
            [self.data['timestamp'].iloc[trade['index']] for trade in sell_signals],
            [trade['price'] for trade in sell_signals],
            marker='v', color='red', label='Sell Signal'
        )
        plt.legend()
        plt.title('Backtest Results')
        plt.show()

if __name__ == "__main__":
    # Carregar os dados históricos
    data = pd.read_csv('BTC_USDT_1m_historical_data.csv')

    # Criar uma instância do Backtest
    backtest = Backtest(data)

    # Criar e configurar a estratégia de IA
    ai_strategy = AIStrategy()
    ai_strategy.train(data)

    def ai_strategy_wrapper(data):
        return ai_strategy.predict(data)

    backtest.set_strategy(ai_strategy_wrapper)

    # Aplicar a estratégia
    backtest.apply_strategy()

    # Simular operações
    backtest.simulate_trades()

    # Calcular desempenho
    performance = backtest.calculate_performance()
    print("Desempenho do Backtest:", performance)

    # Plotar os resultados
    backtest.plot_results()
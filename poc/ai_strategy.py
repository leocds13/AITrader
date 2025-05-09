import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
import joblib
warnings.filterwarnings("ignore")  # Ignorar warnings desnecessários

class AIStrategy:
    def __init__(self, model_path='ai_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Carrega o modelo treinado do disco, se existir."""
        if os.path.exists(self.model_path):
            print(f"Carregando modelo salvo de {self.model_path}...")
            self.model = joblib.load(self.model_path)
        else:
            print("Nenhum modelo salvo encontrado. Criando um novo modelo...")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def save_model(self):
        """Salva o modelo treinado no disco."""
        print(f"Salvando modelo em {self.model_path}...")
        joblib.dump(self.model, self.model_path)

    def preprocess_data(self, data):
        """Prepara os dados para treinamento e previsão."""
        # Garantir que os dados estejam consistentes
        if data.isnull().values.any():
            print("Aviso: Dados contêm valores nulos. Removendo...")
        
        # Criar features baseadas nos dados históricos
        data['return'] = data['close'].pct_change()  # Retorno percentual
        data['volatility'] = data['return'].rolling(window=10).std()  # Volatilidade
        data['momentum'] = data['close'] - data['close'].shift(10)  # Momento

        # Remover valores NaN
        data = data.dropna()

        # Criar a variável alvo (1 para alta, 0 para baixa)
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

        # Features e alvo
        features = ['return', 'volatility', 'momentum']
        X = data[features]
        y = data['target']

        return X, y, data

    def optimize_model(self, X_train, y_train):
        """Ajusta hiperparâmetros do modelo para melhorar o desempenho."""
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        print("Melhores parâmetros encontrados:", grid_search.best_params_)
        self.model = grid_search.best_estimator_

    def train(self, data):
        """Treina o modelo com os dados históricos."""
        X, y, _ = self.preprocess_data(data)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Otimizar o modelo
        self.optimize_model(X_train, y_train)

        # Treinar o modelo
        self.model.fit(X_train, y_train)

        # Salvar o modelo treinado
        self.save_model()

        # Avaliar o modelo
        y_pred = self.model.predict(X_val)
        print("Relatório de Classificação:\n", classification_report(y_val, y_pred))

    def calculate_trading_metrics(self, data):
        """Calcula métricas específicas de trading."""
        total_trades = len(data[data['signal'] != 0])
        profitable_trades = len(data[(data['signal'] == 1) & (data['close'].shift(-1) > data['close'])])
        losing_trades = total_trades - profitable_trades

        sharpe_ratio = (data['return'].mean() / data['return'].std()) * (252 ** 0.5)  # Anualizado

        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'sharpe_ratio': sharpe_ratio
        }

    def predict(self, data):
        """Gera sinais de compra/venda com base nos dados."""
        X, _, data = self.preprocess_data(data)
        data['signal'] = self.model.predict(X)
        data['signal'] = data['signal'].shift(1)  # Ajustar para evitar lookahead bias
        return data

def get_historical_data_filename(symbol, timeframe):
    """Gera o nome do arquivo com base no ativo e timestamp."""
    return f"{symbol.replace('/', '_')}_{timeframe}_historical_data.csv"

if __name__ == "__main__":
    # Definir o ativo e o timeframe
    symbol = 'BTC/USDT'
    timeframe = '1m'

    # Gerar o nome do arquivo
    filename = get_historical_data_filename(symbol, timeframe)

    # Verificar se o arquivo existe
    if not os.path.exists(filename):
        print(f"Arquivo {filename} não encontrado. Coletando dados...")
        from collect_data import collect_historical_data
        import ccxt
        exchange = ccxt.binance()
        since = exchange.parse8601('2023-01-01T00:00:00Z')
        data = collect_historical_data(symbol, timeframe, since)
        data.to_csv(filename, index=False)
        print(f"Dados históricos salvos em {filename}")
    else:
        print(f"Carregando dados do arquivo {filename}")
        data = pd.read_csv(filename)

    # Criar e treinar a estratégia de IA
    ai_strategy = AIStrategy()
    ai_strategy.train(data)

    # Prever sinais
    data_with_signals = ai_strategy.predict(data)

    # Calcular métricas de trading
    metrics = ai_strategy.calculate_trading_metrics(data_with_signals)
    print("Métricas de Trading:", metrics)
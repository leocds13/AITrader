import ccxt
import pandas as pd

def collect_historical_data(symbol, timeframe, since):
    exchange = ccxt.binance() 
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

if __name__ == "__main__":
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1m'
    filename = f"{symbol.replace('/', '_')}_{timeframe}_historical_data.csv"
    final_timestamp = pd.to_datetime('2023-10-01T00:00:00Z').timestamp() * 1000  # Timestamp final em milissegundos
    # Verifica se o arquivo já existe
    try:
        data = pd.read_csv(filename)
        # timestamp do dado mais recente
        last_timestamp = pd.to_datetime(data['timestamp'].iloc[-1])
        print(f"Último timestamp encontrado: {last_timestamp}")
        since = int(last_timestamp.timestamp() * 1000) + 1  # Adiciona 1 milissegundo
    except FileNotFoundError:
        print(f"Arquivo {filename} não encontrado. Coletando dados...")
        since = exchange.parse8601('2023-01-01T00:00:00Z')

    # Coleta dados até o timestamp final
    while since < final_timestamp:
        print(f"Coletando dados desde {since}...")
        data = collect_historical_data(symbol, timeframe, since)
        if data.empty:
            print("Nenhum dado coletado. Parando a coleta.")
            break
        # Atualiza o timestamp de início para o próximo ciclo
        since = int(pd.to_datetime(data['timestamp'].iloc[-1]).timestamp() * 1000) + 1
        # Salva os dados em um arquivo CSV
        data.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)
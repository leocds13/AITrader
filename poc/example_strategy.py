def moving_average_crossover_strategy(data, short_window=10, long_window=50):
    """Estratégia de cruzamento de médias móveis.
    Compra quando a média móvel curta cruza acima da média móvel longa.
    Venda quando a média móvel curta cruza abaixo da média móvel longa.
    """
    data['SMA_short'] = data['close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['close'].rolling(window=long_window).mean()
    data['signal'] = 0
    data.loc[data['SMA_short'] > data['SMA_long'], 'signal'] = 1  # Sinal de compra
    data.loc[data['SMA_short'] <= data['SMA_long'], 'signal'] = -1  # Sinal de venda
    return data

# Para usar esta estratégia, importe-a no arquivo backtest.py e defina-a como a estratégia ativa:
# from example_strategy import moving_average_crossover_strategy
# backtest.set_strategy(moving_average_crossover_strategy)
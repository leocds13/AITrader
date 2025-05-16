import numpy as np
import pandas as pd

def calculate_sma(data, window):
    """Calcula a Média Móvel Simples (SMA)."""
    return data['close'].rolling(window=window).mean()

def calculate_rsi(data, window):
    """Calcula o Índice de Força Relativa (RSI)."""
    delta = data['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fastperiod=12, slowperiod=26, signalperiod=9):
    """Calcula o MACD e a linha de sinal."""
    ema_fast = data['close'].ewm(span=fastperiod, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slowperiod, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd, signal
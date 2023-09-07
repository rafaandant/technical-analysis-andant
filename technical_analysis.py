
"""PROGRAM DONE BY FERNANDO KRIKUN AND RAFAEL ANDANT"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import sys
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, cross


class MaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 50
    n2 = 200

    def init(self):
        # Precompute the two moving averages
        self.ma1 = self.I(self.func_ma1, self.data.Close, self.n1, overlay=True)
        self.ma2 = self.I(self.func_ma2, self.data.Close, self.n2, overlay=True)

    def next(self):
        if cross(self.data.Close, self.ma1):
            self.position.close()
        if crossover(self.ma1, self.ma2) or (
            crossover(self.data.Close, self.ma1) and self.ma1[-1] > self.ma2[-1]
        ):
            self.buy()

        elif crossover(self.ma2, self.ma1) or (
            crossover(self.ma1, self.data.Close) and self.ma1[-1] < self.ma2[-1]
        ):
            self.sell()


class SmaCross(MaCross):
    def func_ma1(self, data, n):
        return pd.Series(data).rolling(n).mean()

    def func_ma2(self, data, n):
        return self.func_ma1(data, n)


class ESmaCross(MaCross):
    def func_ma1(self, data, n):
        return pd.Series(data).ewm(span=n, adjust=False).mean()

    def func_ma2(self, data, n):
        return pd.Series(data).rolling(n).mean()


class Aroon(Strategy):
    n = 14

    def _aroon(self, data, n, up=True):
        aroon_pct = (
            pd.Series(data.High).rolling(n).apply(lambda x: x.argmax() / n)
            if up
            else pd.Series(data.Low).rolling(n).apply(lambda x: x.argmin() / n)
        )
        return 100 * aroon_pct

    def _aroon_osc(self, data, n):
        aroon_up = self._aroon(data, n, True)
        aroon_down = self._aroon(data, n, False)
        return aroon_up - aroon_down

    def init(self):
        self.aroon_up = self.I(
            self._aroon, self.data, self.n, True, name="Aroon Up", plot=False
        )
        self.aroon_down = self.I(
            self._aroon, self.data, self.n, False, name="Aroon Down", plot=False
        )
        self.I(
            lambda data, n: (self._aroon(data, n, True), self._aroon(data, n, False)),
            self.data,
            self.n,
            name="Aroon",
            overlay=False,
        )
        self.aroon_osc = self.I(
            self._aroon_osc, self.data, self.n, name="Aroon Oscilator", overlay=False
        )

    def next(self):
        if cross(self.aroon_up, self.aroon_down):
            self.position.close()

        elif (
            (crossover(self.aroon_up, 70) and self.aroon_down[-1] < 30)
            or (crossover(30, self.aroon_down) and self.aroon_up[-1] > 70)
        ) and not self.position:
            self.buy()

        elif (
            (crossover(self.aroon_down, 70) and self.aroon_up[-1] < 30)
            or (crossover(30, self.aroon_up) and self.aroon_down[-1] > 70)
        ) and not self.position:
            self.sell()


class ADX(Strategy):
    n = 14

    def _shift(self, array):
        return np.resize(np.insert(array, 0, np.nan), len(array))

    def _tr(self, data, n):
        close = self._shift(data.Close)
        return np.max(
            [
                data.High - data.Low,
                data.High - close,
                close - data.Low,
            ],
            axis=0,
        )

    def _atr(self, data, n):
        tr = self._tr(data, n)
        return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean()

    def _dmi(self, data, n):
        positive = data.High - self._shift(data.High)
        negative = self._shift(data.Low) - data.Low
        dm_plus = np.where((positive > negative) & (positive > 0), positive, 0.0)
        dm_minus = np.where((positive < negative) & (negative > 0), negative, 0.0)
        atr = self._atr(data, n)
        dmi_plus = 100 * pd.Series(dm_plus).ewm(alpha=1 / n, adjust=False).mean() / atr
        dmi_minus = (
            100 * pd.Series(dm_minus).ewm(alpha=1 / n, adjust=False).mean() / atr
        )
        return dmi_plus, dmi_minus

    def _adx(self, data, n):
        dmi_plus, dmi_minus = self._dmi(data, n)
        dx = 100 * np.abs(dmi_plus - dmi_minus) / (dmi_plus + dmi_minus)
        return dx.ewm(alpha=1 / n, adjust=False).mean()

    def init(self):
        self.atr = self.I(self._atr, self.data, self.n, name="ATR", overlay=False)
        self.dmi = self.I(self._dmi, self.data, self.n, name="DMI", overlay=False)
        self.adx = self.I(self._adx, self.data, self.n, name="ADX", overlay=False)
        self.dmi_plus = self.I(
            lambda data, n: self._dmi(data, n)[0], self.data, self.n, plot=False
        )
        self.dmi_minus = self.I(
            lambda data, n: self._dmi(data, n)[1], self.data, self.n, plot=False
        )

    def next(self):
        if crossover(self.dmi_plus, self.dmi_minus):
            self.position.close()
            self.buy()
        elif crossover(self.dmi_minus, self.dmi_plus):
            self.position.close()
            self.sell()


if __name__ == "__main__":
    ticker = input("Ticker: ")
    ticker = yf.Ticker(ticker)
    period = input("Period of time to implement (default: max - Options: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max ): ") or "max"
    interval = input("Time interval (default: 1d - Optinons: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo): ") or "1d"
    data = ticker.history(period, interval)
    strategy_name = (
        input("Strategy you'd like to use (default: SmaCross - Options: SmaCross, ESmaCross, Aroon, ADX): ") or "SmaCross"
    )
    strategy = getattr(sys.modules[__name__], strategy_name)

    if "macross" in strategy_name.lower():
        strategy.n1 = int(input("Type the short MA you'd like to use (default: 50): ") or "50")
        strategy.n2 = int(
            input("Type the long MA you'd like to use (default: 200): ") or "200"
        )
    elif strategy_name.lower() == "aroon":
        strategy.n = int(
            input("Time period (default: 14): ") or "14"
        )

    bt = Backtest(data, strategy)
    stats = bt.run()
    print("\nRESULTADOS")
    print("==================")
    print(stats)
    bt.plot()
    plt.show()
    
    
"""PROGRAM DONE BY FERNANDO KRIKUN AND RAFAEL ANDANT"""

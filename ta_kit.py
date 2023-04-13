# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:33:20 2023

@author: IKU-Trader
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './'))

from technical_analysis import TechnicalAnalysis as ta
from const import const


class TAKit:
    
    @staticmethod
    def basic():
        trend_params = {ta.MA_KEYS:['SMA5', 'SMA20', 'SMA60'], ta.THRESHOLD:0.05}
        patterns = {
                        ta.SOURCE: 'MA_TREND',
                        ta.PATTERNS:[
                                [[ta.NO_TREND, ta.UPPER_TREND], 1, 0],
                                [[ta.UPPER_SUB_TREND, ta.UPPER_TREND], 1, 0],
                                [[ta.NO_TREND, ta.LOWER_TREND], 2, 0],
                                [[ta.LOWER_SUB_TREND, ta.LOWER_TREND], 2, 0]
                                ]
                    }

        params = [
                    [ta.SMA, {ta.WINDOW: 5}, 'SMA5'],
                    [ta.SMA, {ta.WINDOW: 20}, 'SMA20'],
                    [ta.SMA, {ta.WINDOW: 60}, 'SMA60'],
                    [ta.ATR, {ta.WINDOW: 14}, 'ATR'],
                    [ta.ADX, {ta.WINDOW: 14}, 'ADX'],
                    [ta.BOLLINGER_BAND_UPPER, {ta.WINDOW: 14, ta.SIGMA: 2.0}, 'BOLLINGER+'],
                    [ta.BOLLINGER_BAND_LOWER, {ta.WINDOW: 14, ta.SIGMA: 2.0}, 'BOLLINGER-'],
                    [ta.SLOPE, {ta.SOURCE: 'SMA5', ta.WINDOW: 3}, 'SLOPE_SMA5'],
                    [ta.SLOPE, {ta.SOURCE: 'SMA20', ta.WINDOW: 3}, 'SLOPE_SMA20'],
                    [ta.SLOPE, {ta.SOURCE: 'SMA60', ta.WINDOW: 3}, 'SLOPE_SMA60'],
                    [ta.MA_TREND_BAND, trend_params, 'MA_TREND'],
                    [ta.PATTERN_MATCH, patterns, 'SIGNAL'],
                    [ta.UPPER_TIMEFRAME, {ta.SOURCE: const.CLOSE, ta.TIMEFRAME: 'H2', ta.WINDOW: 20}, 'H2_SMA20'],
                    [ta.UPPER_TIMEFRAME, {ta.SOURCE: const.CLOSE, ta.TIMEFRAME: 'H4', ta.WINDOW: 20}, 'H4_SMA20']
                ]
        return params
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:33:20 2023

@author: IKU-Trader
"""

from STA import TechnicalAnalysis as ta
from const import const


class TAKit:
    
    @staticmethod
    def matrend():
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
                    [ta.MA_TREND_BAND, trend_params, 'MA_TREND'],
                    [ta.PATTERN_MATCH, patterns, 'SIGNAL']
                ]
        return params

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:49:18 2023

@author: IKU
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../MarketData'))

import pandas as pd
from MarketData import MarketData
from TA import TechnicalAnalysis as ta

def test_adx(ohcv):

    pass

def test_adx():
    path = './adx_test_data.csv'
    df = pd.read_csv(path)
    op = df['Open']
    hi = df['High']
    lo = df['Low']
    cl = df['Close']
    
    adx = ta.adx(hi, lo, cl, 14)
    
    pass

def test_upperTimeframe():
    data = MarketData.fxData('GBPJPY', {}, [2022], [1], 5)
    dic = data.dic
    array = ta.upperTimeframe(dic, 'close', 'H2', ma_window=20)
    print(len(array))

    
if __name__ == '__main__':
    test_upperTimeframe()
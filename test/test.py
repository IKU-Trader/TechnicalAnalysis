# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:49:18 2023

@author: IKU
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd
from TA import TechnicalAnalysis as ta

def test_adx(ohcv):

    pass

def test():
    path = './adx_test_data.csv'
    df = pd.read_csv(path)
    op = df['Open']
    hi = df['High']
    lo = df['Low']
    cl = df['Close']
    
    adx = ta.adx(hi, lo, cl, 14)
    
    pass

    
if __name__ == '__main__':
    test()
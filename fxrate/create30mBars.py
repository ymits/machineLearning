#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import fxrate
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def createBars(df, from_date, to_date, bar_range):
    bars = []
    calc_date_from = from_date
    while calc_date_from < to_date:
        calc_date_to = calc_date_from + timedelta(minutes=bar_range)
        df_range = df[(df.time >= calc_date_from.strftime(format)) & (df.time < calc_date_to.strftime(format))]
        
        calc_date_from = calc_date_to
        if len(df_range.index) == 0:
            continue
        
        first_row = df_range.head(1)
        last_row = df_range.tail(1)
        bars.append([
            first_row.time.values[0], 
            (float(first_row.bid_op.values[0]) + float(first_row.ask_op.values[0]) )/ 2, 
            max(df_range.bid_hi), 
            min(df_range.ask_lo), 
            (float(last_row.bid_cl.values[0]) + float(last_row.ask_cl.values[0])) / 2
        ])
    return np.array(bars)

def calcMA(bars):
    ma_types = [10, 25, 75]
    mas = np.zeros((len(bars), len(ma_types)))
    
    for i in range(len(bars)):
        for j in range(len(ma_types)):
            ma_type = ma_types[j]
            if i - ma_type < 0:
                break
            
            sub_bars = np.array(bars[i - ma_type : i])
            avg = np.average(sub_bars[:,1:].astype(np.float), axis=0)
            mas[i][j] = avg[3]
    return mas

df = fxrate.load_csv(r'data/**')

format = "%Y/%m/%d %H:%M:%S"
from_date = datetime.strptime('2016/01/01 07:00:00', format)
to_date = datetime.strptime('2018/01/01 00:00:00', format)

bars_30m = createBars(df, from_date, to_date, 30)
bars_with_ma_30m = np.append(bars_30m, calcMA(bars_30m), axis=1)
bars_1d = createBars(df, from_date, to_date, 60 * 24)
bars_with_ma_1d = np.append(bars_1d, calcMA(bars_1d), axis=1)

bar_with_ma_30m_df = pd.DataFrame(bars_with_ma_30m[75:], columns=['time', 'op', 'hi', 'lo', 'cl', '10ma', '25ma', '75ma'])
bar_with_ma_30m_df.to_csv("data/bar_30m.csv", index=False)
bar_with_ma_1d_df = pd.DataFrame(bars_with_ma_1d[75:], columns=['time', 'op', 'hi', 'lo', 'cl', '10ma', '25ma', '75ma'])
bar_with_ma_1d_df.to_csv("data/bar_1d.csv", index=False)


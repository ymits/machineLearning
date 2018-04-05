#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import glob
import os
import pickle

# 前準備1 20140620以前の1分足作成
def create_bar1m_before_20140620():
    all_files = glob.glob(os.path.join('data/**', "*.csv"), recursive=True)

    before_20140620_files = [f for f in all_files if f < 'data/2014/201406/USDJPY_20140620.csv']
    df_from_each_file = (pd.read_csv(f, header=0, names=('time', 'op', 'hi', 'lo', 'cl'), encoding="shift-jis" ) for f in before_20140620_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    df = df.sort_values(by='time')

    df.to_csv("data_dist/bar_1m_before_20140620.csv", index=False)

# 前準備２ 20140620以降20151231以前の1分足作成
def create_bar1m_before_20151231():
    all_files = glob.glob(os.path.join('data/**', "*.csv"), recursive=True)

    before_20151231_files = [f for f in all_files if f >= 'data/2014/201406/USDJPY_20140620.csv' and f < 'data/2015/201512/USDJPY_20151231.csv']
    df_from_each_file = (pd.read_csv(f, header=0, names=('time', 'op', 'hi', 'lo', 'cl', 'a', 'b'), encoding="shift-jis" ) for f in before_20151231_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    df = df.sort_values(by='time')
    del df['a']
    del df['b']
    df.to_csv("data_dist/bar_1m_before_20151231.csv", index=False)

# 前準備3 20160101以降の１分足作成
def create_bar1m_after_20160101():
    all_files = glob.glob(os.path.join('data/**', "*.csv"), recursive=True)

    after_20160101_files = [f for f in all_files if f >= 'data/2016/201601/USDJPY_20160101.csv']
    df_from_each_file = (pd.read_csv(f, header=0, names=('time', 'bid_op', 'bid_hi', 'bid_lo', 'bid_cl', 'ask_op', 'ask_hi', 'ask_lo', 'ask_cl'), encoding="shift-jis" ) for f in after_20160101_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    df = df.sort_values(by='time')

    new_df = pd.DataFrame({ 
         'time' : df['time'].str.replace(r'(/|:| )', ''),
         'op' : (df['bid_op'].astype(float) + df['ask_op'].astype(float)) /2,
         'hi' : df['bid_hi'],
         'lo' : df['ask_lo'],
         'cl' : (df['bid_cl'].astype(float) + df['ask_cl'].astype(float)) /2
         })
    new_df = new_df[['time', 'op', 'hi', 'lo', 'cl']]
    new_df.to_csv("data_dist/bar_1m_after_20160101.csv", index=False)

# FX時価のCSVファイルロード
def load_csv(path):
    all_files = glob.glob(os.path.join(path, "*.csv"), recursive=True)

    df_from_each_file = (pd.read_csv(f, names=('time', 'bid_op', 'bid_hi', 'bid_lo', 'bid_cl', 'ask_op', 'ask_hi', 'ask_lo', 'ask_cl'), encoding="shift-jis" ) for f in all_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    df = df.sort_values(by='time')

    return df

# FX時価データから足データを作成
def createBars(df, from_date, to_date, bar_range):
    time_format = "%Y%m%d%H%M%S"
    bars = []
    calc_date_from = from_date
    while calc_date_from < to_date:
        calc_date_to = calc_date_from + timedelta(minutes=bar_range)
        val_from = int(calc_date_from.strftime(time_format))
        val_to = int(calc_date_to.strftime(time_format))
        if val_from % 100000000 == 1000000:
            print(bar_range, val_from / 1000000)
        df_range = df[(df.time >= val_from) & (df.time < val_to)]
        
        calc_date_from = calc_date_to
        if len(df_range.index) == 0:
            continue
        
        first_row = df_range.head(1)
        last_row = df_range.tail(1)
        bars.append([
            first_row.time.values[0], 
            float(first_row.op.values[0]), 
            max(df_range.hi), 
            min(df_range.lo), 
            float(last_row.cl.values[0])
        ])
    return np.array(bars)

# 移動平均線を計算して返します
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

# 1分足を作成します。
def create_1m_bar():
    create_bar1m_before_20140620()
    create_bar1m_before_20151231()
    create_bar1m_after_20160101()

# 30分足を作成します。
def create_30m_1d_bar():
    df１ = pd.read_csv('data_dist/bar_1m_before_20140620.csv', header=0, names=('time', 'op', 'hi', 'lo', 'cl'), encoding="shift-jis" )
    df2 = pd.read_csv('data_dist/bar_1m_before_20151231.csv', header=0, names=('time', 'op', 'hi', 'lo', 'cl'), encoding="shift-jis" )
    df3 = pd.read_csv('data_dist/bar_1m_after_20160101.csv', header=0, names=('time', 'op', 'hi', 'lo', 'cl'), encoding="shift-jis" )
    df_before_20160101 = pd.concat([df1, df2])
    df_after_20160101 = df3

    format = "%Y/%m/%d %H:%M:%S"
    from_date = datetime.strptime('2007/01/01 07:00:00', format)
    change_point = datetime.strptime('2016/01/01 07:00:00', format)
    to_date = datetime.strptime('2019/01/01 00:00:00', format)

    bars_30m_before_20160101 = createBars(df_before_20160101, from_date, change_point, 30)
    bars_30m_before_20160101_df = pd.DataFrame(bars_30m_before_20160101, columns=['time', 'op', 'hi', 'lo', 'cl'])
    bars_30m_before_20160101_df.to_csv("data_dist/bar_30m_before_20160101.csv", index=False)
    bars_1d_before_20160101 = createBars(df_before_20160101, from_date, change_point, 60 * 24)
    bars_1d_before_20160101_df = pd.DataFrame(bars_1d_before_20160101, columns=['time', 'op', 'hi', 'lo', 'cl'])
    bars_1d_before_20160101_df.to_csv("data_dist/bar_1d_before_20160101.csv", index=False)
    bars_30m_after_20160101 = createBars(df_after_20160101, change_point, to_date, 30)
    bars_30m_after_20160101_df = pd.DataFrame(bars_30m_after_20160101, columns=['time', 'op', 'hi', 'lo', 'cl'])
    bars_30m_after_20160101_df.to_csv("data_dist/bar_30m_after_20160101.csv", index=False)
    bars_1d_after_20160101 = createBars(df_after_20160101, change_point, to_date, 60 * 24)
    bars_1d_after_20160101_df = pd.DataFrame(bars_1d_after_20160101, columns=['time', 'op', 'hi', 'lo', 'cl'])
    bars_1d_after_20160101_df.to_csv("data_dist/bar_1d_after_20160101.csv", index=False)

# 30分足に移動平均を付与します。
def add_ma():
    df_30m_before = pd.read_csv('data_dist/bar_30m_before_20160101.csv', header=0 , names=('time', 'op', 'hi', 'lo', 'cl'))
    np_30m_bevore = df_30m_before.values
    np_30m_ma_bevore = np.append(np_30m_bevore, calcMA(np_30m_bevore), axis=1)
    df_30m_ma_bevore = pd.DataFrame(np_30m_ma_bevore[75:], columns=['time', 'op', 'hi', 'lo', 'cl', '10ma', '25ma', '75ma'])
    df_30m_ma_bevore.to_csv("data_dist/bar_30m_ma_before_20160101.csv", index=False)
    
    df_30m_after = pd.read_csv('data_dist/bar_30m_after_20160101.csv', header=0 , names=('time', 'op', 'hi', 'lo', 'cl'))
    np_30m_after = df_30m_after.values
    np_30m_ma_after = np.append(np_30m_after, calcMA(np_30m_after), axis=1)
    df_30m_ma_after = pd.DataFrame(np_30m_ma_after[75:], columns=['time', 'op', 'hi', 'lo', 'cl', '10ma', '25ma', '75ma'])
    df_30m_ma_after.to_csv("data_dist/bar_30m_ma_after_20160101.csv", index=False)
    
    df_1d_before = pd.read_csv('data_dist/bar_1d_before_20160101.csv', header=0 , names=('time', 'op', 'hi', 'lo', 'cl'))
    np_1d_bevore = df_1d_before.values
    np_1d_ma_bevore = np.append(np_1d_bevore, calcMA(np_1d_bevore), axis=1)
    df_1d_ma_bevore = pd.DataFrame(np_1d_ma_bevore[75:], columns=['time', 'op', 'hi', 'lo', 'cl', '10ma', '25ma', '75ma'])
    df_1d_ma_bevore.to_csv("data_dist/bar_1d_ma_before_20160101.csv", index=False)
    
    df_1d_after = pd.read_csv('data_dist/bar_1d_after_20160101.csv', header=0 , names=('time', 'op', 'hi', 'lo', 'cl'))
    np_1d_after = df_1d_after.values
    np_1d_ma_after = np.append(np_1d_after, calcMA(np_1d_after), axis=1)
    df_1d_ma_after = pd.DataFrame(np_1d_ma_after[75:], columns=['time', 'op', 'hi', 'lo', 'cl', '10ma', '25ma', '75ma'])
    df_1d_ma_after.to_csv("data_dist/bar_1d_ma_after_20160101.csv", index=False)
   
    
def load_bars(bar_num, forward=1, diff = 0, normalize=True):
    def load_bars_from_file(filename_30m, finename_1d, bar_num, forward=1, diff = 0, normalize=True):
        parser = lambda date: datetime.strptime(date, '%Y%m%d%H%M%S.0')
        bars30m = pd.read_csv(filename_30m, parse_dates=[0], header=0, date_parser = parser)
        bars1d = pd.read_csv(finename_1d, parse_dates=[0], header=0, date_parser = parser)

        trains = []
        tests = []
        for i in range(len(bars30m)):
            if i - bar_num < 0:
                continue

            if i + forward >= len(bars30m):
                break

            current_bar = bars30m[i - 1 : i]
            forward_bar = bars30m[i + forward - 1 : i + forward]

            train_30m_data = bars30m[i - bar_num : i]
            current_time = current_bar.time.values[0]
            train_1d_data = bars1d[(bars1d.time <= current_time) ].tail(bar_num)

            if len(train_1d_data) != bar_num:
                continue

            train_data = np.append(train_30m_data.iloc[:,1:].values, train_1d_data.iloc[:,1:].values, axis=1)
            # ノーマライズ処理
            if normalize:
                train_data = train_data - np.amin(train_data)
                train_data = train_data / np.amax(train_data)
            trains.append(train_data)

            test_data = [
                forward_bar.cl.values[0] - current_bar.cl.values[0] - diff >= 0,
                current_bar.cl.values[0] - forward_bar.cl.values[0] - diff >= 0,
               ( forward_bar.cl.values[0] - current_bar.cl.values[0] - diff < 0 ) and ( current_bar.cl.values[0] - forward_bar.cl.values[0] - diff < 0 )
            ]
            tests.append(test_data)

        return (np.asarray(trains), np.asarray(tests))
    
    def pickle_name(filename_30m, finename_1d, bar_num, forward=1, diff = 0, normalize=True):
        return  "temp/" + filename_30m.replace('/', '') + finename_1d.replace('/', '') + str(bar_num) + str(forward) + str(diff) + str(normalize)
        
    def load_bars_wrapper(filename_30m, finename_1d, bar_num, forward=1, diff = 0, normalize=True):
        f_name = pickle_name(filename_30m, finename_1d, bar_num, forward, diff, normalize)
        if not os.path.exists(f_name):
            bars = load_bars_from_file(filename_30m, finename_1d, bar_num, forward, diff, normalize)
            with open(f_name, 'wb') as f:
                pickle.dump(bars, f, -1)

        with open(f_name, 'rb') as f:
            bars = pickle.load(f)
        
        return bars            
    
    (x_train, y_train) = load_bars_wrapper("data_dist/bar_30m_ma_before_20160101.csv", "data_dist/bar_1d_ma_before_20160101.csv", bar_num, forward, diff, normalize)
    (x_test, y_test) = load_bars_wrapper("data_dist/bar_30m_ma_after_20160101.csv", "data_dist/bar_1d_ma_after_20160101.csv", bar_num, forward, diff, normalize)
    
    return (x_train, y_train),  (x_test, y_test)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import pandas as pd
import os
import numpy as np

# CSVロード
def load_csv(path):
    all_files = glob.glob(os.path.join(path, "*.csv"), recursive=True)

    df_from_each_file = (pd.read_csv(f, names=('time', 'bid_op', 'bid_hi', 'bid_lo', 'bid_cl', 'ask_op', 'ask_hi', 'ask_lo', 'ask_cl'), encoding="shift-jis" ) for f in all_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    df = df.sort_values(by='time')

    return df

# FXデータロード
def load_data(path, bar_num, forward, flatten=True, normalize=True):
    forward_index = bar_num + forward - 1
    current_index = bar_num -1

    df = load_csv(path )
    max = len(df)
    index = 0
    train_data = []
    test_data = []

    while index + bar_num + forward <= max:
        df_unit = df.iloc[index: index + bar_num + forward, :]
        vals = df_unit.iloc[:10,1:]

        # フラット処理
        if flatten:
            vals = vals.values.flatten()

        # ノーマライズ処理
        if normalize:
            vals = vals - np.amin(vals)
            vals = vals / np.amax(vals)

        train_data.append(vals.tolist())
        forward_bar = df_unit.iloc[forward_index]
        current_bar = df_unit.iloc[current_index]
        test_data.append([
                forward_bar.bid_cl > current_bar.ask_cl,
                (forward_bar.bid_cl<= current_bar.ask_cl) & (forward_bar.ask_cl >= current_bar.bid_cl),
                forward_bar.ask_cl < current_bar.bid_cl
            ])

        index+=1

    return (np.asarray(train_data), np.asarray(test_data))

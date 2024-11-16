#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:41:41 2024

@author: antonio
"""

##½½
# Peticiones get defi llama api con pd.dataframe
import pandas as pd
import requests

url_chain =  "https://api.llama.fi/v2/historicalChainTvl/"
chains = ['Ethereum', 'Solana', 'BSC', 'Tron', 'Tron', 'Sui', 'Aptos']
dfs = []

for chain in chains:
    url = url_chain + chain

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)

        df['date'] = pd.to_datetime(df['date'], unit='s')
        df.set_index('date', inplace=True)
        df.rename(columns={'tvl':chain}, inplace=True)
        dfs.append(df)
    else:
        print(f'Error al realizar la solicitud. Código de estado: {response.status_code}')

df = pd.concat(dfs, axis=1)
df
##½½
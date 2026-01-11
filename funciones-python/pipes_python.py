#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:13:36 2024

@author: antonio
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('datos_procesados_penguins.csv')

(
    df
    .isnull()
    .melt(value_name='missing')
    .pipe(lambda pre_df: (
            sns.displot(data=pre_df,
                        y='variable',
                        hue='missing',
                        multiple='fill',
                        aspect=2))))
plt.show()


(
    df.assign(variable='').pipe(
    lambda df:(sns.displot(data=df,
                           x='variable',
                           hue='species',
                           multiple='fill'
                           )))
)
plt.title('Proporciones por especies')
plt.show()
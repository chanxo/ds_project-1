# Importing Libraries
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Referencing folders and data names
path = os.getcwd()
file_name = 'winequality-red.csv'
path_file = f'{path}/code_python/project1_wine/data/{file_name}'

# Loading the data
wine_data = pd.read_csv(path_file)

# Exploratory analysis

# Checking overall structure of the data
wine_data.info()

# Since there are no null or NAs we can proceed to analyse the data further

# Using Pearson Correlation
plt.figure(figsize=(12, 10))
corr_mat = wine_data.corr()
ut_mask = np.triu(np.ones_like(corr_mat, dtype=bool))
corr_heatmap = sns.heatmap(corr_mat, mask=ut_mask, annot=True, linewidths=.8, cmap='YlGnBu')
corr_heatmap.set_title('Correlation Heatmap - Wine characteristics', fontdict={'fontsize': 21})
plt.savefig(f'{path}/document_files/figs/wine_heatmap.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Subplots with KDE to check for clusters
fig, axes = plt.subplots(3, 4, sharey=True, figsize=(12, 11))
fig.suptitle('KDE plots - Quality against each feature', fontsize='xx-large')
#  Quality vs. each wine feature
row = 0
column = 0
feature = 0
while True:
    sns.kdeplot(ax=axes[row, column], x=wine_data.iloc[:, feature], y=wine_data.iloc[:, 11])
    if row == 2 and column == 2:
        sns.histplot(ax=axes[row, column+1], data=wine_data, y='quality', kde=True)
        # fig.delaxes(ax=axes[row, column+1])
        break
    feature += 1
    column += 1
    if column % 4 == 0:
        row += 1
        column = 0
plt.savefig(f'{path}/document_files/figs/wine_kde.pdf', format='pdf', bbox_inches='tight')
plt.show()

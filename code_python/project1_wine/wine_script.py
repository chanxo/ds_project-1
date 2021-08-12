# Importing Libraries
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Referencing folders and data names
path = os.getcwd()
# path = os.path.abspath(os.path.join(path, os.pardir))  # when in latex
file_name = 'winequality-red.csv'
path_file = f'{path}/code_python/project1_wine/data/{file_name}'

# Loading the data
wine_data = pd.read_csv(path_file)

# Exploratory analysis

# Checking overall structure of the data
wine_data_info = wine_data.info()
# Since there are no null or NAs we can proceed to analyse the data further

# Using Pearson Correlation
plt.figure(figsize=(12, 10))
corr_mat = wine_data.corr()
ut_mask = np.triu(np.ones_like(corr_mat, dtype=bool))
corr_heatmap = sns.heatmap(corr_mat, mask=ut_mask, annot=True, linewidths=.8, cmap='YlGnBu')
corr_heatmap.set_title('Correlation Heatmap - Wine characteristics', fontdict={'fontsize': 21})
# plt.savefig(f'{path}/document_files/figs/wine_heatmap.pdf', format='pdf', bbox_inches='tight')
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
        # fig.delaxes(ax=axes[row, column+1]) # if we want a blank output in the last subplot
        break
    feature += 1
    column += 1
    if column % 4 == 0:
        row += 1
        column = 0
# plt.savefig(f'{path}/document_files/figs/wine_kde.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Let us make an Appendix at the end with the description of each variable.
description = {'fixed acidity': 'most acids involved with wine or fixed or nonvolatile (do not evaporate readily).',
               'volatile acidity': 'the amount of acetic acid in wine, which at too high of levels can lead to an '
                                   'unpleasant, vinegar taste.',
               'citric acidity': 'found in small quantities, citric acid can add ``freshness" and flavor to wines.',
               'residual sugar': 'the amount of sugar remaining after fermentation stops, it`s rare to find wines '
                                 'with less than 1 gram/liter and wines with greater than 45 grams/liter are '
                                 'considered sweet.',
               'chlorides': 'the amount of salt in the wine.',
               'free sulfur dioxide': 'the free form of SO2 exists in equilibrium between molecular SO2 (as a '
                                      'dissolved gas) and bi-sulfate ion; it prevents microbial growth and the '
                                      'oxidation of wine.',
               'total sulfur dioxide': 'amount of free and bound forms of S02; in low concentrations, SO2 is mostly '
                                       'undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes '
                                       'evident in the nose and taste of wine.',
               'density': 'the density of water is close to that of water depending on the percent alcohol and sugar '
                          'content.',
               'pH': 'describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); '
                     'most wines are between 3-4 on the pH scale.',
               'sulphates': 'a wine additive which can contribute to sulfur dioxide gas (S02) levels, which acts as '
                            'an antimicrobial and antioxidant.',
               'alcohol': 'the percent alcohol content of the wine.',
               'quality': 'output variable (based on sensory data, score between 0 and 10).'
               }
description_df = pd.DataFrame(list(description.items()), columns=['Characteristic', 'Description'])
# pd.options.display.max_colwidth = 400
# print(description_df.to_latex(index=False, multirow=True),
#      file=open(f'{path}/code_python/project1_wine/description_wine.tex', "a"))

# End of Exploratory Analysis




















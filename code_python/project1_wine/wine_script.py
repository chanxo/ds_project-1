# Importing Libraries
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

## Referencing folders and data names
path = os.getcwd()
# path = os.path.abspath(os.path.join(path, os.pardir))  # when in latex
file_name = 'winequality-red.csv'
path_file = f'{path}/code_python/project1_wine/data/{file_name}'

## Loading the data
wine_data = pd.read_csv(path_file)
features = wine_data.columns

## Exploratory analysis

# Checking overall structure of the data
wine_data_info = wine_data.info()
wine_data.describe().round(decimals=2).transpose()
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
fig.suptitle('KDE plots - Quality against each wine characteristic', fontsize='xx-large')
#  Quality vs. each wine feature
row = 0
column = 0
feature = 0
while True:
    sns.kdeplot(ax=axes[row, column], x=wine_data.iloc[:, feature], y=wine_data.iloc[:, 11])
    if row == 2 and column == 2:
        sns.histplot(ax=axes[row, column + 1], data=wine_data, y='quality', kde=True)
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


## Quality prediction.
# Loss-function assessment
from sklearn.metrics import mean_squared_error as rmse, r2_score as score

# rms = mean_squared_error(y_actual, y_predicted, squared=False)

# Pre-processing
# Shaping data
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']
# Defining a seed for shuffling the data
seed = 123
# Splitting data into train/test.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
# Converting 1D arrays to dataframe
y_train = pd.DataFrame(Y_train, columns=['quality'])
# Y_test = pd.DataFrame(Y_test, columns=['quality'])
# Scaling training samples
train_X_scaler = StandardScaler()
train_Y_scaler = StandardScaler()
# we first fit the training data to different scaling objects to keep track of them
train_X_scaler.fit(X_train)
train_Y_scaler.fit(y_train)
x_train = train_X_scaler.transform(X_train)
y_train = train_Y_scaler.transform(y_train)

# Methods
results = {}

# Linear Regression
# Normal Linear Regression (L2 Norm)
from sklearn.linear_model import LinearRegression as lr

# We first initialise the model and then fit with the training observations
normal_lr = lr(fit_intercept=True, normalize=False)
normal_lr.fit(X=x_train, y=y_train)
# Predicting values, we first need to transform the X_test matrix using our earlier defined scale
nlr_y = normal_lr.predict(train_X_scaler.transform(X_test))
# Reverting our predicted values to their level using the scale determined from the training sample
nlr_y = train_Y_scaler.inverse_transform(nlr_y)
nlr_rmse = rmse(y_test, nlr_y, squared=False)
nlr_score = score(y_true=y_test, y_pred=nlr_y)
results['linear regression'] = [nlr_rmse, nlr_score]

# Lasso Linear Regression (L1 Norm)
# we do not standardize here otherwise the lasso regression might turn all coefficients to 0 only to use the intercept.
from sklearn.linear_model import Lasso as lasso

# We first initialise the model and then fit with the training observations
# also do not use intercept, if the intercept is allowed, again makes most of the coefficients to be 0.
lasso_lr = lasso(fit_intercept=False, normalize=False)
lasso_lr.fit(X=X_train, y=Y_train)
# Predicting values, we first need to transform the X_test matrix using our earlier defined scale
lassolr_y = lasso_lr.predict(X_test)
# Reverting our predicted values to their level using the scale determined from the training sample
lassolr_y = train_Y_scaler.inverse_transform(lassolr_y)
lassolr_rmse = rmse(y_test, lassolr_y, squared=False)
lassolr_score = score(y_true=y_test, y_pred=lassolr_y)
results['lasso linear regression'] = [lassolr_rmse, lassolr_score]
# todo here we can see that the lasso rmse is higher than the normal linear regression, to be expected due the
#  objective function to minimise
# todo explain that a negative score just means that the particular model is performing quite poorly.

# Neural Network
from sklearn.neural_network import MLPRegressor

# try:
NN_scikit = MLPRegressor(random_state=seed, max_iter=1000, hidden_layer_sizes=(22, 22)).fit(x_train, y_train)
# we use (22,) in the hidden layers to try to capture the features and their interactions, we will do it
# because it is too little data. This is pushing it a bit, given the small sample size
nn_scikit_y = NN_scikit.predict(train_X_scaler.transform(X_test))
# Reverting our predicted values to their level using the scale determined from the training sample
nn_scikit_y = train_Y_scaler.inverse_transform(nn_scikit_y)
nn_scikit_rmse = rmse(y_test, nn_scikit_y, squared=False)
nn_scikit_score = score(y_true=y_test, y_pred=nn_scikit_y)
results['NN Scikit'] = [nn_scikit_rmse, nn_scikit_score]

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 11))
axes[0].set_title('Training Sample')
axes[1].set_title('Test Sample')
axes[2].set_title('NN predicted Quality')
sns.histplot(x=Y_train, ax=axes[0], kde=True, stat="probability")
sns.histplot(x=y_test, ax=axes[1], kde=True, stat="probability")
sns.histplot(x=nn_scikit_y, ax=axes[2], kde=True, stat="probability")
plt.show()

# Regression Tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import graphviz

tree_model = DecisionTreeRegressor(max_depth=4)
tree_model.fit(X_train, Y_train)
tree_y = tree_model.predict(X_test)
tree_rmse = rmse(y_test, tree_y, squared=False)
tree_score = score(y_true=y_test, y_pred=tree_y)
results['Regression Tree'] = [tree_rmse, tree_score]

plt.figure(figsize=(12, 10))
tree_viz = tree.export_graphviz(tree_model, out_file=None,
                                feature_names=features[0:10],
                                class_names=features[11],
                                filled=True, rounded=True,
                                special_characters=True)
tree_graph = graphviz.Source(tree_viz)
# tree_graph.render('wine_data')
tree_graph
plt.show()

plt.figure(figsize=(12, 10))
tree.plot_tree(tree_model)
plt.show()

pd.DataFrame(results).round(decimals=2).transpose()

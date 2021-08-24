# Importing Libraries
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Some inspiration
# https://medium.com/dataman-in-ai/the-shap-with-more-elegant-charts-bc3e73fa1c0c

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
# Saving table to latex output
# pd.options.display.max_colwidth = 400
# print(description_df.to_latex(index=False, multirow=True),
#      file=open(f'{path}/code_python/project1_wine/description_wine.tex', "a"))

# End of Exploratory Analysis


## Quality prediction.
# Loss-function assessment
from sklearn.metrics import mean_squared_error as rmse, r2_score as score, mean_absolute_percentage_error as mape

# rms = mean_squared_error(y_actual, y_predicted, squared=False)

# Pre-processing
# Shaping data
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']
# Defining a seed for shuffling the data
seed = 123
# Splitting data into train/test.
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
# Y_test = pd.DataFrame(Y_test, columns=['quality'])

# Converting 1D arrays to dataframe
y_train = pd.DataFrame(Y_train, columns=['quality'])
# Scaling training samples
from sklearn.preprocessing import StandardScaler

train_X_scaler = StandardScaler()
train_Y_scaler = StandardScaler()
# we first fit the training data to different scaling objects to keep track of them
train_X_scaler.fit(X_train)
train_Y_scaler.fit(y_train)
x_train = train_X_scaler.transform(X_train)
y_train = train_Y_scaler.transform(y_train)

# Methods
results = {}
model_predictions = pd.DataFrame()

# Linear Regression
# Normal Linear Regression (L2 Norm)
from sklearn.linear_model import LinearRegression as lr

# We first initialise the model and then fit with the training observations
normal_lr = lr(fit_intercept=True, normalize=False)
normal_lr.fit(X=x_train, y=y_train)
# Predicting values, we first need to transform the X_test matrix
# using our earlier defined scale
nlr_y = normal_lr.predict(train_X_scaler.transform(X_test))
# Reverting our predicted values to their level using the scale determined
# from the training sample
nlr_y = train_Y_scaler.inverse_transform(nlr_y)
nlr_rmse = rmse(y_test, nlr_y, squared=False)
nrl_mape = mape(y_true=y_test, y_pred=nlr_y)
nlr_score = score(y_true=y_test, y_pred=nlr_y)
results['linear regression'] = [nlr_rmse, nrl_mape, nlr_score]
model_predictions['linear regression'] = np.ravel(nlr_y)

# Lasso Linear Regression (L1 Norm)
# we do not standardize here otherwise the lasso regression might turn
# all coefficients to 0 only to use the intercept.
from sklearn.linear_model import Lasso as lasso

# We first initialise the model and then fit with the training observations
# also do not use intercept, if the intercept is allowed, again makes
# most of the coefficients to be 0.
lasso_lr = lasso(fit_intercept=False, normalize=False)
lasso_lr.fit(X=X_train, y=Y_train)
lassolr_y = lasso_lr.predict(X_test)

lassolr_rmse = rmse(y_test, lassolr_y, squared=False)
lassorl_mape = mape(y_true=y_test, y_pred=lassolr_y)
lassolr_score = score(y_true=y_test, y_pred=lassolr_y)
results['lasso linear regression'] = [lassolr_rmse, lassorl_mape, lassolr_score]
model_predictions['lasso linear regression'] = np.ravel(lassolr_y)
# todo here we can see that the lasso rmse is higher than the normal linear regression, to be expected due the
#  objective function to minimise

# Neural Network
from sklearn.neural_network import MLPRegressor

NN_scikit = MLPRegressor(random_state=seed,
                         max_iter=500,
                         hidden_layer_sizes=(12, 12*11,),
                         activation='logistic').fit(x_train, np.ravel(y_train))
NN_scikit.out_activation_ = 'identity'
# we use (22,) in the hidden layers to try to capture the features and
# their interactions, we will do it because it is too little data.
# Using more neurons or hidden layers in this case might induce
# over-fitting, given the small sample size.
nn_scikit_y = NN_scikit.predict(train_X_scaler.transform(X_test))
# Reverting our predicted values to their level using the scale determined
# from the training sample.
nn_scikit_y = train_Y_scaler.inverse_transform(nn_scikit_y)
nn_scikit_rmse = rmse(y_test, nn_scikit_y, squared=False)
nn_scikit_mape = mape(y_true=y_test, y_pred=nn_scikit_y)
nn_scikit_score = score(y_true=y_test, y_pred=nn_scikit_y)
results['NN Scikit'] = [nn_scikit_rmse, nn_scikit_mape, nn_scikit_score]
model_predictions['NN Scikit'] = np.ravel(nn_scikit_y)

# NN With Keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(seed)


def squared_loss(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_sum(squared_difference, axis=-1)  # Note the `axis=-1`


nn_keras_tf = Sequential()
nn_keras_tf.add(Dense(12,
                      input_dim=11,  # expects 11 inputs
                      activation='sigmoid'))
nn_keras_tf.add(Dense((12*11),
                      input_dim=11,  # expects 22 inputs
                      activation='sigmoid'))
nn_keras_tf.add(Dense(1, activation='linear'))
# since we are using keras to call TF we need to compile the model we designed in keras
# for it to be into the TF framework.
# print(dir(tf.keras.optimizers))
# print(dir(tf.keras.losses))
# print(dir(tf.keras.metrics))
nn_keras_tf.compile(loss=squared_loss, optimizer='adam')
nn_keras_tf.fit(x_train, y_train, epochs=500, verbose=0)
nn_keras_tf_y = train_Y_scaler.inverse_transform(nn_keras_tf.predict(train_X_scaler.transform(X_test)))
nn_keras_tf_rmse = rmse(y_test, nn_keras_tf_y, squared=False)
nn_keras_tf_mape = mape(y_true=y_test, y_pred=nn_keras_tf_y)
nn_keras_tf_score = score(y_true=y_test, y_pred=nn_keras_tf_y)
results['NN Keras TF'] = [nn_keras_tf_rmse, nn_keras_tf_mape, nn_keras_tf_score]
model_predictions['NN Keras TF'] = np.ravel(nn_keras_tf_y)

# from keras.utils.vis_utils import plot_model
# import graphviz
# import pydot
# plot_model(nn_keras_tf)


# Regression Tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import graphviz

tree_model = DecisionTreeRegressor(max_depth=4)
# Data without scaling.
tree_model.fit(X_train, Y_train)
tree_y = tree_model.predict(X_test)
tree_rmse = rmse(y_test, tree_y, squared=False)
tree_mape = mape(y_true=y_test, y_pred=tree_y)
tree_score = score(y_true=y_test, y_pred=tree_y)
results['Regression Tree'] = [tree_rmse, tree_mape, tree_score]
model_predictions['Regression Tree'] = np.ravel(tree_y)

# problem with the visualisation
'''plt.figure(figsize=(12, 10))
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
'''

# SVM (Regression)
from sklearn.svm import LinearSVR, SVR

# svr = LinearSVR(random_state=seed, tol=1e-4, max_iter=10000, loss='squared_epsilon_insensitive')
svr = SVR()

svr.fit(x_train, np.ravel(y_train))
svr_y = train_Y_scaler.inverse_transform(svr.predict(train_X_scaler.transform(X_test)))
svr_rmse = rmse(y_test, svr_y, squared=False)
svr_mape = mape(y_true=y_test, y_pred=svr_y)
svr_score = score(y_true=y_test, y_pred=svr_y)
results['SVR'] = [svr_rmse, svr_mape, svr_score]
model_predictions['SVR'] = np.ravel(svr_y)

# Checking the results

results_table = pd.DataFrame(results).round(decimals=2).transpose()
results_table.columns = ['RMSE', 'MAPE', 'Score']
results_table
# print(results_table.to_latex(index=True, multirow=True,
#                              label='tab:model_results',
#                              caption='Metric comparison different competing models.',
#                              position='h!'),
#       file=open(f'{path}/code_python/project1_wine/model_results.tex', "w"))

fig, axes = plt.subplots(4, 2, sharey=True, figsize=(12, 13))
fig.suptitle('(Normalised) Relative frequency - Comparison train/test/predicted samples', fontsize='xx-large')
axes[0, 0].set_title('Quality in training Sample')
sns.histplot(x=Y_train, ax=axes[0, 0], kde=True, stat="probability", color='red')
axes[0, 1].set_title('Quality in test Sample')
sns.histplot(x=y_test, ax=axes[0, 1], kde=True, stat="probability", color='red')
r_count = 1
col_count = 0
for column_i in model_predictions.columns:
    axes[r_count, col_count].set_title(f'{column_i} predicted quality')
    sns.histplot(data=model_predictions, x=column_i, ax=axes[r_count, col_count], kde=True, stat="probability")
    col_count += 1
    if col_count == 2:
        col_count = 0
        r_count += 1
    if r_count > 3:
        break
fig.subplots_adjust(hspace=0.4)
# plt.savefig(f'{path}/document_files/figs/prediction_histograms.pdf', format='pdf', bbox_inches='tight')
plt.show()

# SHAP and LIME analysis
import ipython_genutils
import shap

import lime


def inverse_transform_shap_values(scaler_x, scaler_y, shap_values_all):
    shap_values_all.data = scaler_x.inverse_transform(shap_values_all.data)
    shap_values_all.base_values = shap_values_all.base_values * scaler_y.scale_ + scaler_y.mean_
    shap_values_all.values = shap_values_all.values * scaler_y.scale_
    return shap_values_all


# for normal linear model
explainer_lr = shap.Explainer(normal_lr, masker=train_X_scaler.transform(X_test))
shap_values_lr = explainer_lr(train_X_scaler.transform(X_test))
shap_values_lr = inverse_transform_shap_values(train_X_scaler, train_Y_scaler, shap_values_lr)

indices_nlr_upper = np.where(nlr_y > 6.7)
indices_nlr_lower = np.where(nlr_y < 4.5)

from matplotlib.backends.backend_pdf import PdfPages

# Observations 11 and 247 are quite similar
fig = plt.figure(figsize=(12, 10))
pdf = PdfPages(f'{path}/document_files/figs/shape_nlr.pdf')
ax1 = fig.add_subplot(211)
ax1.set_title('Waterfall analysis - (normal) linear regression obs. 12')
shap.plots.waterfall(shap.Explanation(values=shap_values_lr[11],
                                      feature_names=X_train.columns.tolist()), show=False)
ax2 = fig.add_subplot(212)
ax2.set_title('Waterfall analysis - (normal) linear regression obs. 248')
shap.plots.waterfall(shap.Explanation(values=shap_values_lr[247],
                                      feature_names=X_train.columns.tolist()), show=False)

fig.subplots_adjust(hspace=0.6)
pdf.savefig(fig, bbox_inches='tight')
pdf.close()
plt.show()


# SVR
explainer_svr = shap.Explainer(svr.predict, masker=train_X_scaler.transform(X_test))
shap_values_svr = explainer_svr(train_X_scaler.transform(X_test))
shap_values_svr = inverse_transform_shap_values(train_X_scaler, train_Y_scaler, shap_values_svr)

indices_svr_upper = np.where(svr_y > 6.7)
indices_svr_lower = np.where(svr_y < 4.5)

# 11 and 64 are most similar observations
fig = plt.figure(figsize=(12, 10))
pdf = PdfPages(f'{path}/document_files/figs/shape_svr.pdf')
ax1 = fig.add_subplot(211)
ax1.set_title('Waterfall analysis - support vector regression obs. 12')
shap.plots.waterfall(shap.Explanation(values=shap_values_svr[11],
                                      feature_names=X_train.columns.tolist()), show=False)
ax2 = fig.add_subplot(212)
ax2.set_title('Waterfall analysis - support vector regression obs. 65')
shap.plots.waterfall(shap.Explanation(values=shap_values_svr[64],
                                      feature_names=X_train.columns.tolist()), show=False)

fig.subplots_adjust(hspace=0.6)
pdf.savefig(fig, bbox_inches='tight')
pdf.close()
plt.show()


## Part 2 - Learning
# Clustering

from sklearn.cluster import KMeans

kmeans_kwargs = {'init': 'random', 'n_init': 10, 'max_iter': 500, 'random_state': seed}

sse = []
log_sse = []
sse_scaled = []
log_sse_scaled = []

max_clusters = 12

for cluster_n in range(2, max_clusters):
    kmeans_model = KMeans(n_clusters=cluster_n, **kmeans_kwargs)
    kmeans_model_scaled = KMeans(n_clusters=cluster_n, **kmeans_kwargs)
    # for the clustering we do not use the quality, since we will not know this ad priori
    kmeans_model.fit(X_train)  # initial dataset
    kmeans_model_scaled.fit(x_train)  # scaled dataset
    sse.append(kmeans_model.inertia_)
    sse_scaled.append(kmeans_model_scaled.inertia_)
    log_sse.append(np.log10(kmeans_model.inertia_))
    log_sse_scaled.append(np.log10(kmeans_model_scaled.inertia_))

sse_dict = {0: sse, 1: sse_scaled, 2: log_sse, 3: log_sse_scaled}
sse_names_dict = {0: 'SSE', 1: 'SSE (scaled features)', 2: 'log SSE', 3: 'log SSE (scaled features)'}

# Elbow method visual
plt.style.use('fivethirtyeight')
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Elbow method SSE and log SSE', fontsize='xx-large')
r_count = 0
col_count = 0
for sse_results in range(len(sse_dict)):
    axes[r_count, col_count].plot(range(2, max_clusters), sse_dict[sse_results])
    axes[r_count, col_count].set_xticks(range(2, max_clusters))
    axes[r_count, col_count].set_xlabel('Number of clusters')
    axes[r_count, col_count].set_ylabel(sse_names_dict[sse_results])
    r_count += 1
    if r_count > 1:
        col_count += 1
        r_count = 0
plt.tight_layout()
# plt.savefig(f'{path}/document_files/figs/n_clusters_elbow.pdf', format='pdf', bbox_inches='tight')
plt.show()

plt.style.use('default')

# let us use 5 clusters
kmeans_model = KMeans(n_clusters=11, **kmeans_kwargs)
kmeans_model.fit(X_train)
kmeans_model_clusters = kmeans_model.labels_

# let us do a PCA analysis to try and plot the clusters in the PCs
from sklearn.decomposition import PCA

pca_init = PCA(n_components=6)
pca_model = pca_init.fit(x_train)
pca_x_train = pd.DataFrame(pca_model.transform(x_train))
pca_x_train.columns = ['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5', 'PC 6']

# let us check the % of the variance we can get by using x PCs
PCs = range(1, pca_model.n_components_+1)
PCs_var = 100*pca_model.explained_variance_ratio_
PCs_cum_var = np.cumsum(PCs_var).round(2)
# fig = plt.figure(figsize=(12, 10))
fig, ax1 = plt.subplots(figsize=(9, 6))
fig.suptitle('PC and their explained variance')
# left y-axis
ax1.set_xlabel('Principal Components')
ax1.bar(PCs, PCs_var, color='tab:blue')
ax1.set_ylabel('Variance %')
ax1.set_xticks(PCs)
# right y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Cumulated variance %')
ax2.plot(PCs, PCs_cum_var, color='black')
# for index, value in enumerate(PCs_cum_var):
#     ax2.annotate(str(value), xy=(index+1, value), xytext=(index+1, value), textcoords='offset pixels')
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(9, 6))
plt.scatter(pca_x_train['PC 1'], pca_x_train['PC 2'], c=kmeans_model_clusters)
plt.tight_layout()
plt.show()


clusters_df = pd.DataFrame(kmeans_model_clusters, index=X_train.index, columns=['Cluster'])
cluster_and_quality = clusters_df.join(Y_train)
high_grades = cluster_and_quality.iloc[np.where(cluster_and_quality['quality']>6)]
low_grades = cluster_and_quality.iloc[np.where(cluster_and_quality['quality']<5)]

# this is not necessarily working as I expected





















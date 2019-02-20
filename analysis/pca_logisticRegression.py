## This script is to run PCA and do logistic regression on the top PCs
############# load packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.decomposition import PCA

## generate train and test set
def preprocessing_train_test(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

## function to run LogisticRegression iteratively, and select features

def train_test (model, gene_names, X_train_input, y_train_input, X_test_input, y_test_input):
    # train
    model.fit(X_train_input, y_train_input)

    # Make prediction on the training data
    y_train_pred = model.predict(X_train_input)
    # p_train_pred = clf.predict_proba(X_train)[:,1]

    # Make predictions on test data
    y_test_pred = model.predict(X_test_input)
    # p_test_pred = clf.predict_proba(X_test)[:,1]

    # combine coeff across model
    Cortex_coeff = np.mean(np.abs(model.coef_), axis=0)

    df_coeffs = pd.DataFrame(list(zip(gene_names, Cortex_coeff.flatten(), list(range(X_train.shape[1]))))).sort_values(by=[1],
                                                                                                                 ascending=False)

    Cortex_confusion_train = confusion_matrix(y_train_input, y_train_pred)
    Cortex_confusion_test = confusion_matrix(y_test_input, y_test_pred)
    score_train = model.score(X_train_input, y_train_input)
    score_test = model.score(X_test_input, y_test_input)
    sparsity = np.mean(model.coef_ == 0) * 100

    training_score_list = [score_train]
    testing_score_list = [score_test]
    data_coeff_list = [df_coeffs]

    return(Cortex_confusion_train, Cortex_confusion_test, sparsity, training_score_list, testing_score_list, data_coeff_list)

## function to process data through pca
def pca_process(x_train, x_test, n_pc):
    pca = PCA(n_components = n_pc, random_state = 0)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    return (x_train, x_test, pca)


## PCA on ziqing's top 400 genes, then logisticRegression on top PCs
## read in ziqing data
ziqing_path = "/Users/yanwengong/Documents/winter_2018/shen_lab/project/scVI/data/feature_selection/original_ordered_files/"
ziqing_raw_latent = pd.read_csv(os.path.join(ziqing_path + "ziqing_raw_latent.csv"), index_col=0)
ziqing_raw_expression = ziqing_raw_latent.iloc[:, :400].values
ziqing_labels = ziqing_raw_latent.cell_labels.values
ziqing_gene_names = ziqing_raw_latent.columns[:400]


## standardize dataset
X_train, X_test, y_train, y_test = preprocessing_train_test(ziqing_raw_expression, ziqing_labels)
X_train.shape
# blow is to check the train-split result regarding labels
# unique, counts = np.unique(y_train, return_counts=True)
# print (np.asarray((unique, counts)).T)
#
# unique, counts = np.unique(y_test, return_counts=True)
# print (np.asarray((unique, counts)).T)
#
# unique, counts = np.unique(ziqing_labels, return_counts=True)
# print (np.asarray((unique, counts)).T)
## PCA
# Make an instance of the Model

X_train, X_test, model = pca_process(X_train, X_test, 10)

## logisticRegression on the 10 PCs
ziqing_PC_names = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
clf = LogisticRegression(penalty='l1', max_iter=200, random_state=0, solver='saga', multi_class='multinomial')
## TODO: by ziqing's train and test split is so biased?, test has lots of label 1??
ziqing_confusion_train_fun, ziqing_confusion_test_func, sparsity_func, training_score_PCA, testing_score_PCA, data_coeff_list_func = train_test(clf, ziqing_PC_names, X_train, y_train, X_test, y_test)
## Q: why it's not that PC1 has the highest weight?？
## TODO: check ConvergeneceWarning: the max_iter was reached which means the coeff_ did not converge

print(training_score_PCA, testing_score_PCA)

## note: top PCs here explained limited variance, likely because the 400 genes have already went through PCA and got selected
np.sum(model.explained_variance_ratio_)
list(model.explained_variance_ratio_)[0]
np.abs(model.components_[0])
len(model.components_)

## make a 10 x 10 table, each column is one PC, row is top 1 - 10 gene, contents are gene name

## for each PC, assign gene names to each weight, sort data by weight, pull out gene names of top 10
genes_in_PCs = pd.DataFrame({'gene_names':ziqing_gene_names})
for i in range(len(model.components_)):
    weight = np.abs(model.components_[i]);
    genes_in_PCs['PC'+str(i+1)] = weight

gene_number_list = []
for i in range(10):
    gene_number_list.append('gene'+ str(i+1))

genes_in_top_PCs = pd.DataFrame({'Name':gene_number_list})
for j in range(1,len(genes_in_PCs.columns),1):
    top10_genes = genes_in_PCs.iloc[:,[0,j]].sort_values(by = 'PC'+str(j), ascending=False).iloc[:10, 0].values
    genes_in_top_PCs['PC'+str(j)] = top10_genes

explained_variance_ratio_list = ['variance explained']
for i in range(len(list(model.explained_variance_ratio_))):
    explained_variance_ratio_list.append(list(model.explained_variance_ratio_)[i])

explained_variance_ratio_series = pd.Series(explained_variance_ratio_list, index = genes_in_top_PCs.columns)
genes_in_top_PCs_final = genes_in_top_PCs.append(explained_variance_ratio_series,  ignore_index=True)
pd.DataFrame(genes_in_top_PCs_final).to_csv(os.path.join(ziqing_path + "ziqing_genes_in_top10PCs_varianceExplained.csv"))


## PCA on Cortex data, and do logisticRegression based on top10 PCs
## read in cortex data
cortex_path = "/Users/yanwengong/Documents/winter_2018/shen_lab/project/scVI/data/feature_selection/original_ordered_files/cortex/"
cortex_raw_latent = pd.read_csv(os.path.join(cortex_path + "cortex_raw_latent.csv"), index_col=0)
cortex_raw_expression = cortex_raw_latent.iloc[:, :558].values
cortex_labels = cortex_raw_latent.cell_labels.values
cortex_gene_names = cortex_raw_latent.columns[:558]

## standardize dataset
X_train, X_test, y_train, y_test = preprocessing_train_test(cortex_raw_expression, cortex_labels)
X_train.shape
## PCA
# Make an instance of the Model
X_train, X_test, model = pca_process(X_train, X_test, 10)

## logisticRegression on the 10 PCs
c_PC_names = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
clf = LogisticRegression(penalty='l1', max_iter=200, random_state=0, solver='saga', multi_class='multinomial')
c_confusion_train_fun, c_confusion_test_func, sparsity_func, training_score_PCA, testing_score_PCA, data_coeff_list_func = train_test(clf, c_PC_names, X_train, y_train, X_test, y_test)
## Q: why it's not that PC1 has the highest weight?？
## TODO: check ConvergeneceWarning: the max_iter was reached which means the coeff_ did not converge

print(training_score_PCA, testing_score_PCA)
np.sum(model.explained_variance_ratio_)

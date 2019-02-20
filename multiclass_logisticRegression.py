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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV

## This is script to do logistic regression on data and also try iteratively select features

## generate train and test set
def preprocessing_train_test(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


## function to run LogisticRegression iteratively, and select features

def train_test_iteration (model, gene_names, X_train_input, y_train_input, X_test_input, y_test_input):
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

    for i in [300, 200, 100, 50, 20, 10]:
        selected_gene_index = list(df_coeffs.iloc[:i, 2])
        selected_gene_name = df_coeffs.iloc[:i, 0]

        ## train another logistic regression
        # y_train = y_train.iloc[:, selected_gene_index]
        model.fit(X_train_input[:, selected_gene_index], y_train_input)
        # Make prediction
        #y_train_pred = model.predict(X_train_input[:, selected_gene_index])
        #y_test_pred = model.predict(X_test_input[:, selected_gene_index])
        Cortex_coeff = np.mean(np.abs(model.coef_), axis=0)
        df_coeffs = pd.DataFrame(
            list(zip(selected_gene_name, Cortex_coeff.flatten(), selected_gene_index))).sort_values(by=[1],
                                                                                                    ascending=False)
        score_train = model.score(X_train_input[:, selected_gene_index], y_train_input)
        score_test = model.score(X_test_input[:, selected_gene_index], y_test_input)

        training_score_list.append(score_train)
        testing_score_list.append(score_test)
        data_coeff_list.append(df_coeffs)


    return(Cortex_confusion_train, Cortex_confusion_test, sparsity, training_score_list, testing_score_list, data_coeff_list)

## function to plot train/test score vs number of gene_names
def plot_score_vs_numberOfGenes(train_score, test_score, gene_number_list):
    d = {'train_score': train_score, 'test_score': test_score, 'n_genes': gene_number_list}
    df = pd.DataFrame(data = d)
    df_melt = pd.melt(df, id_vars = [('n_genes')], value_vars = ['train_score', 'test_score'], var_name='type', value_name='score')
    ax = sns.lineplot(x = "n_genes", y = "score", hue = "type", data = df_melt)
    return df_melt


## read in cortex data
cortex_path = "/Users/yanwengong/Documents/winter_2018/shen_lab/project/scVI/data/feature_selection/original_ordered_files/cortex/"
cortex_raw_latent = pd.read_csv(os.path.join(cortex_path + "cortex_raw_latent.csv"), index_col=0)
cortex_raw_expression = cortex_raw_latent.iloc[:, :558].values
cortex_labels = cortex_raw_latent.cell_labels.values
cortex_gene_names = cortex_raw_latent.columns[:558]


## run the function
clf = LogisticRegression(penalty='l1', max_iter=100, random_state=0, solver='saga', multi_class='multinomial')
X_train, X_test, y_train, y_test = preprocessing_train_test(cortex_raw_expression, cortex_labels)
Cortex_confusion_train_fun, Cortex_confusion_test_func, sparsity_func, training_score_list_func, testing_score_list_func, data_coeff_list_func = train_test_iteration(clf, cortex_gene_names, X_train, y_train, X_test, y_test)


## read in ziqing data
ziqing_path = "/Users/yanwengong/Documents/winter_2018/shen_lab/project/scVI/data/feature_selection/original_ordered_files/"
ziqing_raw_latent = pd.read_csv(os.path.join(ziqing_path + "ziqing_raw_latent.csv"), index_col=0)
ziqing_raw_expression = ziqing_raw_latent.iloc[:, :400].values
ziqing_labels = ziqing_raw_latent.cell_labels.values
ziqing_gene_names = ziqing_raw_latent.columns[:400]


## run the function, ziqing
clf = LogisticRegression(penalty='l1', max_iter=100, random_state=0, solver='saga', multi_class='multinomial')
X_train, X_test, y_train, y_test = preprocessing_train_test(ziqing_raw_expression, ziqing_labels)
ziqing_confusion_train_fun, ziqing_confusion_test_func, sparsity_func, training_score_list_func, testing_score_list_func, data_coeff_list_func = train_test_iteration(clf, ziqing_gene_names, X_train, y_train, X_test, y_test)

## plot score vs number of genes
ziqing_score_melt = plot_score_vs_numberOfGenes(training_score_list_func, testing_score_list_func, [321, 300, 200, 100, 50, 20, 10])


## run ziqing's sample with Zn
ziqing_raw_expression_with_latent = ziqing_raw_latent.iloc[:, :-1].values
ziqing_gene_names_latent = ziqing_raw_latent.columns[:-1]

clf = LogisticRegression(penalty='l1', max_iter=200, random_state=0, solver='saga', multi_class='multinomial')
X_train, X_test, y_train, y_test = preprocessing_train_test(ziqing_raw_expression_with_latent, ziqing_labels)
ziqing_confusion_train_fun_zn, ziqing_confusion_test_func_zn, sparsity_func, training_score_list_func_zn, testing_score_list_func_zn, data_coeff_list_func_ziqing_zn = train_test_iteration(clf, ziqing_gene_names_latent, X_train, y_train, X_test, y_test)
data_coeff_list_func_ziqing_zn[-2]


ziqing_score_melt_zn = plot_score_vs_numberOfGenes(training_score_list_func_zn, testing_score_list_func_zn, [321, 300, 200, 100, 50, 20, 10])
ziqing_score_melt_zn.type = ziqing_score_melt_zn.type.str.replace('score', 'score_zn')
## combine ziqing_score_melt and ziqing_score_melt_zn
ziqing_score_combined = pd.concat([ziqing_score_melt, ziqing_score_melt_zn])


ax = sns.lineplot(x = "n_genes", y = "score", hue = "type", data = ziqing_score_combined)
plt.title("Ziqing Data with or without Zn")
plt.legend(loc = 10)
plt.savefig("/Users/yanwengong/Documents/winter_2018/shen_lab/project/scVI/plot/ziqing_correct_order/ziqing_score_with_zn.png")

## This script is to organzie dataset

############# load packages
import os
import numpy as np
import pandas as pd

##### Ziqing's data
## read in ziqing data
ziqing_path = "/Users/yanwengong/Documents/winter_2018/shen_lab/project/scVI/data/feature_selection/original_ordered_files/"

ziqing_raw = pd.io.parsers.read_csv(ziqing_path + "ziqing_raw_expression.csv", index_col=0) ## 321 * 400
ziqing_latent = pd.io.parsers.read_csv(ziqing_path + "all_latent.csv", index_col=0) ## 321 * 10
ziqing_gene_names = list(pd.read_csv(ziqing_path + "top_400_pca_genes.csv").iloc[:,0])
ziqing_cell_labels = list(pd.read_csv(ziqing_path + "ziqing_cell_labels.csv", index_col=0).iloc[:,0])
## add Zn at the end of gene names
ziqing_gene_names_Zn = ziqing_gene_names + ['Zn1', 'Zn2', 'Zn3', 'Zn4', 'Zn5', 'Zn6', 'Zn7', 'Zn8', 'Zn9', 'Zn10']
ziqing_raw_latent = pd.concat([ziqing_raw, ziqing_latent], axis = 1)
## add gene names as the col names
ziqing_raw_latent.columns = ziqing_gene_names_Zn
## add cell labels as a new col
ziqing_raw_latent['cell_labels'] = ziqing_cell_labels

## output ziqing's
pd.DataFrame(ziqing_raw_latent).to_csv(os.path.join(ziqing_path + "ziqing_raw_latent.csv"))

###### Cortest data_list
cortex_path = "/Users/yanwengong/Documents/winter_2018/shen_lab/project/scVI/data/feature_selection/original_ordered_files/cortex/"
cortex_raw = pd.io.parsers.read_csv(cortex_path + "original_cortext.csv", index_col=0) ## 3005 * 558
cortex_latent = pd.io.parsers.read_csv(cortex_path + "all_latent.csv", index_col=0) ## 3005 * 10
cortex_gene_names = list(pd.read_csv(cortex_path + "Cortex_gene_names.csv").iloc[:,1])
cortex_cell_labels = list(pd.read_csv(cortex_path + "labels_cortext.csv", index_col=0).iloc[:,0])
## add Zn at the end of gene names
cortex_gene_names_Zn = cortex_gene_names + ['Zn1', 'Zn2', 'Zn3', 'Zn4', 'Zn5', 'Zn6', 'Zn7', 'Zn8', 'Zn9', 'Zn10']
cortex_raw_latent = pd.concat([cortex_raw, cortex_latent], axis = 1)
## add gene names as the col names
cortex_raw_latent.columns = cortex_gene_names_Zn
## add cell labels as a new col
cortex_raw_latent['cell_labels'] = cortex_cell_labels
pd.DataFrame(cortex_raw_latent).to_csv(os.path.join(cortex_path + "cortex_raw_latent.csv"))

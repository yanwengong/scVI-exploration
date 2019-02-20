## This is the script to output two cells that in labels but not in gene expression file;
## and four samples in expression but not labels_cortext
import os
import numpy as np
import pandas as pd
## read in expression files and label files
path='/Users/yanwengong/Documents/winter_2018/shen_lab/project/scVI/data/feature_selection/'

expression_file  = pd.read_csv(os.path.join(path + "rawcounts_d3579_325cells_updated.csv"), index_col=0)
cell_names_expression = list(expression_file.columns)

cell_names_labelfile  = list(pd.read_csv(os.path.join(path + "ziqing_expression_data_cellLabels.csv")).Cell_ID)

np.setdiff1d(cell_names_expression,cell_names_labelfile)
np.setdiff1d(cell_names_labelfile, cell_names_expression)

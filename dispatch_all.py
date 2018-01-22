import multiprocessing as mp
import time as tm
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA

def pca_each_csv(act_file_id = 1):
    df = pd.read_csv('ACT{0}_competition_training.csv'.format(act_file_id), header = 0)
    #if there are duplicated row
    if len((df.duplicated())[df.duplicated() == True]) == 0:
        df.drop_duplicates()
        print("There are row duplicates!")
    #Select nonzero column
    nonzero_stack = df.apply(lambda col: np.count_nonzero(col) != 0, axis =0)
    x = (df.iloc[:, np.array(nonzero_stack) == True]).iloc[:, 2:]
    #normalize the data
    x_normalized = preprocessing.normalize(x, norm = 'l2', axis = 0)
    #PCA Tunning
    guess_components = 70
    variance_tunning = 0.70
    iter_flag = True
    while iter_flag:
        pca = PCA(n_components = guess_components, svd_solver = 'auto')
        pca.fit(x_normalized)
        if sum(pca.explained_variance_ratio_) < variance_tunning:
            guess_components += 5
        else: 
            iter_flag = False

    return "For {0:.2f}% variance, there are {1} components.\n".format(100* sum(pca.explained_variance_ratio_), guess_components)

if __name__ == "__main__":
    t = tm.time()
    pool = mp.Pool(processes = 4)
    result = pool.map(pca_each_csv, [1,2,3,4,5])
    file = open('result.txt', 'w')
    file.write(result)
    file.write("Total time: {0}".format(tm.time()-t) )
    file.close()
    
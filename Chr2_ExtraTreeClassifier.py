# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:31:24 2019

@author: Hira
"""

import numpy as np

def strip_first_col(fname, delimiter=None):
    with open(fname, 'r') as fin:
        for line in fin:
            try:
               yield line.split(delimiter, 1)[1]
            except IndexError:
               continue
neut = np.loadtxt(strip_first_col("neut.fvec"), skiprows=(1))
soft = np.loadtxt(strip_first_col("soft.fvec"), skiprows=(1))
hard = np.loadtxt(strip_first_col("hard.fvec"), skiprows=(1))
linkSoft = np.loadtxt(strip_first_col("linkedSoft.fvec"), skiprows=(1))
linkHard = np.loadtxt(strip_first_col("linkedHard.fvec"), skiprows=(1))

X = np.concatenate((neut,soft,hard,linkSoft,linkHard))
y = [0]*len(neut) + [1]*len(soft) + [2]*len(hard) + [3]*len(linkSoft) + [4]*len(linkHard) 
Y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1)

from sklearn.ensemble import ExtraTreesClassifier


lf = ExtraTreesClassifier(n_estimators=100, 
                           max_depth=10, max_features = 3,
                           min_samples_split=3,
                           min_samples_leaf = 3, bootstrap = True,
                           criterion = "gini")    
clf = clf.fit(X_train, Y_train)
preds=clf.predict(X_test)
counts=[[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]]
for i in range(len(Y_test)):
    counts[Y_test[i]][preds[i]] += 1
counts.reverse()
classOrderLs=['neut','soft','hard','linkedSoft','linkedHard']
import collections

X1 = np.loadtxt(strip_first_col("chr2_200kb_134M_140M_CEU3.fvec"), skiprows=(1))
clfbot1 = clf.predict(X1)
bott1 = np.array(clfbot1)
values = collections.Counter(bott1)
print(values)

np.savetxt('chromosomeExtratreeSecondTrial.out', clfbot1)


#Values of the parameters are changeable. These commented parameters can be used too.
#==============================================================================
# maxMaxFeatures = len(X[0])
# param_grid_forest = {"max_depth": [3, 10, None],
#               "max_features": [1, 3, int(maxMaxFeatures**0.5), maxMaxFeatures],
#               "min_samples_split": [1, 3, 10],
#               "min_samples_leaf": [1, 3, 10],
#               "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"]}



c
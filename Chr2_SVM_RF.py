# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 23:23:31 2019

@author: Hira
"""

#SVM

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

from sklearn import svm

clf = svm.SVC(kernel='rbf', gamma=0.1, C=1)
clf = clf.fit(X_train, Y_train)
preds=clf.predict(X_test)
counts=[[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]]
for i in range(len(Y_test)):
    counts[Y_test[i]][preds[i]] += 1
counts.reverse()

import collections

X1 = np.loadtxt(strip_first_col("chr2_200kb_134M_140M_CEUpr2.fvec"), skiprows=(1))
clfbot1 = clf.predict(X1)
bott1 = np.array(clfbot1)
values = collections.Counter(bott1)
print(values)


np.savetxt('chromosomeprSVM.out', clfbot1)


#RF
#SVM

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


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100,n_jobs=10)
clf = clf.fit(X_train, Y_train)
preds=clf.predict(X_test)
counts=[[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]]
for i in range(len(Y_test)):
    counts[Y_test[i]][preds[i]] += 1
counts.reverse()
classOrderLs=['equil','soft','hard','linkedSoft','linkedHard']




import collections

X1 = np.loadtxt(strip_first_col("chr2_200kb_134M_140M_CEUpr2.fvec"), skiprows=(1))
len(X1[1])
clfbot1 = clf.predict(X1)
bott1 = np.array(clfbot1)
values = collections.Counter(bott1)
print(values)


np.savetxt('chromosomeprSVM.out', clfbot1)

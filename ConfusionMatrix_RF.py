# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 23:03:53 2019

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

X1 = np.loadtxt(strip_first_col("neut.fvec"), skiprows=(1))
X2 = np.loadtxt(strip_first_col("soft.fvec"), skiprows=(1))
X3 = np.loadtxt(strip_first_col("hard.fvec"), skiprows=(1))
X4 = np.loadtxt(strip_first_col("linkedSoft.fvec"), skiprows=(1))
X5 = np.loadtxt(strip_first_col("linkedHard.fvec"), skiprows=(1))

X = np.concatenate((X1,X2,X3,X4,X5))


y = [0]*len(X1) + [1]*len(X2) + [2]*len(X3) + [3]*len(X4) + [4]*len(X5)
Y = np.array(y)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1)





#These two functions (taken from scikit-learn.org) plot the decision boundaries for a classifier.
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def make_meshgrid(x, y, h=.05):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

#Let's do the plotting
import matplotlib.pyplot as plt


from sklearn.preprocessing import normalize

#here's the confusion matrix function




#here's the confusion matrix function
def makeConfusionMatrixHeatmap(data, title, trueClassOrderLs, predictedClassOrderLs, ax):
    data = np.array(data)
    data = normalize(data, axis=1, norm='l1')
    heatmap = ax.pcolor(data, cmap=plt.cm.Greens, vmin=0.0, vmax=1.0)

    for i in range(len(predictedClassOrderLs)):
        for j in reversed(range(len(trueClassOrderLs))):
            val = 100*data[j, i]
            if val > 50:
                c = '0.9'
            else:
                c = 'black'
            ax.text(i + 0.5, j + 0.5, '%.2f%%' % val, horizontalalignment='center', verticalalignment='center', color=c, fontsize=8)

    cbar = plt.colorbar(heatmap, cmap=plt.cm.Greens, ax=ax)
    cbar.set_label("Fraction of simulations assigned to class", rotation=270, labelpad=20, fontsize=11)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.axis('tight')
    ax.set_title(title)

    #labels
    ax.set_xticklabels(predictedClassOrderLs, minor=False, fontsize=8, rotation=45)
    ax.set_yticklabels(reversed(trueClassOrderLs), minor=False, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")



#now the actual work
#first get the predictions

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100,n_jobs=10)
clf = clf.fit(X_train, Y_train)
preds=clf.predict(X_test)


counts=[[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]]
len
for i in range(len(Y_test)):
    counts[Y_test[i]][preds[i]] += 1
counts.reverse()
classOrderLs=['equil','soft','hard','linkedSoft','linkedHard']

#now do the plotting
fig,ax= plt.subplots(1,1)
makeConfusionMatrixHeatmap(counts, "Confusion matrix", classOrderLs, classOrderLs, ax)
plt.show()
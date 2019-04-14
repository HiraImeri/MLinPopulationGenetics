# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 23:52:50 2019

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
btlnck = np.loadtxt("originalmodeltrain_10.fvec", skiprows = 1)
len(linkHard)

X = np.concatenate((neut,soft,hard,linkSoft,linkHard,btlnck))
y = [0]*len(neut) + [1]*len(soft) + [2]*len(hard) + [3]*len(linkSoft) + [4]*len(linkHard) + [5]*len(btlnck)
Y = np.array(y)
len(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1)

from sklearn import svm

clf = svm.SVC(kernel='rbf', gamma=0.1, C=1)
clf = clf.fit(X_train, Y_train)
preds=clf.predict(X_test)
counts=[[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]]
for i in range(len(Y_test)):
    counts[Y_test[i]][preds[i]] += 1
counts.reverse()
classOrderLs=['neut','soft','hard','linkedSoft','linkedHard','btlnck']


#Calculating the percentage of sweeps
#predict the counts( how many 0,1,2,3,4 which correspond to models ) and calculate the percentage


import collections
from pandas import DataFrame

pop = ["Pop1","Pop2","Pop3","Pop4","Pop5","Pop6","Pop7","Pop8","Pop9","Pop10",
       "Pop11","Pop12","Pop13","Pop14","Pop15","Pop16","Pop17","Pop18","Pop19","Pop20"]


#BOTTLENECK 1

X1 = np.loadtxt("originalmodel_1.fvec", skiprows=(1))
clfbot1 = clf.predict(X1)
bott1 = np.array(clfbot1)
values = collections.Counter(bott1)
print(values)


#BOTTLENECK2
X2 = np.loadtxt("originalmodel_2.fvec", skiprows=(1))
clfbot2 = clf.predict(X2)
bott2 = np.array(clfbot2)
values = collections.Counter(bott2)
print(values)

#BOTTLENECK3
X3 = np.loadtxt("originalmodel_3.fvec", skiprows=(1))
clfbot3 = clf.predict(X3)
bott3 = np.array(clfbot3)
values = collections.Counter(bott3)
print(values)

#BOTTLENECK4
X4 = np.loadtxt("originalmodel_4.fvec", skiprows=(1))
clfbot4 = clf.predict(X4)
#take counts how many 0,1,2,3,4 and than 
bott4 = np.array(clfbot4)
values = collections.Counter(bott4)
print(values)


X5 = np.loadtxt("originalmodel_5.fvec", skiprows=(1))
clfbot5 = clf.predict(X5)
bott5 = np.array(clfbot5)
values = collections.Counter(bott5)
print(values)
     

X6 = np.loadtxt("originalmodel_6.fvec", skiprows=(1))
clfbot6 = clf.predict(X6)
bott6 = np.array(clfbot6)
values = collections.Counter(bott6)
print(values)


#BOTTLENECK2
X7 = np.loadtxt("originalmodel_7.fvec", skiprows=(1))
clfbot7 = clf.predict(X7)
bott7 = np.array(clfbot7)
values = collections.Counter(bott7)
print(values)

#BOTTLENECK3
X8 = np.loadtxt("originalmodel_8.fvec", skiprows=(1))
clfbot8 = clf.predict(X8)
bott8 = np.array(clfbot8)
values = collections.Counter(bott8)
print(values)

#BOTTLENECK4
X9 = np.loadtxt("originalmodel_9.fvec", skiprows=(1))
clfbot9 = clf.predict(X9)
#take counts how many 0,1,2,3,4 and than 
bott9 = np.array(clfbot9)
values = collections.Counter(bott9)
print(values)


X10 = np.loadtxt("originalmodel_10.fvec", skiprows=(1))
clfbot10 = clf.predict(X10)
bott10 = np.array(clfbot10)
values = collections.Counter(bott10)
print(values)  

#BOTTLENECK 1

X11 = np.loadtxt("originalmodel_11.fvec", skiprows=(1))
clfbot11 = clf.predict(X11)
bott11 = np.array(clfbot11)
values = collections.Counter(bott11)
print(values)


#BOTTLENECK2
X12 = np.loadtxt("originalmodel_12.fvec", skiprows=(1))
clfbot12 = clf.predict(X12)
bott12 = np.array(clfbot12)
values = collections.Counter(bott12)
print(values)

#BOTTLENECK3
X13 = np.loadtxt("originalmodel_13.fvec", skiprows=(1))
clfbot13 = clf.predict(X13)
bott13 = np.array(clfbot13)
values = collections.Counter(bott13)
print(values)

#BOTTLENECK4
X14 = np.loadtxt("originalmodel_14.fvec", skiprows=(1))
clfbot14 = clf.predict(X14)
#take counts how many 0,1,2,3,4 and than 
bott14 = np.array(clfbot14)
values = collections.Counter(bott14)
print(values)


X15 = np.loadtxt("originalmodel_15.fvec", skiprows=(1))
clfbot15 = clf.predict(X15)
bott15 = np.array(clfbot15)
values = collections.Counter(bott15)
print(values)
     

X16 = np.loadtxt("originalmodel_16.fvec", skiprows=(1))
clfbot16 = clf.predict(X16)
bott16 = np.array(clfbot16)
values = collections.Counter(bott16)
print(values)


#BOTTLENECK2
X17 = np.loadtxt("originalmodel_17.fvec", skiprows=(1))
clfbot17 = clf.predict(X17)
bott17 = np.array(clfbot17)
values = collections.Counter(bott17)
print(values)

#BOTTLENECK3
X18 = np.loadtxt("originalmodel_18.fvec", skiprows=(1))
clfbot18 = clf.predict(X18)
bott18 = np.array(clfbot18)
values = collections.Counter(bott18)
print(values)

#BOTTLENECK4
X19 = np.loadtxt("originalmodel_19.fvec", skiprows=(1))
clfbot19 = clf.predict(X19)
#take counts how many 0,1,2,3,4 and than 
bott19 = np.array(clfbot19)
values = collections.Counter(bott19)
print(values)


X20 = np.loadtxt("originalmodel_20.fvec", skiprows=(1))
clfbot20 = clf.predict(X20)
bott20 = np.array(clfbot20)
values = collections.Counter(bott19)
print(values)

l0 = []
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []

df = DataFrame({'Models': pop, 'Neutral': l0, 'Soft': l1, 'Hard': l2, 'LinkedSoft': l3, 'LinkedHard': l4, 'Btlnck': l5}, 
        columns = ['Models', 'Neutral', 'Soft', 'Hard', 'LinkedSoft', 'LinkedHard', 'Btlnck'])
df


df.to_excel('20btlnOriginalModelSVMBottlencktrain.xlsx', sheet_name='sheet1', index=False)


#RF



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

X = np.concatenate((neut,soft,hard,linkSoft,linkHard,btlnck))
y = [0]*len(neut) + [1]*len(soft) + [2]*len(hard) + [3]*len(linkSoft) + [4]*len(linkHard) + [5]*len(btlnck)
Y = np.array(y)


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



#Calculating the percentage of sweeps
#predict the counts( how many 0,1,2,3,4 which correspond to models ) and calculate the percentage


import collections
from pandas import DataFrame

pop = ["Pop1","Pop2","Pop3","Pop4","Pop5","Pop6","Pop7","Pop8","Pop9","Pop10",
       "Pop11","Pop12","Pop13","Pop14","Pop15","Pop16","Pop17","Pop18","Pop19","Pop20"]


#BOTTLENECK 1

X1 = np.loadtxt("originalmodel_1.fvec", skiprows=(1))
clfbot1 = clf.predict(X1)
bott1 = np.array(clfbot1)
values = collections.Counter(bott1)
print(values)


#BOTTLENECK2
X2 = np.loadtxt("originalmodel_2.fvec", skiprows=(1))
clfbot2 = clf.predict(X2)
bott2 = np.array(clfbot2)
values = collections.Counter(bott2)
print(values)

#BOTTLENECK3
X3 = np.loadtxt("originalmodel_3.fvec", skiprows=(1))
clfbot3 = clf.predict(X3)
bott3 = np.array(clfbot3)
values = collections.Counter(bott3)
print(values)

#BOTTLENECK4
X4 = np.loadtxt("originalmodel_4.fvec", skiprows=(1))
clfbot4 = clf.predict(X4)
#take counts how many 0,1,2,3,4 and than 
bott4 = np.array(clfbot4)
values = collections.Counter(bott4)
print(values)


X5 = np.loadtxt("originalmodel_5.fvec", skiprows=(1))
clfbot5 = clf.predict(X5)
bott5 = np.array(clfbot5)
values = collections.Counter(bott5)
print(values)
     

X6 = np.loadtxt("originalmodel_6.fvec", skiprows=(1))
clfbot6 = clf.predict(X6)
bott6 = np.array(clfbot6)
values = collections.Counter(bott6)
print(values)


#BOTTLENECK2
X7 = np.loadtxt("originalmodel_7.fvec", skiprows=(1))
clfbot7 = clf.predict(X7)
bott7 = np.array(clfbot7)
values = collections.Counter(bott7)
print(values)

#BOTTLENECK3
X8 = np.loadtxt("originalmodel_8.fvec", skiprows=(1))
clfbot8 = clf.predict(X8)
bott8 = np.array(clfbot8)
values = collections.Counter(bott8)
print(values)

#BOTTLENECK4
X9 = np.loadtxt("originalmodel_9.fvec", skiprows=(1))
clfbot9 = clf.predict(X9)
#take counts how many 0,1,2,3,4 and than 
bott9 = np.array(clfbot9)
values = collections.Counter(bott9)
print(values)


X10 = np.loadtxt("originalmodel_10.fvec", skiprows=(1))
clfbot10 = clf.predict(X10)
bott10 = np.array(clfbot10)
values = collections.Counter(bott10)
print(values)  

#BOTTLENECK 1

X11 = np.loadtxt("originalmodel_11.fvec", skiprows=(1))
clfbot11 = clf.predict(X11)
bott11 = np.array(clfbot11)
values = collections.Counter(bott11)
print(values)


#BOTTLENECK2
X12 = np.loadtxt("originalmodel_12.fvec", skiprows=(1))
clfbot12 = clf.predict(X12)
bott12 = np.array(clfbot12)
values = collections.Counter(bott12)
print(values)

#BOTTLENECK3
X13 = np.loadtxt("originalmodel_13.fvec", skiprows=(1))
clfbot13 = clf.predict(X13)
bott13 = np.array(clfbot13)
values = collections.Counter(bott13)
print(values)

#BOTTLENECK4
X14 = np.loadtxt("originalmodel_14.fvec", skiprows=(1))
clfbot14 = clf.predict(X14)
#take counts how many 0,1,2,3,4 and than 
bott14 = np.array(clfbot14)
values = collections.Counter(bott14)
print(values)


X15 = np.loadtxt("originalmodel_15.fvec", skiprows=(1))
clfbot15 = clf.predict(X15)
bott15 = np.array(clfbot15)
values = collections.Counter(bott15)
print(values)
     

X16 = np.loadtxt("originalmodel_16.fvec", skiprows=(1))
clfbot16 = clf.predict(X16)
bott16 = np.array(clfbot16)
values = collections.Counter(bott16)
print(values)


#BOTTLENECK2
X17 = np.loadtxt("originalmodel_17.fvec", skiprows=(1))
clfbot17 = clf.predict(X17)
bott17 = np.array(clfbot17)
values = collections.Counter(bott17)
print(values)

#BOTTLENECK3
X18 = np.loadtxt("originalmodel_18.fvec", skiprows=(1))
clfbot18 = clf.predict(X18)
bott18 = np.array(clfbot18)
values = collections.Counter(bott18)
print(values)

#BOTTLENECK4
X19 = np.loadtxt("originalmodel_19.fvec", skiprows=(1))
clfbot19 = clf.predict(X19)
#take counts how many 0,1,2,3,4 and than 
bott19 = np.array(clfbot19)
values = collections.Counter(bott19)
print(values)


X20 = np.loadtxt("originalmodel_20.fvec", skiprows=(1))
clfbot20 = clf.predict(X20)
bott20 = np.array(clfbot20)
values = collections.Counter(bott19)
print(values)

l0 = []
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []

df = DataFrame({'Models': pop, 'Neutral': l0, 'Soft': l1, 'Hard': l2, 'LinkedSoft': l3, 'LinkedHard': l4, 'Btlnck': l5}, 
        columns = ['Models', 'Neutral', 'Soft', 'Hard', 'LinkedSoft', 'LinkedHard', 'Btlnck'])
df


df.to_excel('20btlnOriginalModelRFBottlencktrain.xlsx', sheet_name='sheet1', index=False)


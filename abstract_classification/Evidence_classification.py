from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

__author__ = 'mikhail'

import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier


train = open('/home/mikhail/Documents/research/hierarchical_classification/JUSTEBM2015.csv')  # check the structure of this file!

X = []
Yr = []
Y = []

csv_train = csv.reader(train)


for line in csv_train:
    '''
    X.append(line[0])
    Y.append(line[1])
      '''
    X.append(line[0])
    Y.append(line[1])
    if 'non-randomized' in line[1]:
       Yr.append('non-rand')
    else:
        if 'randomized' in line[1]:
            Yr.append('rand')



print("1")


preprocessing.LabelBinarizer().fit_transform()
Ybin = preprocessing.MultiLabelBinarizer().fit_transform(Y)


f = open('/home/mikhail/Documents/research/predict_result.csv', 'wt')
writer = csv.writer(f)
Ybr = np.array(Yr)
Yb = np.array(Y)

vectorizer = TfidfVectorizer(min_df=4, use_idf=True, smooth_idf=True,
                             stop_words='english', ngram_range=(1, 3),
                             strip_accents='unicode',
                             norm='l2')


X_train = vectorizer.fit_transform(X)

from sklearn.ensemble import RandomForestClassifier
ens = RandomForestClassifier(n_estimators=100)
'''
svm_mod = OneVsRestClassifier(SVC(probability=True))

ens_mod = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000, max_depth=3, max_features=None, criterion='entropy'))
'''
simple_svm = SVC(probability=True)

'''
from sklearn.cross_validation import KFold


kf = KFold(6153, n_folds=2)
'''

'''
for train, test in kf:
    print(train)
    print(test)
    X_tr, X_test = X_train[train], X_train[test]
    Yran_train, Yran_test = Ybr[np.array(train)], Ybr[np.array(test)]
    Ybi_train, Ybi_test = Yb[np.array(train)], Yb[np.array(test)]

print("2")
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
'''

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, Yr, test_size=0.33, random_state=42)
'''
ada = AdaBoostClassifier(ens, n_estimators=100)
est = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

ens = RandomForestClassifier(n_estimators=100)
ada = AdaBoostClassifier(ens, n_estimators=100)
'''

simple_svm.fit(X_train, y_train)


y_svm_predicted = simple_svm.predict(X_test)


print(y_svm_predicted)



'''

for i in y_svm_predicted:
    writer.writerow((i))

writer.writerow(("!!!!!!!!"))

print("3")
ada.fit(X_tr.toarray(), Ybi_train)

ada_predict = ada.predict_proba(X_test.toarray())

for i in ada_predict:
    writer.writerow((i))

writer.writerow(("!!!!!!!!"))

print("4")

get_res = ada_predict / y_svm_predicted

for i in get_res:
    writer.writerow((i))

writer.writerow(("!!!!!!!!"))

f.close()
'''

print "MODEL: RBF SVM\n"


print 'The precision for this classifier is ' + str(metrics.precision_score(y_test, y_svm_predicted))
print 'The recall for this classifier is ' + str(metrics.recall_score(y_test, y_svm_predicted))
print 'The f1 for this classifier is ' + str(metrics.f1_score(y_test, y_svm_predicted))
print 'The accuracy for this classifier is ' + str(metrics.accuracy_score(y_test, y_svm_predicted))

print 'The precision for this classifier is ' + str(metrics.precision_score(y_test, y_svm_predicted, average=None))
print 'The recall for this classifier is ' + str(metrics.recall_score(y_test, y_svm_predicted, average=None))
print 'The f1 for this classifier is ' + str(metrics.f1_score(y_test, y_svm_predicted))

print '\nHere is the classification report:'



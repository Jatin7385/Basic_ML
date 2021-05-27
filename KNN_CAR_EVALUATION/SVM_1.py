import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import  KNeighborsClassifier

#svm is a classifier

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target


x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

#print(x_train,y_train)
classes = ['malignant' 'benign']

clf = svm.SVC(kernel="linear", C=2) #C is for the soft margin
clf.fit(x_train,y_train)

y_prediction = clf.predict(x_test)

acc = metrics.accuracy_score(y_test,y_prediction)

clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(x_train,y_train)
y_knnPrediction = clf_knn.predict(x_test)

acc1 = metrics.accuracy_score(y_test,y_knnPrediction)

print(acc,acc1)



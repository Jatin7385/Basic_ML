from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

data=load_iris()
x=data.data
y=data.target
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x,y)
p=knn.predict([[3,5,4,2]])
if p==1:
    print("Versicolor")
elif p==0:
    print("Setosa")
elif p==2:
    print("Virginica")


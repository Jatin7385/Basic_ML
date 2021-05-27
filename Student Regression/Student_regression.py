import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")


data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3"

X = np.array(data.drop([predict],1))
Y = np.array(data[predict])

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train,y_train)

acc = linear.score(x_test,y_test)
print(acc)

#with open("studentmodel.pickle","wb") as f:
#        pickle.dump(linear,f)

#pickle_in = open("studentmodel.pickle","rb")
#linear = pickle.load(pickle_in)

print("Co: \n",linear.coef_)
print("Intercept: \n",linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])

plot = "G1" # Change this to G1, G2, studytime or absences to see other graphs
plt.scatter(data[plot], data["G3"])
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()
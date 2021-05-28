import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pickle

data = pd.read_csv("adult.data")

le = preprocessing.LabelEncoder()
age = le.fit_transform(list(data["age"]))
workclass = le.fit_transform(list(data["workclass"]))
fnlwgt = le.fit_transform(list(data["fnlwgt"]))
education = le.fit_transform(list(data["education"]))
Masters = le.fit_transform(list(data["Masters"]))
income = le.fit_transform(list(data["income"]))
marital_status = le.fit_transform(list(data["marital-status"]))
occupation = le.fit_transform(list(data["occupation"]))
relationship = le.fit_transform(list(data["relationship"]))
race = le.fit_transform(list(data["race"]))
sex = le.fit_transform(list(data["sex"]))
capital_gain = le.fit_transform(list(data["capital-gain"]))
capital_loss = le.fit_transform(list(data["capital-loss"]))
hours_per_week = le.fit_transform(list(data["hours-per-week"]))
native_country = le.fit_transform(list(data["native-country"]))

x = list(zip(age,workclass,fnlwgt,education,Masters,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country))
y = list(income)

maxacc = 0

n = 3

for i in range(10):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    model = KNeighborsClassifier(n_neighbors=n)
    n+=2
    model.fit(x_train,y_train)

    acc = model.score(x_test,y_test)
    print(acc)

    if(maxacc<acc):
        maxacc = acc
        with open("adultmodel.pickle", "wb") as f:
            pickle.dump(model,f)


print("Max accuracy = ",maxacc)

pickle_in = open("adultmodel.pickle","rb")
model = pickle.load(pickle_in)
predicted = model.predict(x_test)

names = [">50K","<=50K"]

for x in range(len(x_test)):
    print("Predicted: ",names[predicted[x]],"Data: ",x_test[x],"Actual: ",names[y_test[x]])
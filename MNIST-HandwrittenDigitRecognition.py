#Fetching Datasets from MNIST
from sklearn.datasets import fetch_openml
import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import LogisticRegression

mnist = fetch_openml('mnist_784')

x, y = mnist['data'], mnist['target']

"""
#Extracting some digit from the data
some_digit = x[36000]
#Each digit in an mnist dataset is of shape 28,28, but to make it compact, each image is put side by side, so to extract one image we need to reshape it to 28,28.
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image,cmp=matplotlib.cm.binary, interpolation="nearest")
"""

x_train, x_test = x[:6000], x[6000:7000]
y_train, y_test = y[:6000], y[6000:7000]

#Shuffling the training data set.
shuffle_index = np.random.permutation(6000)
x_train, y_train=x_train[shuffle_index], y_train[shuffle_index]

clf =LogisticRegression()

clf.fit(x_train, y_train)

predicted=clf.predict([x[3600]])
print(predicted)

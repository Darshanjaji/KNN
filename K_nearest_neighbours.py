import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
data = np.load("mnist_train_small.npy")
x = data[:, 1:]
y = data[:, 0]

plt.imshow(x[0].reshape(28,28),cmap="gray")
x_train , x_test,y_train,y_test = train_test_split(x,y,random_state=42)
model = KNeighborsClassifier()
model.fit(x_train,y_train)
print(model.predict(x_test[:10]))
print(y_test[:10])
print(model.score(x_train,y_train))
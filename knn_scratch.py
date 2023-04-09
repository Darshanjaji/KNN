import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = np.load("mnist_train_small.npy")
x = data[ : ,1:]
y = data[: , 0]
plt.imshow(x[28].reshape(28,28))
print(y[28])
plt.waitforbuttonpress()
class KnnClassifier:
    def __init__(self,n_neighbours = 5):
        self.n_neighbours = n_neighbours

    def fit(self,x,y):
        self._x = x.astype(np.int64)
        self._y = y
    def predict_point(self,point):
        dist = []
        for x_point,y_point in zip(self._x,self._y):
            distance = ((point - x_point) ** 2).sum()
            dist.append([distance,y_point])
        sorted_dist = sorted(dist)
        top_k = sorted_dist[:self.n_neighbours]
        item,count = np.unique(np.array(top_k)[:,1],return_counts=True)
        ans = item[np.argmax(count)]
        return ans
    def predict(self,x):
        results = []
        for point in x:
            results.append(self.predict_point(point))
        return np.array(results,dtype=int)
    def score(self,x,y):
        return  sum(self.predict(x) == y) / len(y)
model = KnnClassifier()
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42)
model.fit(x_train,y_train)
print(model.predict(x_test[:10]))
print(y_test[:10])
print(model.score(x_test,y_test))


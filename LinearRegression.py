import numpy as np
class LinearRegression:
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = 0

    def fit(self,X,y):
        self.X = X
        self.y = y
        n_sample, n_features = self.X.shape
        self.weight = np.zeros(n_features)

        for _ in range(self.n_iter):
            y_predicted = np.dot(self.X, self.weight) + self.bias

            dw = (1/n_sample) * np.dot(self.X.T, (self.y - y_predicted))
            db = (1/n_sample) * np.sum(self.y - y_predicted)

            self.weight -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self,X):
        y_predicted = np.dot(X, self.weight) + self.bias
        return y_predicted

    def score(self):
        print('your RMSE is ',np.sqrt(np.mean((self.y-np.dot(self.X, self.weight) + self.bias)**2)))
        
# Implementation
'''
from sklearn.model_selection import train_test_split 
from sklearn import datasets
 
X,y = datasets.make_regression(n_samples=1000, n_features=2, n_targets=1, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1)

def rmse(y_true,y_pred):
    return np.sqrt(np.mean((y_true-y_pred)**2))

reg = LinearRegression()
reg.fit(X_train, y_train)
reg.score()
>> your RMSE is  106.08751866705288
'''

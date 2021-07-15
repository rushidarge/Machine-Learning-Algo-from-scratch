import numpy as np
class LogisticRegression:
    def __init__(self,Lr=0.1, n_iter=1000):
        '''
        Parameters: 
            Lr: Learning rate
            n_iter: Total number of iteration

        Return None
        '''
        self.Lr = Lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        '''
        Parameters:
            X training features
            y training labels

        Return None
        '''
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features) 
        self.bias = 0

        for _ in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model) 

            dw = (1 / n_sample) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_sample) * np.sum(y_predicted - y)

            self.weights -= self.Lr * dw
            self.bias -= self.Lr * db

    def predict(self,X):
        '''
        Parameters:
            X dataset that you want to predict the values

        Return Prediction on your dataset
        '''
        linear_model = np.dot(X, self.weights) + self.bias
        y_predict = self._sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predict]
        return y_predicted_class

    def _sigmoid(self,X):
        '''
        Sigmoid function use for squashing
        '''
        return 1 /(1+np.exp(-X))
    
# implementation

'''
def accuracy_score(y_true, y_pred):
    acc = (np.sum(y_true == y_pred)) / len(y_true)
    return acc
    
from sklearn.model_selection import train_test_split 
from sklearn import datasets

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(accuracy_score(y_test,y_pred))
'''

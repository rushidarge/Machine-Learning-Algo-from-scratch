# Importing libraries for mathematical calculations
import numpy as np
from collections import Counter

# to find out distance between two points
def euclidean_dist(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:

    def __init__(self,k=3):
        '''
        Parameters: 
            k=3 Number of nearest neighbour

        Return None
        '''
        # defin k for nearest neighbour 
        self.k = k

    def fit(self,X,y):
        '''
        Parameters:
            X training features
            y training labels

        Return None
        '''
        # save x,y for distance calculation 
        self.X_train = X
        self.y_train = y

    def predict(self,X):
        '''
        Parameters:
            X dataset that you want to predict the values

        Return Prediction on your dataset
        '''
        # return the prediction
        predicted_out = [self._predict(x) for x in X]
        return predicted_out

    def _predict(self,x):
        # Internal function to calculate distance, find nearest neighbours and return label
        dist_x = [euclidean_dist(x,X_tr) for X_tr in self.X_train]
        points = [np.argsort(dist_x)[:self.k]]
        y_pre = [self.y_train[i] for i in points]
        ret_y = Counter(y_pre[0]).most_common(1)
        return ret_y[0][0]

    def score(self):
        '''
        Return
            Accuracy score of your model on train dataset
        '''
        pre = self.predict(self.X_train)
        print(np.sum(self.y_train==pre)/len(self.y_train))
        
#  Implementation Code       
'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
iris = datasets.load_iris()
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


clf = KNN(k=5)
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
clf.score()
'''

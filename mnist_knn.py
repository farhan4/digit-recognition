
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataTrain = pd.read_csv('train.csv')
X_train = dataTrain.iloc[:,1:].values.astype(np.float32)
y_train = dataTrain.iloc[:,0].values.astype(np.float32)

dataTest = pd.read_csv('test.csv')
X_test = dataTest.iloc[:,:].values.astype(np.float32)


#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

knn = cv2.ml.KNearest_create()
knn.train(X_train, cv2.ml.ROW_SAMPLE,y_train)
ret,y_pred,neighbours,dist = knn.findNearest(X_test,k=3)


#from sklearn.metrics import accuracy_score
#accuracy_score(y_test, y_pred)*100

imageId = np.arange(1,28001)
p = y_pred.tolist()
sub = pd.DataFrame({'ImageId': imageId , 'Label': p})

sub.to_csv('submission.csv', index=False)





import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img = cv2.imread('digits.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

X = np.array(cells)
X_train = X[:,:70].reshape(-1,400).astype(np.float32)
X_test  = X[:,70:100].reshape(-1,400).astype(np.float32)

k = np.arange(10)

y_train = np.repeat(k,350)[:,np.newaxis]
y_test  = np.repeat(k,150)[:,np.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(X_train, cv2.ml.ROW_SAMPLE,y_train)
ret,y_pred,neighbours,dist = knn.findNearest(X_test,k=3)

matches = y_pred==y_test
correct = np.count_nonzero(matches)
accuracy = correct*100.0/y_pred.size
print(accuracy)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)*100
















import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataTrain = pd.read_csv('train.csv')
X = dataTrain.iloc[:, 1:].values.reshape(dataTrain.shape[0],1,28, 28).astype( 'float32')

y = dataTrain.iloc[:,0].values.astype(np.float32)

#dataTest = pd.read_csv('test.csv')
#X_test = dataTest.iloc[:,:].values.astype(np.float32)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train = X_train / 255
X_test = X_test / 255

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K

K.set_image_dim_ordering('th')

def build_classifier():
    classifier = Sequential()
    
    classifier.add(Convolution2D(32,5, 5, input_shape = (1, 28, 28), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    classifier.add(Convolution2D(16, 3, 3,activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    classifier.add(Flatten())
    
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 64, activation = 'relu'))
    classifier.add(Dense(units = 10, activation = 'sigmoid'))
    
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = build_classifier()

classifier.fit(X_train, y_train, epochs=14, batch_size=200)

score = classifier.evaluate(X_test, y_test)

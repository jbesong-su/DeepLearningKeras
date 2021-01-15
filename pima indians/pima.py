#first neural net imports 
from numpy import loadtxt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

print('Creating our first neural net')

#load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

#split data into input X and output y
X = dataset[:,:8] #all rows, all columns but last
y = dataset[:,8] # all rows, only last column


#define keras model

model = Sequential()

model.add(Dense(12, input_dim=8, activation='relu')) #12 neurons in this layer, expects 8 input vars

model.add(Dense(8, activation='relu')) #8 neurons in 2nd layer

model.add(Dense(1, activation='sigmoid')) #output layer with 1 neuron. Activation function

#compile keras model

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

#Fit model
model.fit(X, y, epochs=150, batch_size=30, verbose=0)

#evaluate model
a, accuracy = model.evaluate(X,y)

#making probability predictions
predictions = model.predict_classes(X)
#predictions = np.argmax(model.predict(X), axis=-1)

for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
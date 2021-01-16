from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder

#load data set
dt = read_csv('iris.csv', header=None)
data = dt.values

X = data[:, 0:4].astype(float)
Y = data[:, 4]

#encoding output variable Y
#encode class values as integers 
encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)
#convert integers to dummy variables(one hot encoded)
#creates matrix i.e
'''
type1  type2  type3
0        1     0
1        0     0...
'''
dummy_Y = np_utils.to_categorical(encoded_Y)

#define baseline model
#define model as a function so that it can be passed to KerasClassifier as argument
#use KerasClassifier in order to use sci-kit learn with everything
def baseline_model():
    #create model
    model = Sequential()
    #hidden layer 8 nodes. 4 inputs
    model.add(Dense(8, input_dim = 4, activation='relu'))
    #3 outputs because onehot-encoding
    #activation as softmax to ensure output values are in range of 0 and 1
    model.add(Dense(3, activation='softmax'))

    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#Pass arguments wo kerasClassifier to specify how model will be trained

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
print("estimator: ", estimator)

#evaluate model with kfold cross-val
kfold = KFold(n_splits=10,shuffle=True)

results = cross_val_score(estimator, X, dummy_Y, cv=kfold)
print("These are results: ", results)
#print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() *100, results.std() * 100))
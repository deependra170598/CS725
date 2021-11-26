import h5py as h
import numpy as np
# import seaborn as sns

# import matplotlib.cm as cm


f = h.File('inputfile','r') # input file at https://data.mendeley.com/datasets/4r4v785rgx/1 with name "jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5" of size 2gb.


jetTau21 = f['tau_21'] # Jet tau2/tau1 ratio
jetSignal = f['signal'] # Signal (W-initiated or QCD background jet; 1 or 0)
#======================================================================================================

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense#, Dropout,Activation,Flatten, 


# Extract signal labels
Y=np.asarray(jetSignal)
# print(np.shape(Y))
# print(np.ndim(Y))
# print(Y.mean())
# print(Y)

# Extract subjettiness ratio
X=np.asarray(jetTau21)
# print(np.shape(X))
# print(np.ndim(X))
# print(X)
print(X.shape[0])

# constructing traning data and test data
X=X
Y= to_categorical(Y)
# print(Y)


# getting data size and split size
n=0.1 # fraction over which data is to be selected..........................(to be changed every time)
data_size_for_run=int(X.shape[0]*n) # n times of original data size
split=0.7
train_size = int(data_size_for_run*split) # convert percentage to size
test_size = data_size_for_run-train_size

# shuffle indices
np.random.seed(10)
ids = np.random.permutation(X.shape[0]) # shuffle indices
# print(ids)
train_id, test_id = ids[:train_size], ids[train_size:train_size+test_size] # split indices
# print(X[0])
X_train, X_test = X[train_id], X[test_id] # separate by index
Y_train, Y_test = Y[train_id], Y[test_id]
# print(np.shape(X_train))
# print(np.shape(X_train[0]))

def Model():
	model=Sequential()
	model.add(Dense(1,activation='relu'))
	model.add(Dense(2,activation='softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model

model2=Model()
result2=model2.fit(X_train,Y_train,validation_data=(X_test, Y_test), epochs=12)
model2.save('model2_n={s}'.format(s=n)) #............................(to be changed)
model2.summary()
# print(result1.history.keys())
# print(result1.history['accuracy'])

#===========================================================================================
#visualisation result
import matplotlib.pyplot as plt
plt.plot(result2.history['accuracy'])
plt.plot(result2.history['val_accuracy'])
plt.title("accuracy of model with n={s}".format(s=n))
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(['train','test'],loc='upper left')
plt.savefig("subjettiness.png")
plt.show()






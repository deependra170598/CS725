# https://medium.com/@sbutalla2012/machine-learning-for-classifyingw-initiated-and-qcd-background-jets-24a44f570c71
# model applied
import h5py as h5
import numpy as np
# import seaborn as sns

# import matplotlib.cm as cm


f = h5.File('jet-images_Mass60-100_pT250-300_R1.25_Pix25 .hdf5','r')
# print(list(f.keys()))
# dataKeys = [ii for ii in f.keys()] # Get keys for the dataset
# print('Dataset Keys')
# for ii in dataKeys: # Print keys
# 	print(ii)

jetImage = f['image'] # 25 x 25 image
jetSignal = f['signal'] # Signal (W-initiated or QCD background jet; 1 or 0)
# jetEta = f['jet_eta'] # Eta coordinate
# jetPhi = f['jet_phi'] # Phi coordinate
# jetMass = f['jet_mass'] # Jet mass
# jetTau21 = f['tau_21'] # Jet tau2/tau1 ratio
# jetPt =f['jet_pt'] # Jet transverse momentum
# # print(jetImage)
# # print('Number of data points: {}'.format(len(jetPt)))

#===========================================================================================================
# part three

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import  Flatten, Dense#, Dropout,Activation,
# # Model Implementation

# input Image Extraction
X=np.asarray(jetImage)
# print(np.shape(X))
# print(np.ndim(X))
# print(X.shape[0])  #872666

# Extract signal labels
Y=np.asarray(jetSignal)
# print(np.shape(Y))
# print(np.ndim(Y))
# print(Y.mean())
# print(Y)

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
# print(np.shape(X_train)) #(61086,25,25)
# print(np.shape(X_train[0])) #(25, 25)

# reshape for model input
X_train = X_train.reshape(train_size,25,25,1)
X_test = X_test.reshape(test_size,25,25,1)

# print(np.shape(X_train))
print(X_train.shape[0])
# print(X_train[0])
# print(Y_train[0])

def Model():
	model=Sequential()
	model.add(Conv2D(10,kernel_size=2,activation='relu',input_shape=(25,25,1)))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(5,kernel_size=2,activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	#model.add(Dense(10, activation='softmax'))
	model.add(Dense(2, activation='softmax'))
	#model.add(Dense(2, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model

model1=Model()
result1=model1.fit(X_train,Y_train,validation_data=(X_test, Y_test), epochs=12)
model1.save('model1_n={s}'.format(s=n)) #............................(to be changed)
model1.summary()
# print(result1.history.keys())
# print(result1.history['accuracy'])


#===========================================================================================
#visualisation result
import matplotlib.pyplot as plt
plt.plot(result1.history['accuracy'])
plt.plot(result1.history['val_accuracy'])
plt.title("accuracy of model with n={s}".format(s=n))
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(['train','test'],loc='upper left')
plt.savefig("CNN.png")
plt.show()

#===========================================================================================
# # Prediction

# print("total test instances=",X_test.shape[0])
# i=4 #ith test instance to be checked .........................................(to be changed every time)
# print("prediction of {i}th instance".format(i=i))
# img=X_test[i].reshape(1,25,25,1)
# predicted_tag=model1.predict(img)
# actual_tag=Y_test[i]

# print(actual_tag,predicted_tag)
# if actual_tag[0]==1 :
# 	print("Originally W Jet")
# else:
# 	print("Originally QCD jet")

# if predicted_tag[0,0]>0.5:
# 	print("pridiction= W jet")
# else:
# 	print("prediction= QCD jet")

#==============================================================================================



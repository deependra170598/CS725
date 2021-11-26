# https://medium.com/@sbutalla2012/machine-learning-for-classifyingw-initiated-and-qcd-background-jets-24a44f570c71
# model applied
import h5py as h
import numpy as np
# import seaborn as sns

# import matplotlib.cm as cm


f = h.File('jet-images_Mass60-100_pT250-300_R1.25_Pix25 .hdf5','r')


jetImage = f['image'] # 25 x 25 image
jetSignal = f['signal'] # Signal (W-initiated or QCD background jet; 1 or 0)


#===========================================================================================================
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

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

size=X.shape[0]
X=X.reshape(size,25,25,1)
Y= to_categorical(Y)
# print(Y)

model1 = load_model('model_n=0.1')
model1.summary()

print("total test instances=",size)
#i=10 #ith test instance to be checked .........................................(to be changed every time)

for i in range(20):
	print("prediction of {i}th instance".format(i=i))
	img=X[i].reshape(1,25,25,1)
	predicted_tag=model1.predict(img)
	actual_tag=Y[i]

	print(actual_tag,predicted_tag)
	if actual_tag[0]==1 :
		print("Originally W Jet")
	else:
		print("Originally QCD jet")

	if predicted_tag[0,0]>0.5:
		print("pridiction= W jet")
	else:
		print("prediction= QCD jet")



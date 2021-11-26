# https://medium.com/@sbutalla2012/machine-learning-for-classifyingw-initiated-and-qcd-background-jets-24a44f570c71
# Data Exploration
import h5py as h5
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm


f = h5.File('input_file_name','r') #input_filename at https://data.mendeley.com/datasets/4r4v785rgx/1 . Download file with name "jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5" with size 2gb.
# print(list(f.keys()))
# dataKeys = [ii for ii in f.keys()] # Get keys for the dataset
# print('Dataset Keys')
# for ii in dataKeys: # Print keys
# 	print(ii)

jetImage = f['image'] # 25 x 25 image
jetSignal = f['signal'] # Signal (W-initiated or QCD background jet; 1 or 0)
jetEta = f['jet_eta'] # Eta coordinate
jetPhi = f['jet_phi'] # Phi coordinate
jetMass = f['jet_mass'] # Jet mass
jetTau21 = f['tau_21'] # Jet tau2/tau1 ratio
jetPt =f['jet_pt'] # Jet transverse momentum
# print(jetImage)
# print('Number of data points: {}'.format(len(jetPt)))

#=====================================================
# # part one
# # Find indices of W and QCD jets for an example plot
# jetSignal = np.asarray(jetSignal)
# indexW = np.where(jetSignal == 1.0)
# indexQCD = np.where(jetSignal == 0.0)
# # print(np.shape(indexQCD))
# # print(np.ndim(indexQCD))
# exampleW = indexW[0][0]
# exampleQCD = indexQCD[0][0]

# # Plot W-initiated jet and QCD background jet
# fig1, ax = plt.subplots(figsize=(5, 5))
# im = ax.imshow(jetImage[exampleW],interpolation='nearest', extent=[-1.25, 1.25, -1.25, 1.25], cmap=cm.plasma)
# plt.colorbar(im, fraction=0.05, pad=0.005)
# plt.xlabel(r'Pseudorapidity $(\eta)$')
# plt.ylabel(r'Azimuthal Angle $(\phi)$')
# plt.title(r'W Boson Initiated Jet')
# fig1.savefig("W.png")

# fig2, ax = plt.subplots(figsize=(5, 5))
# im = ax.imshow(jetImage[exampleQCD],interpolation='nearest', extent=[-1.25, 1.25, -1.25, 1.25], cmap=cm.plasma)
# plt.colorbar(im, fraction=0.05, pad=0.005)
# plt.xlabel(r'Pseudorapidity $(\eta)$')
# plt.ylabel(r'Azimuthal Angle $(\phi)$')
# plt.title('QCD Background Jet')
# fig2.savefig("QCD.png")

#=====================================================
# #Part two
# sns.set(style="whitegrid")
# print('Histograms of Observables of Interest')


# jetEta=np.asarray(jetEta)
# fig3, ax3 = plt.subplots(figsize=(5, 5))
# ax3 = sns.distplot(jetEta, kde=False)
# plt.xlabel(r'Pseudorapidity $(\eta)$')
# plt.title('Jet Eta Distribution')
# fig3.savefig('jetEtaDist.png')
# #fig3.show()


# jetPhi=np.asarray(jetPhi)
# fig4, ax4 = plt.subplots(figsize=(5, 5))
# ax4 = sns.distplot(jetPhi, kde=False)
# plt.xlabel(r'Azimuthal Angle $(\phi)$ [rad]')
# plt.title('Jet Phi Distribution')
# fig4.savefig('jetPhiDist.png')


# jetMass=np.asarray(jetMass)
# fig5, ax5 = plt.subplots(figsize=(5, 5))
# ax5 = sns.distplot(jetMass, kde=False)
# plt.xlabel(r'Jet Mass (GeV)')
# plt.title('Jet Mass Distribution')
# fig5.savefig('jetMassDist.png')


# jetPt=np.asarray(jetPt)
# fig6, ax6 = plt.subplots(figsize=(5, 5))
# ax6 = sns.distplot(jetPt, kde=False)
# plt.xlabel(r'Jet $p_{T}$ (GeV)')
# plt.title('Jet $p_{T}$ Distribution')
# fig6.savefig('jetPtDist.png')


# jetTau21=np.asarray(jetTau21)
# jetSignal = np.asarray(jetSignal)
# indexW = np.where(jetSignal == 1.0)
# indexQCD = np.where(jetSignal == 0.0)
# jetTau21_W = jetTau21[indexW]
# jetTau21_QCD = jetTau21[indexQCD]

# fig7, ax7 = plt.subplots(figsize=(8, 8))
# ax7 = sns.distplot(jetTau21_W, kde=False, label='W-initiated jets')
# ax7 = sns.distplot(jetTau21_QCD, kde=False, label='QCD background jets')
# plt.xlabel(r'Jet $\tau_{21}$ Distribution')
# plt.title(r'W and QCD Jet $\tau_{21}$')
# ax7.legend()
# fig7.savefig('subjettinessDist2.png')

#====================================================================================
# part three

# from tensorflow.keras.utils import to_categorical
# # Model Implementation

# input Image Extraction
# imgarr=np.asarray(jetImage)
# print(np.shape(imgarr))
# print(np.ndim(imgarr))
#print(imgarr.shape[0])

# # Extract signal labels
# signalarr=np.asarray(jetSignal)
# # print(np.shape(signalarr))
# # print(np.ndim(signalarr))
# # print(signalarr.mean())

# # constructing traning data and test data
# X=imgarr
# Y= to_categorical(signalarr)
# # print(signalarr)
# # print(Y)

# # getting split size
# split=0.7
# train_size = int(X.shape[0]*split) # convert percentage to size
# test_size = X.shape[0]-train_size

# shuffle indices
# ids = np.random.permutation(X.shape[0]) # shuffle indices
# print(ids)
# print(X[0,:])

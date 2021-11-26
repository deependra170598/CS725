# https://medium.com/@sbutalla2012/machine-learning-for-classifyingw-initiated-and-qcd-background-jets-24a44f570c71

import uproot
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#%matplotlib inline

# Create object for the Delphes tree stored in the .root file
events = uproot.open("out_pp_zp_ww.root")["Delphes"]

# Store tower data in awkward array
towerEta=events.get("Tower.Eta")
towerEta=towerEta.array()# Eta coordinate

towerPhi=events.get("Tower.Phi")
towerPhi=towerPhi.array()# Phi coordinate

towerE=events.get("Tower.E")
towerE=towerE.array()# Total energy deposited


calData = np.zeros((1000,300,3), dtype=np.ndarray) # Initialize array
subArraySize = np.zeros(1000) # Array for tracking number of hits per event

print(calData.shape)


for i,iEvent in enumerate(towerEta):
	count=0
	for j,jPart in enumerate(iEvent):
		calData[i][j][0]=towerEta[i][j]
		calData[i][j][1]=towerPhi[i][j]
		calData[i][j][2]=towerE[i][j]
		count+=1
		subArraySize[i]=count


eventNum=13 # choose an event
# Get event data
event = calData[eventNum]
numHits = event.shape[0]
#print(event)

#df = pd.DataFrame(event, columns=['eta', 'phi','E'])
#print(df)

# Put coordinates in np arrays
eta=event[:,0].astype(float)
phi=event[:,1].astype(float)
energy=event[:,2].astype(float)

#print(eta,phi,energy)
# Plot: hexagonal binning with E value color-coding

fig, ax = plt.subplots(figsize=(5, 5))
plt.title(r'Calorimeter Energy Deposition (GeV), Event {}'.format(eventNum))
gridsize=100 # adjust bin sizing
hist = ax.hexbin(eta, phi, C=energy, gridsize=gridsize, cmap=cm.jet, bins=None)
ax.set(xlim=(-4, 4), ylim=(-6.28, 6.28))
plt.xlabel(r'Pseudorapidity $(\eta)$')
plt.ylabel(r'Azimuthal Angle $(\phi)$')
cb = plt.colorbar(hist, fraction=0.05, pad=0.005)
cb.set_label('Energy (GeV)')
plt.show()




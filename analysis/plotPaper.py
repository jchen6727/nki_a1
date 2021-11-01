"""
plotPaper.py

Wavelet and CSD/LFP plotting for oscillation events in A1 model, using NetPyNE 

Contributors: ericaygriffith@gmail.com 
"""
from netpyne import sim
import os
import utils
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import netpyne


### set layer bounds:
layerBounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550, 'L6': 2000}

### Cell populations: 
allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4',
'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B',
'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI']#, 'IC']
L1pops = ['NGF1']
L2pops = ['IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2']
L3pops = ['IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3']
L4pops = ['ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4']
L5Apops = ['IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A']
L5Bpops = ['IT5B', 'PT5B', 'CT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B']
L6pops = ['IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6']
thalPops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI']
ECortPops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B']
ICortPops = ['NGF1', 
			'PV2', 'SOM2', 'VIP2', 'NGF2', 
			'PV3', 'SOM3', 'VIP3', 'NGF3',
			'PV4', 'SOM4', 'VIP4', 'NGF4',
			'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',
			'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',
			'PV6', 'SOM6', 'VIP6', 'NGF6']
EThalPops = ['TC', 'TCM', 'HTC']				# TEpops = ['TC', 'TCM', 'HTC']
IThalPops = ['IRE', 'IREM', 'TI', 'TIM']		# TIpops = ['IRE', 'IREM', 'TI', 'TIM']
reticPops = ['IRE', 'IREM']
matrixPops = ['TCM', 'TIM', 'IREM']
corePops = ['TC', 'HTC', 'TI', 'IRE']


### set path to data files
#based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/'
#based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/lfpCellRecordings/v34_batch27_0_0_LFP_L5_REDO/'
based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/v34_batch27_0_3_IT2_PT5B_SHORT/'

### get .pkl data filenames 
allFiles = os.listdir(based)
allDataFiles = []
for file in allFiles:
	if '.pkl' in file:
		allDataFiles.append(file)

testFiles = ['v34_batch27_0_3_IT2_PT5B_SHORT_data.pkl'] #['A1_v34_batch27_v34_batch27_0_3.pkl']


###### Set timeRange ######

#timeRange = [6050, 6250] 			# 0_0.pkl, beta group (1, 2, 12)
#timeRange = [6050, 6250]			# 0_0.pkl, gamma group (9, 11, 12)

#timeRange = [800, 900]				# 0_1.pkl, gamma

#timeRange = [3425, 3550]			# 0_2.pkl, gamma (chan 16 & 17) 

#timeRange = [180, 500] 			# !! 0_3.pkl, alpha (chan 13, 19), beta (chan 11, 13, 19)
#timeRange = [175, 300]
timeRange = [175, 350] 			## !! 0_3 SHORT RUN 

#timeRange = [1875, 2200] 			# 0_4.pkl (alpha, chan 6 & 12)

#timeRange = [4150, 4400] 			# 1_0.pkl, channels 1 (beta), 11 (beta, gamma), 13 (gamma) 

#timeRange = [5100, 5350] 			# 1_1.pkl, (Beta -- chan 6, 18, 19) (Gamma -- chan 6, 11)

#timeRange = [1350, 2175] 			# 1_2.pkl, chan 11 & 12 -- theta
#timeRange = [3800, 3950] 			# 1_2.pkl, chan 10 & 14 -- theta

#timeRange = [5850, 6050]			# 2_0.pkl

#timeRange = [2000, 2150]			# 2_1.pkl (gamma, chan 2 & 3)

#timeRange = [175, 800] 			# ** !! 2_3.pkl (alpha / theta --> chan 14) (alpha --> chan 19)
#timeRange = [200, 500]				# ** !! 
#[3500, 3600] #[5150, 5300]			# 2_3.pkl
#timeRange = [2200, 2300]

#timeRange = [2197, 2356] 			# 2_4.pkl, beta 228, chan_3
#timeRange = [205, 433] 			# ** !! 2_4.pkl, beta 436, chan_6
#timeRange = [160, 450] 			# ** !! 2_4.pkl, beta 625, chan_9 (& beta chan 5)
#timeRange = [275 , 375]			# ** !!  gamma, chan 4 , 320
#timeRange = [2450, 2550]			# gamma, chan 4, 330
#timeRange = [4130, 4300]

#timeRange = [350, 675]				# 3_0.pkl

###### LOADING SIM DATA and GETTING LFP CONTRIBUTION DATA ####
if len(testFiles) > 0:
	dataFiles = testFiles
else:
	dataFiles = allDataFiles 


for fn in dataFiles:
	fullPath = based + fn
	sim.load(fullPath, instantiate=False)
	#sim.load(dataFile, instantiate=False)

	# Create time lists 
	fullTimeRange = [0, sim.cfg.duration]
	t_full = np.arange(fullTimeRange[0], fullTimeRange[1], sim.cfg.recordStep)
	t_full = list(t_full)  # turn into a list so .index function can be used 

	# NOTE: timeRange is declared earlier
	t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)  ## make an array w/ these time points
	t = list(t)

	# Find the indices of the timeRange within the full range, to correspond to the desired segment of LFP data
	beginIndex = t_full.index(timeRange[0])
	endIndex = t_full.index(timeRange[-1])


	allData = sim.allSimData
	#LFPData = allData['LFP']
	LFPPops = allData['LFPPops'].keys()

	for pop in LFPPops:
		electrodeShortData1 = allData['LFPPops'][pop][beginIndex:endIndex,1]


	for pop in LFPPops:
		plt.plot(t_full, allData['LFPPops'][pop][:, 1], label=pop)

	plt.legend()
	plt.show()

	# # for saving from multiple pkl files:
	# cellLFPData = {}

	# include = sim.cfg.saveLFPCells

	# if include is not False:
	# 	cellsIncluded, cellGids, _ = netpyne.analysis.utils.getCellsInclude(include)

	# 	cellIDs = {}

	# 	for i in range(len(cellsIncluded)):
	# 		cellGID = cellsIncluded[i]['gid']
	# 		cellPop = cellsIncluded[i]['tags']['pop']
	# 		cellIDs[cellGID] = cellPop

	# 	LFPCells = sim.allSimData['LFPCells']

	# 	cells = list(LFPCells.keys()) ## This is a list of cell GIDs
	# 	## ^^ to get the name / pop of these cells --> cellIDs[cells[i]]

	# 	for cell in cells:
	# 		cellLFPData[cellIDs[cell]] = {}
	# 		numElectrodes = LFPCells[cell].shape[1]
	# 		for elec in range(numElectrodes):
	# 			cellLFPData[cellIDs[cell]][elec] = {}

	# 			fullLFPTrace = list(LFPCells[cell][:,elec])
	# 			cellLFPData[cellIDs[cell]][elec]['fullLFP'] = fullLFPTrace

	# 			LFPTrace = fullLFPTrace[beginIndex:endIndex]	# This is the segmented LFP trace, by time point
	# 			cellLFPData[cellIDs[cell]][elec]['timeRangeLFP'] = LFPTrace


########################
####### PLOTTING #######
########################

### LFP, CSD, TRACES ### ## CHANGE THESE TO ARGUMENTS ## 
LFP = 0
LFPcellContrib = 0
CSD = 1
traces = 0
waveletNum = 1
electrodes = ['all']#[4,5,6,7]  	# CHANGE THIS TO DESIRED ELECTRODES 
waveletImg = 0


### INDIVIDUAL LFP CONTRIBUTION ###  
# cells = []  		# CHANGE THIS TO DESIRED CELLS 		--> ## ANOTHER GOOD CANDIDATE FOR AN ARG? 

if LFPcellContrib:
	cells = list(cellLFPData.keys())
	for cell in cells:
		for elec in electrodes:
			ax1 = fig.add_subplot(gs[0,2])
			ax1.plot(t, cellLFPData[cell][elec]['timeRangeLFP'], label = cell)
			#plt.plot(t, cellLFPData[cell][elec]['timeRangeLFP'], label = cell)
			#plt.legend()
			#plt.title('Individual cell contrib to LFP, electrode ' + str(elec))
			#plt.show()


### Doesn't matter which file was last to load for sim in this case --> should all be the same except for subsets of LFP cell contrib saved 
if LFP:
	sim.analysis.plotLFP(plots=['PSD', 'timeSeries', 'spectrogram'],electrodes=electrodes,showFig=True, timeRange=timeRange), #figSize=(5,5)) # electrodes=[2,6,11,13] # saveFig=figname, saveFig=True, plots=['PSD', 'spectrogram']
if CSD:
	sim.analysis.plotCSD(spacing_um=100, timeRange=timeRange, overlay='LFP', hlines=0, layerLines=1, layerBounds = layerBounds,saveFig=0, figSize=(5,5), showFig=1) # LFP_overlay=True
if traces:
	sim.analysis.plotTraces(include=[(pop, 0) for pop in allpops], timeRange = timeRange, oneFigPer='trace', overlay=True, saveFig=False, showFig=True)#, figSize=(6,8))#figSize=(12,8))


#### VERIFY saveLFPPops worked 


# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------

#if __name__ == '__main__':







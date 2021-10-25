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
based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/'
#based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/lfpCellRecordings/v34_batch27_0_0_LFP_L5_REDO/'


### get .pkl data filenames 
allFiles = os.listdir(based)
allDataFiles = []
for file in allFiles:
	if '.pkl' in file:
		allDataFiles.append(file)

testFiles = ['A1_v34_batch27_v34_batch27_1_4.pkl']	
#testFiles = ['v34_batch27_0_0_LFP_L5_REDO_data.pkl']	#['v32_batch28_data.pkl'] #['A1_v34_batch27_v34_batch27_2_4.pkl'] # ['A1_v32_batch20_v32_batch20_0_0.pkl'] 


###### WAVELETS ######

## Add in a line that will ... maybe extract the subdir, but for now hard-code it
waveletDir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/wavelets/A1_v34_batch27_v34_batch27_1_4/chan_4/' 
#waveletDir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/wavelets/A1_v34_batch27_v34_batch27_0_0/chan_5/'
## Add in line(s) that will line the chan_# up with the electrodes list

waveletFile = waveletDir + 'waveletInfo.txt'


with open(waveletFile) as f:
	lines = f.readlines() 

## Add in line(s) that will account for index # of the oscillation event!!

# absminT = {}
# absmaxT = {}


#### TIMING INFO --> good candidate for function 
for line in lines:
	if 'absminT' in line:
		absminT = float(line[12:-1])
	if 'absmaxT' in line:
		absmaxT = float(line[12:-1])

print('absminT = ' + str(absminT))
print('absmaxT = ' + str(absmaxT))


###### Set timeRange ######
#timeRange = [1100, 1500]		# CHANGE THIS TO DESIRED TIME RANGE ## GOOD CANDIDATE FOR AN ARGUMENT (when turning it into a function)
timeRange = [4446, 4569]
#timeRange = [round(absminT), round(absmaxT)]



# return timeRange 
# have timeRange as an arg? --> then if timeRange is None, use this. 




###### LOADING SIM DATA and GETTING LFP CONTRIBUTION DATA ####
if len(testFiles) > 0:
	dataFiles = testFiles
else:
	dataFiles = allDataFiles 


for fn in dataFiles:
	fullPath = based + fn
	sim.load(fullPath, instantiate=False)

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

	# for saving from multiple pkl files:
	cellLFPData = {}

	include = sim.cfg.saveLFPCells

	if include is not False:
		cellsIncluded, cellGids, _ = netpyne.analysis.utils.getCellsInclude(include)

		cellIDs = {}

		for i in range(len(cellsIncluded)):
			cellGID = cellsIncluded[i]['gid']
			cellPop = cellsIncluded[i]['tags']['pop']
			cellIDs[cellGID] = cellPop

		LFPCells = sim.allSimData['LFPCells']

		cells = list(LFPCells.keys()) ## This is a list of cell GIDs
		## ^^ to get the name / pop of these cells --> cellIDs[cells[i]]

		for cell in cells:
			cellLFPData[cellIDs[cell]] = {}
			numElectrodes = LFPCells[cell].shape[1]
			for elec in range(numElectrodes):
				cellLFPData[cellIDs[cell]][elec] = {}

				fullLFPTrace = list(LFPCells[cell][:,elec])
				cellLFPData[cellIDs[cell]][elec]['fullLFP'] = fullLFPTrace

				LFPTrace = fullLFPTrace[beginIndex:endIndex]	# This is the segmented LFP trace, by time point
				cellLFPData[cellIDs[cell]][elec]['timeRangeLFP'] = LFPTrace


########################
####### PLOTTING #######
########################

### LFP, CSD, TRACES ### ## CHANGE THESE TO ARGUMENTS ## 
LFP = 0
LFPcellContrib = 0
CSD = 1
traces = 0
waveletNum = 1
electrodes = [5]  	# CHANGE THIS TO DESIRED ELECTRODES 

### Set up figure ###
fig = plt.figure(figsize=(10,6))
gs = GridSpec(nrows = 2, ncols = LFP + LFPcellContrib + CSD + traces + len(electrodes) + waveletNum)


### Plot Wavelet ###
imgName = 'A1_v34_batch27_v34_batch27_1_4_SIM_gcp_wavelet_chan_4_beta_268' 
#imgName = 'A1_v34_batch27_v34_batch27_0_0_SIM_gcp_wavelet_chan_5_beta_368' 
waveletImgFile = waveletDir + imgName + '.png'

## Add in line for reading in image
img = mpimg.imread(waveletImgFile)
#imgplot = plt.imshow(img)
ax0 = fig.add_subplot(gs[0,0])
ax0.imshow(img)


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
	sim.analysis.plotLFP(plots=['timeSeries'],electrodes=electrodes,showFig=True, timeRange=timeRange) # electrodes=[2,6,11,13] # saveFig=figname, saveFig=True, plots=['PSD', 'spectrogram']
if CSD:
	sim.analysis.plotCSD(spacing_um=100, timeRange=timeRange, overlay=None, hlines=1, layerLines=1, layerBounds = layerBounds,saveFig=0, showFig=1) # LFP_overlay=True
if traces:
	sim.analysis.plotTraces(include=[(pop, 0) for pop in L5Bpops], timeRange = timeRange, oneFigPer='trace', overlay=False, saveFig=False, showFig=True, figSize=(12,8))



# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':







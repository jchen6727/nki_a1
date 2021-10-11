from netpyne import sim
import os
import utils
from matplotlib import pyplot as plt
import numpy as np


### set layer bounds:
layer_bounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550, 'L6': 2000}

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
based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/lfpCellRecordings/v34_batch27_0_0_LFP_L5_REDO/'
#based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/'
#based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/v32_batch28/'

### set path to .csv layer file 
#dbpath = based + 'simDataLayers.csv'

### get .pkl data filenames 
allFiles = os.listdir(based)
allDataFiles = []
for file in allFiles:
	if '.pkl' in file:
		allDataFiles.append(file)

testFiles = ['v34_batch27_0_0_LFP_L5_REDO_data.pkl']	#['v32_batch28_data.pkl'] #['A1_v34_batch27_v34_batch27_2_4.pkl'] # ['A1_v32_batch20_v32_batch20_0_0.pkl'] 


### PLOTTING 
LFP = 0
CSD = 0
traces = 0


if len(testFiles) > 0:
	dataFiles = testFiles
else:
	dataFiles = allDataFiles 

for fn in dataFiles:
	fullPath = based + fn
	sim.load(fullPath, instantiate=False)
	if LFP == 1:
		sim.analysis.plotLFP(plots=['spectrogram'],electrodes=[2,6,11,13],showFig=True)# timeRange=[1300,2300] # saveFig=figname, saveFig=True, plots=['PSD', 'spectrogram']
	if CSD == 1:
		sim.analysis.plotCSD(spacing_um=100, timeRange=[1000,1200], overlay=True, layer_lines=True, saveFig=0, showFig=1) # LFP_overlay=True
	if traces == 1:
		sim.analysis.plotTraces(include=[(pop, 0) for pop in allpops], oneFigPer='trace', overlay=False, saveFig=False, showFig=True, figSize=(12,8))


###########################################
## full LFP
#lfp = np.array(sim.allSimData['LFP'])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]


###########################################
## Individual LFP cell contributions 
LFPCells = sim.allSimData['LFPCells']
cells = list(LFPCells.keys()) 

## Time 
timeRange = [0, sim.cfg.duration]
t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)

## Plot 
for cell in cells[0:2]:
	elec = 4  	# arbitrary -- which electrode do you want to plot?
	LFPtrace = LFPCells[cell][:,elec]
	LFPtrace = list(LFPtrace) ## necessary?
	plt.plot(t,LFPtrace)

plt.show()




# ## Get LFP cell contributions:
# allSimData = sim.allSimData
# LFPCells = allSimData['LFPCells']
# cells = list(LFPCells.keys()) 

# for cell in cells:
# 	print('cell --> ' + str(cell))


# ###########################################
# # Loading individual LFP traces
# LFPCellDict = sim.allSimData['LFPCells']
# print('Dict keys for LFPCells: ' + str(LFPCellDict.keys()))

# LFPCells = LFPCellDict.keys()
# for cell in LFPCells:
# 	elec = 4  	# arbitrary -- which electrode do you want to plot?
# 	LFPtrace = LFPCellDict[cell][:,elec]
# 	LFPtrace = list(LFPtrace) ## necessary?
# 	plt.plot(t,LFPtrace)
# 	plt.show()


## ADD LINES FOR:
### (1) Calling load.py possibly, or having the timeRange entered manually?
### (2) Look at spiking activity in the timeRange specified for each oscillation in question
### (3) Look at LFP in the timeRange specific for each oscillation in question 


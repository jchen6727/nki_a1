from netpyne import sim
import os
import utils
from matplotlib import pyplot as plt
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
based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/lfpCellRecordings/v34_batch27_0_0_LFP_L5_REDO/'
#based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/'
#based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/v32_batch28/'


### get .pkl data filenames 
allFiles = os.listdir(based)
allDataFiles = []
for file in allFiles:
	if '.pkl' in file:
		allDataFiles.append(file)

testFiles = ['v34_batch27_0_0_LFP_L5_REDO_data.pkl']	#['v32_batch28_data.pkl'] #['A1_v34_batch27_v34_batch27_2_4.pkl'] # ['A1_v32_batch20_v32_batch20_0_0.pkl'] 


###### Set timeRange ######
timeRange = [1100, 1500]		# CHANGE THIS TO DESIRED TIME RANGE ## GOOD CANDIDATE FOR AN ARGUMENT (when turning it into a function)



###### PLOTTING LFP, CSD, TRACES #######
LFP = 0
CSD = 0
traces = 0
#multiplePkls = 0
electrodes = [5]  			## Change this to desired electrodes!!! 


if len(testFiles) > 0:
	dataFiles = testFiles
else:
	dataFiles = allDataFiles 


for fn in dataFiles:
	fullPath = based + fn
	sim.load(fullPath, instantiate=False)


	if LFP:
		sim.analysis.plotLFP(plots=['timeSeries'],electrodes=[2,6,11,13],showFig=True, timeRange=timeRange) # saveFig=figname, saveFig=True, plots=['PSD', 'spectrogram']
	if CSD:
		sim.analysis.plotCSD(spacing_um=100, timeRange=timeRange, overlay='CSD', layerLines=0, layerBounds = layerBounds,saveFig=0, showFig=1) # LFP_overlay=True
	if traces:
		sim.analysis.plotTraces(include=[(pop, 0) for pop in L5Apops], timeRange = timeRange, oneFigPer='trace', overlay=False, saveFig=False, showFig=True, figSize=(12,8))


	## Create time lists 
	fullTimeRange = [0, sim.cfg.duration]
	t_full = np.arange(fullTimeRange[0], fullTimeRange[1], sim.cfg.recordStep)
	t_full = list(t_full)  # turn into a list so .index function can be used 

	## NOTE: timeRange is declared earlier
	t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)  ## make an array w/ these time points
	t = list(t)

	## Find the indices of the timeRange within the full range, to correspond to the desired segment of LFP data
	beginIndex = t_full.index(timeRange[0])
	endIndex = t_full.index(timeRange[-1])


	# for saving from multiple pkl files:
	cellDataByFile = {}
	cellDataByFile[fn] = {} 

	#cellDataByFile[fn]['include'] = sim.cfg.saveLFPCells
	include = sim.cfg.saveLFPCells

	#cellsIncluded, cellGids, _ = netpyne.analysis.utils.getCellsInclude(cellDataByFile[fn]['include'])
	cellsIncluded, cellGids, _ = netpyne.analysis.utils.getCellsInclude(include)


	#cellDataByFile[fn]['cellIDs'] = {} 			 
	cellIDs = {}

	for i in range(len(cellsIncluded)):
		cellGID = cellsIncluded[i]['gid']
		cellPop = cellsIncluded[i]['tags']['pop']
		#cellDataByFile[fn]['cellIDs'][cellGID] = cellPop    
		cellIDs[cellGID] = cellPop

	#cellDataByFile[fn]['LFPCells'] = sim.allSimData['LFPCells']
	LFPCells = sim.allSimData['LFPCells']

	#cells = list(cellDataByFile[fn]['LFPCells'].keys())  # List of cell GIDs
	cells = list(LFPCells.keys())

	for cell in cells:
		cellDataByFile[fn][cellIDs[cell]] = {}
		for elec in electrodes:
			electrodeKey = 'elec_' + str(elec)
			cellDataByFile[fn][cellIDs[cell]][electrodeKey] = {}

			fullLFPTrace = list(LFPCells[cell][:,elec])			#cellDataByFile[fn]['LFPCells'][cell][:,elec]
			cellDataByFile[fn][cellIDs[cell]][electrodeKey]['fullLFP'] = fullLFPTrace

			LFPTrace = fullLFPTrace[beginIndex:endIndex]	# This is the segmented LFP trace, by time point
			cellDataByFile[fn][cellIDs[cell]][electrodeKey]['timeRangeLFP'] = LFPTrace






# ###### cell GID <--> cell type correspondence ######
# include = sim.cfg.saveLFPCells # + sim.cfg.analysis['plotTraces']['include']
# cellsIncluded, cellGids, _ = netpyne.analysis.utils.getCellsInclude(include)

# cellIDs = {}

# for i in range(len(cellsIncluded)):
# 	cellGID = cellsIncluded[i]['gid']
# 	cellPop = cellsIncluded[i]['tags']['pop']
# 	cellIDs[cellGID] = cellPop



# ###### Individual cell LFP contributions ######
# LFPCells = sim.allSimData['LFPCells']
# cells = list(LFPCells.keys()) ## This is a list of cell GIDs
# ## ^^ to get the name / pop of these cells --> cellIDs[cells[i]]



# ## Plot individual cell LFPs
# for cell in cells:
# 	elec = 5  							# arbitrary -- which electrode do you want to plot?
# 	LFPtrace = LFPCells[cell][:,elec] 	# This is the whole trace, unsegmented 
# 	LFPtrace = list(LFPtrace)  
# 	LFPtrace = LFPtrace[beginIndex:endIndex]  	 # This is the segmented LFP trace, by time point
# 	plt.plot(t,LFPtrace, label=cellIDs[cell])
# 	plt.legend()
# 	# Create option to not overlay these traces!! 

# plt.title('Individual cell contributions to LFP')
# plt.show()





## ADD LINES FOR:
### (1) timeRange --> manually or call from load.py?
### (2) If taking timeRange from load.py --> have this correspond to timeRange variable as it functions now



from netpyne import sim
import os

# set path to data files
based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/speech/'

# set path to .csv layer file 
dbpath = based + 'simDataLayers.csv'

# set layer bounds:
layer_bounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550, 'L6': 2000}


# get .mat data filenames 
allFiles = os.listdir(based)
pklfn = []
for file in allFiles:
	if '.pkl' in file:
		pklfn.append(file)



for fn in pklfn:
	fullPath = based + fn
	sim.load(fullPath, instantiate=False) 	# instantiate=False gets rid of hoc error 
	# lfp_data = np.array(sim.allSimData['LFP'])
	# dt = sim.cfg.recordStep/1000.0 # this is in ms by default -- convert to seconds
	# sampr = 1./dt 	# sampling rate (Hz)
	# spacing_um = sim.cfg.recordLFP[1][1] - sim.cfg.recordLFP[0][1]
	# CSD_data = sim.analysis.getCSD()

	[lfp_data, CSD_data, sampr, spacing_um, dt] = sim.analysis.getCSD(getAllData=True)

	# get onset of stim and other info about speech stim
	if 'speech' in based:
		thalInput = sim.cfg.ICThalInput
		stimFile = thalInput['file']#.split('/')[-1]
		stimStart = thalInput['startTime']


	#fignameCSD = 'CSD_fullTimeRange_%s' % fn[:-4] 		#fn.split("/")[-1][:-4]
	#sim.analysis.plotCSD(**{'spacing_um': 100, 'overlay': None, 'layer_lines': 1, 'layer_bounds': layer_bounds, 'timeRange': None, 'saveFig': fignameCSD, 'showFig': True})

	# sampr is incorrect!!! 
	fignameAvgCSD = 'AvgCSD_%s' % fn[:-4]
	### AVG CSD FUNC IN NETPYNE DOES NOT EXIST YET!! sim.analysis.
	startTime = stimStart #-50
	endTime = stimStart + 50	#sim.cfg.duration # CAN CHANGE AS NEEDED
	sim.analysis.plotCSD(**{'spacing_um': 100, 'overlay': None, 'layer_lines': 1, 'layer_bounds': layer_bounds, 'timeRange': [startTime,endTime], 'stim_start_time':stimStart, 'saveFig': fignameAvgCSD, 'showFig': True})














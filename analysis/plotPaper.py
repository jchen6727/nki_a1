from netpyne import sim
import os
import utils
from matplotlib import pyplot as plt


### set layer bounds:
layer_bounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550, 'L6': 2000}

### all pops: 
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
matrixPops = ['TCM', 'IREM']
corePops = ['TC', 'HTC', 'IRE']


### set path to data files
based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/'

### set path to .csv layer file 
dbpath = based + 'simDataLayers.csv'

### get .pkl data filenames 
allFiles = os.listdir(based)
allDataFiles = []
for file in allFiles:
	if '.pkl' in file:
		allDataFiles.append(file)

testFiles = ['A1_v34_batch27_v34_batch27_2_4.pkl'] # ['A1_v32_batch20_v32_batch20_0_0.pkl'] 


### PLOTTING 
LFP = 0
CSD = 1
traces = 0


if len(testFiles) > 0:
	fullPath  = based + testFiles[0]
	sim.load(fullPath, instantiate=False)
	if LFP == 1:
		sim.analysis.plotLFP(plots=['spectrogram'],electrodes=[2,6,11,13],timeRange=[1300,2300], showFig=True)# timeRange=[1300,2300], # saveFig=figname, #,saveFig=True)#, 'PSD', 'spectrogram'])
	if CSD == 1:
		sim.analysis.plotCSD(spacing_um=100, timeRange=[1000,1200], saveFig=False, showFig=True, layer_lines=True, layer_bounds=layer_bounds, overlay='CSD_bandpassed') # LFP_overlay=True, # layer_lines=True, 
	if traces == 1:
		sim.analysis.plotTraces(include=[(pop, 0) for pop in thalPops], oneFigPer='trace', overlay=False, saveFig=False, showFig=True, figSize=(8,8))



else:
	for fn in allDataFiles:
		fullPath = based + fn
		sim.load(fullPath, instantiate=False)
		if LFP == 1:
			sim.analysis.plotLFP(plots=['spectrogram'],electrodes=[2,6,11,13],showFig=True)# timeRange=[1300,2300], # saveFig=figname, #,saveFig=True)#, 'PSD', 'spectrogram'])
		if CSD == 1:
			sim.analysis.plotCSD(spacing_um=100, timeRange=[1000,1200], LFP_overlay=True, layer_lines=True, saveFig=0, showFig=1)
		if traces == 1:
			sim.analysis.plotTraces(include=[(pop, 0) for pop in allpops], oneFigPer='trace', overlay=False, saveFig=False, showFig=True, figSize=(12,8))



# ### TRACES -- 
# #cfg.analysis['plotTraces'] = {'include': [(pop, 0) for pop in allpops], 'oneFigPer': 'trace', 'overlay': False, 'saveFig': True, 'showFig': False, 'figSize':(12,8)} #[(pop,0) for pop in alltypes]		## Seen in M1 cfg.py (line 68) 
# sim.analysis.plotTraces('include': [(pop, 0) for pop in allpops], 'oneFigPer': 'trace', 'overlay': False, 'saveFig': True, 'showFig': False, 'figSize':(12,8))

# ### CSD -- 
# fullPath = based + fn
# #timeRange=[1100, 1400]
# #sim = utils.loadFromFile('../data/v25_batch5/v25_batch5_0_0_2.json')
# sim.load(fullPath, instantiate=False)
# sim.analysis.plotCSD(spacing_um=100, timeRange=timeRange, LFP_overlay=True, layer_lines=True, saveFig=1, showFig=0)
# #plt.savefig('v25_batch5_0_0_2_1100-1400.png')   


# ### LFP -- 
# for fn in testpkl: #pklfn: 
# 	fullPath = based + fn
# 	sim.load(fullPath, instantiate=False) 	# instantiate=False gets rid of hoc error 
# 	# lfp_data = np.array(sim.allSimData['LFP'])
# 	# dt = sim.cfg.recordStep/1000.0 # this is in ms by default -- convert to seconds
# 	# sampr = 1./dt 	# sampling rate (Hz)
# 	# spacing_um = sim.cfg.recordLFP[1][1] - sim.cfg.recordLFP[0][1]
# 	# CSD_data = sim.analysis.getCSD()

# 	#[lfp_data, CSD_data, sampr, spacing_um, dt] = sim.analysis.getCSD(getAllData=True)
# 	#figname = '/Users/ericagriffith/Desktop/NEUROSIM/A1/analysis/' + fn + '_timeSeries.png'
# 	figname = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/' + fn + '_spect.png'
# 	#figname = '/Users/ericagriffith/Desktop/NEUROSIM/A1/analysis/' + fn + '_PSD.png'

# 	sim.analysis.plotLFP(plots=['spectrogram'],electrodes=[2,6,11,13],timeRange=[1300,2300],saveFig=figname,showFig=True)#,saveFig=True)#, 'PSD', 'spectrogram'])

# 	# # get onset of stim and other info about speech stim
# 	# if 'speech' in based:
# 	# 	thalInput = sim.cfg.ICThalInput
# 	# 	stimFile = thalInput['file']#.split('/')[-1]
# 	# 	stimStart = thalInput['startTime']
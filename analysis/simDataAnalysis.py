"""
plotPaper.py

Wavelet and CSD/LFP plotting for oscillation events in A1 model, using NetPyNE 

Contributors: ericaygriffith@gmail.com 
"""

### IMPORTS ### 
from netpyne import sim
import os
import utils
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import netpyne
from numbers import Number



### FUNCTIONS ####

def getDataFiles(based):
	### based: str ("base directory")
	### returns list of .pkl files --> allDataFiles

	allFiles = os.listdir(based)
	allDataFiles = []
	for file in allFiles:
		if '.pkl' in file:
			allDataFiles.append(file)
	return allDataFiles 


#### THIS FUNCTION STILL NOT WORKING THE BEST FOR TIME RANGES WITH LOW VALUES AT timeRange[0]!!! FIGURE OUT WHY THIS IS! 
def plotMUA(dataFile, colorList, timeRange=None, pops=None):
	### DEPRECATED -- based: str ("base directory"); path to data files
	### DEPRECATED -- dataFiles: list of data files 
	### dataFile: str --> path and filename (complete, e.g. '/Users/ericagriffith/Desktop/.../0_0.pkl')
	### timeRange: list --> e.g. [start, stop]
	### pops: list --> e.g. ['IT2', 'NGF3']
			# ``None`` is default --> plots spiking data from all populations


	## Load sim data from .pkl file 
	sim.load(dataFile, instantiate=False)

	## timeRange
	if timeRange is None:
		timeRange = [0,sim.cfg.duration]

	## Get spiking data (timing and cell ID)
	spkt = sim.allSimData['spkt'] 
	spkid = sim.allSimData['spkid']

	spktTimeRange = []
	spkidTimeRange = []

	for t in spkt:
		if t >= timeRange[0] and t <= timeRange[1]:
			spktTimeRange.append(t)
			spkidTimeRange.append(spkid[spkt.index(t)])

	spikeGids = {}
	spikes = {}

	## Get list of populations to plot spiking data for 
	if pops is None:
		pops = list(sim.net.allPops.keys())			# all populations! 

	## Separate overall spiking data into spiking data for each cell population 
	for pop in pops:
		spikeGids[pop] = sim.net.allPops[pop]['cellGids']
		spikes[pop] = []
		for i in range(len(spkidTimeRange)):
			if spkidTimeRange[i] in spikeGids[pop]:
				spikes[pop].append(spktTimeRange[i])
		print('spikes in pop ' + str(pop) + ': ' + str(len(spikes[pop])))


	for pop in pops:
		print('Plotting spiking data for ' + pop)
		color=colorList[pops.index(pop)%len(colorList)] 
		val = pops.index(pop) 
		spikeActivity = np.array(spikes[pop])
		plt.plot(spikeActivity, np.zeros_like(spikeActivity) + (val*2), '|', label=pop, color=color)
		plt.text(timeRange[0]-20, val*2, pop, color=color)


	## PLOT ALL POPULATIONS ON THE SAME PLOTTING WINDOW ! 
	ax = plt.gca()
	ax.invert_yaxis()
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.get_yaxis().set_visible(False)


def plotLFPPopsData(dataFile, plots, electrodes=['avg'], timeRange=None, pops=None):
	### dataFile: str --> path and filename (complete, e.g. '/Users/ericagriffith/Desktop/.../0_0.pkl')
	### plots: list --> e.g. ['spectrogram', 'timeSeries', 'PSD']
	### electrodes: list ---> e.g. ['avg'] <-- can this be combined / used with numbered electrodes??
	### timeRange: list --> e.g. [start, stop]
	### pops: list of populations to plot data for 

	## Load sim data
	sim.load(dataFile, instantiate=False)

	## Get cell populations:
	if pops is None:
		pops = list(sim.net.allPops.keys())			# all populations! 

	## timeRange
	if timeRange is None:
		timeRange = [0, sim.cfg.duration]

	## PLOTTING:
	for pop in pops:
		for plot in plots:
			sim.analysis.plotLFP(pop=pop,timeRange=timeRange, plots=[plot], electrodes=electrodes) ### fix / clean up 'avg' situation!


### USEFUL VARIABLES ### 
## set layer bounds:
layerBounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550, 'L6': 2000}

## Cell populations: 
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
ECortPops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6']
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


## electrodes <--> layer
L1electrodes = [0]
L2electrodes = [1]
L3electrodes = [2,3,4,5,6,7,8,9]
L4electrodes = [9,10,11,12]
L5Aelectrodes = [12,13]
L5Belectrodes = [13,14,15]
L6electrodes = [15,16,18,19]
allElectrodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
supra = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
gran = [9, 10, 11, 12]
infra = [12, 13, 14, 15, 16, 17, 18, 19]

## for COLORS in LFP PLOTTING !! 
colorList = [[0.42,0.67,0.84], [0.90,0.76,0.00], [0.42,0.83,0.59], [0.90,0.32,0.00],
            [0.34,0.67,0.67], [0.90,0.59,0.00], [0.42,0.82,0.83], [1.00,0.85,0.00],
            [0.33,0.67,0.47], [1.00,0.38,0.60], [0.57,0.67,0.33], [0.5,0.2,0.0],
            [0.71,0.82,0.41], [0.0,0.2,0.5], [0.70,0.32,0.10]]*3


#####################
#### DATA FILES #####
#####################

######### SET LOCAL BOOL	!!
local = 1								# if using local machine (1) 	# if using neurosim or other (0)
## Path to data files
if local:
	based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/'  # '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/miscRuns/shortRuns/'


######### SET TEST BOOL		!!
test = 1								# if using testFiles (1) 		# if using all data files (0)
## Data files to use 
testFiles = ['A1_v34_batch65_v34_batch65_0_0_data.pkl']

if test:
	dataFiles = testFiles
else:
	dataFiles = getDataFiles(based)		# get all .pkl data filenames


######### SET TIME RANGE 	!!
timeRange = [300,500] #[175, 350]					# AT SOME POINT MAKE THIS A FUNCTION THAT EXTRACTS THIS FROM THE WAVELET??




########################
####### PLOTTING #######
########################

### WHAT TO PLOT --> LFP, CSD, SPIKING, TRACES ### 
MUA = 0			## bool (0 or 1) 
MUApops = 0		## bool OR list of pops --> e.g. ['ITP4', 'ITS4'] # ECortPops.copy()


LFP = 1				## bool (0 or 1)
LFPpops = ['IT2', 'IT3']	#0  	## bool OR list of pops --> e.g. ['IT2']


LFPPopContrib = 0 #['IT2']	# 0 #['ITP4', 'ITS4']	#ECortPops.copy() #['ITP4', 'ITS4']	#1			## Can be '1', '0', OR list of pops! 
filtFreq = 0 #[13,30]


CSD = 0
traces = 0
#waveletNum = 0 #1
electrodes = ['avg']	#[3, 4, 5, 6]	#'avg' #		# list of electrodes, or 'all', or 'avg' <-- NOTE: make sure 'all' and 'avg' both work as expected!!! 
#waveletImg = 0


## SPIKING MUA 
if MUA: 
	for dataFile in dataFiles:
		dataFileFull = based + dataFile

		if MUApops:
			plotMUA(dataFileFull, colorList, timeRange, pops=MUApops)
		else:
			plotMUA(dataFileFull, colorList, timeRange, pops=None)


## LFP DATA

if LFP:
	plots = ['spectrogram', 'timeSeries']
	for dataFile in dataFiles:
		dataFileFull = based + dataFile
		plotLFPPopsData(dataFileFull, plots, electrodes=['avg'], timeRange=timeRange, pops=LFPpops)


	#sim.load(dataFileFull, instantiate=False)
	# if LFPpops:
	# 	if type(LFPpops) is list:
	# 		pops = LFPpops
	# 	else:
	# 		pops = list(sim.net.allPops.keys())

	# for pop in pops:
	# 	sim.analysis.plotLFP(pop=pop,timeRange=timeRange, plots=['spectrogram'], electrodes=['avg'])

###

	# for dataFile in dataFiles:
	# 	dataFileFull = based + dataFile

	# 	sim.load(dataFileFull, instantiate=False)


	# 	if LFPpops:
	# 		if type(LFPpops) is list:
	# 			pops = LFPpops
	# 		else:
	# 			pops = list(sim.net.allPops.keys())

	# 	for pop in pops:
	# 		sim.analysis.plotLFP(pop=pop,timeRange=timeRange, plots=['spectrogram'], electrodes=['avg'])



#### Plot LFP Pops contribution: 
### NOTE: right now in this code block I specify LFPPops for ease of example generating 
### NOTE: I also specify fileName for fig 
if LFPPopContrib:  ## <-- this basically plots the timeSeries for individual LFP pops if this data is recorded 
	allLFPData = {}
	allLFPData['LFPPops'] = {}

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


		## LFP contributions knitted together:
		singleSimData = sim.allSimData
		LFPPops = list(singleSimData['LFPPops'].keys())
		for pop in LFPPops:
			allLFPData['LFPPops'][pop] = singleSimData['LFPPops'][pop]



	plt.figure(figsize=(6,8))

	if type(LFPPopContrib) == list:
		LFPPops = LFPPopContrib.copy()
	else:
		LFPPops = list(allLFPData['LFPPops'].keys())

	print('LFP pops included --> ' + str(LFPPops))

	if type(electrodes) is list:
		for pop in LFPPops:
			popColorNum = LFPPops.index(pop)
			color = colorList[popColorNum%len(colorList)]
			#print('pop: ' + str(pop) + ' color: ' + str(color))

			lfp = np.array(allLFPData['LFPPops'][pop])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]


			data = {'lfp': lfp}  # returned data
			ydisp = 0.01 #0.02 #0.0025 #np.absolute(lfp).max() * 1.0 ## (1.0 --> separation)
			offset = 1.0*ydisp


			if filtFreq:
				from scipy import signal
				fs = 1000.0/sim.cfg.recordStep
				nyquist = fs/2.0
				filtOrder = 3
				if isinstance(filtFreq, list): # bandpass
					Wn = [filtFreq[0]/nyquist, filtFreq[1]/nyquist]
					b, a = signal.butter(filtOrder, Wn, btype='bandpass')
				elif isinstance(filtFreq, Number): # lowpass
					Wn = filtFreq/nyquist
					b, a = signal.butter(filtOrder, Wn)
				for i in range(lfp.shape[1]):
					lfp[:,i] = signal.filtfilt(b, a, lfp[:,i])



			first = True ## for labeling purposes!! 
			for i,elec in enumerate(electrodes):
				if isinstance(elec, Number): 
					lfpPlot = lfp[:, elec]
					lw = 1.0

				if first:
					first = False
					plt.plot(t, -lfpPlot+(i*ydisp),  linewidth=lw, label=pop, color = color) #-lfpPlot+(i*ydisp) #color=color,
				else:
					plt.plot(t, -lfpPlot+(i*ydisp),  linewidth=lw, color = color)

				if len(electrodes) > 1:
					plt.text(timeRange[0]-0.07*(timeRange[1]-timeRange[0]), (i*ydisp), elec, ha='center', va='top', fontweight='bold') # fontsize=fontSize, color=color,

			ax = plt.gca()

			data['lfpPlot'] = lfpPlot
			data['ydisp'] =  ydisp
			data['t'] = t

		if len(electrodes) > 1:
			plt.text(timeRange[0]-0.14*(timeRange[1]-timeRange[0]), (len(electrodes)*ydisp)/2.0, 'LFP electrode', color='k', ha='left', va='bottom', rotation=90) # fontSize=fontSize, 
			plt.ylim(-offset, (len(electrodes))*ydisp)
		else:
			plt.suptitle('LFP Signal', fontweight='bold') #fontSize=fontSize, 

		ax.invert_yaxis()
		plt.xlabel('time (ms)') # fontsize=fontSize
		plt.legend()
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.get_yaxis().set_visible(False)
		plt.subplots_adjust(hspace=0.2)#bottom=0.1, top=1.0, right=1.0) # top = 1.0


		filename = based + 'try_this.png'
		plt.savefig(filename,bbox_inches='tight')

	elif electrodes is 'avg': # works as == 'avg' or is 'avg'  ### NEED TO CHANGE HOW THIS DISTINGUISHES BTWN AVG AND NUMBERED ELECTRODES!!
		print('avg electrode plotting') ## PRINT TEST LINE 



#### Plot LFP or CSD or TRACES ##### 
if LFP or CSD or traces:
	fileNameFull = based + dataFiles[0]
	sim.load(fileNameFull, instantiate=False) ## Doesn't matter which file was last to load for sim in this case --> should all be the same except for subsets of LFP cell contrib saved 


	if LFP and not LFPPopContrib:
		sim.analysis.plotLFP(plots=['spectrogram'],filtFreq = filtFreq,normSignal=True,electrodes=electrodes,showFig=True, timeRange=timeRange) #figSize=(5,5)) # electrodes=[2,6,11,13] # saveFig=figname, saveFig=True, plots=['PSD', 'spectrogram']
		#sim.plotting.plotSpectrogram()
	elif LFP and LFPPopContrib:  ## maybe change this? sorta strange; at least needs labels for which pop I'm looking at. 
		LFPPops = list(allLFPData['LFPPops'].keys())
		for pop in LFPPops:
			sim.analysis.plotLFP(plots=['spectrogram'],pop=pop,filtFreq = filtFreq,normSignal=True,electrodes=electrodes,showFig=True, timeRange=timeRange), #figSize=(5,5)) # electrodes=[2,6,11,13] # saveFig=figname, saveFig=True, plots=['PSD', 'spectrogram']

	if CSD:
		sim.analysis.plotCSD(spacing_um=100, timeRange=timeRange, overlay='CSD', hlines=0, layerLines=1, layerBounds = layerBounds,saveFig=0, figSize=(5,5), showFig=1) # LFP_overlay=True

	if traces:
		sim.analysis.plotTraces(timeRange = timeRange, oneFigPer='cell', overlay=True, saveFig=False, showFig=True)# include=[(pop, 0) for pop in L4pops],  #, figSize=(6,8))#figSize=(12,8))




# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------

#if __name__ == '__main__':







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
import seaborn as sns 
import pandas as pd 
from mpl_toolkits.axes_grid1 import make_axes_locatable



######################################################################
##### plotMUA and plotCustomLFPTimeSeries are NOT BEING USED !!! #####
def plotMUA(dataFile, colorList, timeRange=None, pops=None):
	#### plotMUA FUNCTION STILL NOT WORKING WELL FOR TIME RANGES WITH LOW VALUES AT timeRange[0]!!! FIGURE OUT WHY THIS IS! 
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
def plotCustomLFPTimeSeries(dataFile, colorList, filtFreq, electrodes=['avg'], showFig=1, saveFig=0, figsize=None, timeRange=None, pops=None):
	### dataFile: str
	### colorList: list
	### filtFreq: list 
	### electrodes: list 
	### showFig: bool
	### saveFig: bool
	### figsize: e.g. (6,8)
	### timeRange: list --> e.g. [start, stop]
	### pops: list 


	## Load sim data
	sim.load(dataFile, instantiate=False)

	## Timepoints 
	if timeRange is None:
		timeRange = [0, sim.cfg.duration]
	t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)  ## make an array w/ these time points
	t = list(t)

	## LFP populations to include!
	if pops is None:
		pops = list(sim.allSimData['LFPPops'].keys())	# all pops with LFP data recorded! 
	print('LFP pops included --> ' + str(pops))

	## figure size 
	if figsize is None:
		figsize = (6,8)

	plt.figure(figsize=figsize)

	if electrodes == ['avg']:
		print('averaged electrodes -- STILL HAVE TO DEVELOP THIS CODE; OR USE OTHER LFP PLOTTING FUNCTION')
	else:  ## THIS IS IF ELECTRODES IS A LIST OF INTS 
		for pop in pops:
			popColorNum = pops.index(pop)
			color = colorList[popColorNum%len(colorList)]
			#print('pop: ' + str(pop) + ' color: ' + str(color))

			lfp = np.array(sim.allSimData['LFPPops'][pop])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]

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


		filename = 'try_this.png'
		if showFig:
			plt.show()
		if saveFig:
			plt.savefig(filename,bbox_inches='tight')
######################################################################



### FUNCTIONS ####
def getWaveletInfo(freqBand, based, verbose=0): 
	## freqband: str  --> e.g. 'delta', 'alpha', 'theta'
	## based: str --> path to directory with the .pkl data files 
	## verbose: bool --> if 0, default to only putting out timeRange and dataFile, if 1 --> include channel as well 

	# waveletInfo = {'delta': {'dataFile': 'A1_v34_batch65_v34_batch65_0_0_data.pkl', 'timeRange': [1480, 2520]},
	# 'beta': {'dataFile': 'A1_v34_batch65_v34_batch65_1_1_data.pkl', 'timeRange': [456, 572]}, 
	# 'alpha': {'dataFile': 'A1_v34_batch65_v34_batch65_1_1_data.pkl', 'timeRange': [3111, 3325]}, 
	# 'theta': {'dataFile': 'A1_v34_batch65_v34_batch65_2_2_data.pkl', 'timeRange': [2785, 3350]}}

	waveletInfo = {'delta': {'dataFile': 'A1_v34_batch65_v34_batch65_0_0_data.pkl', 'timeRange': [1480, 2520], 'channel': 14},
	'beta': {'dataFile': 'A1_v34_batch67_v34_batch67_0_0_data.pkl', 'timeRange': [456, 572], 'channel': 14}, 
	'alpha': {'dataFile': 'A1_v34_batch67_v34_batch67_0_0_data.pkl', 'timeRange': [3111, 3325], 'channel': 9}, 
	'theta': {'dataFile': 'A1_v34_batch67_v34_batch67_1_1_data.pkl', 'timeRange': [2785, 3350], 'channel': 8}}

	timeRange = waveletInfo[freqBand]['timeRange']
	dataFileNoPath = waveletInfo[freqBand]['dataFile']
	dataFile = based + dataFileNoPath
	channel = waveletInfo[freqBand]['channel']

	if verbose:
		return timeRange, dataFile, channel
	else:
		return timeRange, dataFile


def getDataFiles(based):
	### based: str ("base directory")
	### returns list of .pkl files --> allDataFiles

	allFiles = os.listdir(based)
	allDataFiles = []
	for file in allFiles:
		if '.pkl' in file:
			allDataFiles.append(file)
	return allDataFiles 


def getDataFrames(dataFile, timeRange, verbose=0):
	### This function will return data frames of peak and average lfp amplitudes for picking cell pops
	### dataFile: str --> .pkl file to load
	### timeRange: list --> e.g. [start, stop]
	### verbose: bool --> if 0, return only the data frames; if 1 - return all lists and dataframes 

	## Load data file
	sim.load(dataFile, instantiate=False)

	## Get all cell pops (cortical)
	thalPops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']
	allPops = list(sim.net.allPops.keys())
	pops = [pop for pop in allPops if pop not in thalPops] 			## exclude thal pops 

	## Get all electrodes 
	evalElecs = []
	evalElecs.extend(list(range(int(sim.net.recXElectrode.nsites))))
	### add 'avg' to electrode list 
	evalElecs.append('avg')


	## Create dict with lfp population data, and calculate average and peak amplitudes for each pop at each electrode!
	lfpPopData = {}

	for pop in pops:
		lfpPopData[pop] = {}

		popLFPdata = np.array(sim.allSimData['LFPPops'][pop])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]

		for i, elec in enumerate(evalElecs):
			if elec == 'avg':
				lfpPopData[pop]['avg'] = {}

				avgPopData = np.mean(popLFPdata, axis=1) # list of lfp data (averaged over all electrodes) for each timepoint 

				avgAvgLFP = np.average(avgPopData)
				lfpPopData[pop]['avg']['avg'] = avgAvgLFP

				peakAvgLFP = np.amax(avgPopData)
				lfpPopData[pop]['avg']['peak'] = peakAvgLFP

			elif isinstance(elec, Number):
				elecKey = 'elec' + str(elec)
				lfpPopData[pop][elecKey] = {}
				lfpPopData[pop][elecKey]['avg'] = np.average(popLFPdata[:, elec]) ### <-- IS THIS WHAT I WANT? 
				lfpPopData[pop][elecKey]['peak'] = np.amax(popLFPdata[:, elec])


	## Rearrange data structure and turn into a pandas data frame
	#### PEAK LFP AMPLITUDES ####
	peakValues = {}

	peakValues['pops'] = []
	peakValues['peakLFP'] = [[] for i in range(len(pops))]  # should be 36! 
	p=0
	for pop in pops:
		peakValues['pops'].append(pop)
		for i in range(len(evalElecs)):
			if isinstance(evalElecs[i], Number):
				elecKey = 'elec' + str(i)
			elif evalElecs[i] == 'avg':
				elecKey = 'avg'
			peakValues['peakLFP'][p].append(lfpPopData[pop][elecKey]['peak'])
		p+=1

	dfPeak = pd.DataFrame(peakValues['peakLFP'], index=pops)


	#### AVERAGE LFP AMPLITUDES ####
	avgValues = {}

	avgValues['pops'] = []
	avgValues['avgLFP'] = [[] for i in range(len(pops))]

	q=0
	for pop in pops:
		avgValues['pops'].append(pop)
		for i in range(len(evalElecs)):
			if isinstance(evalElecs[i], Number):
				elecKey = 'elec' + str(i)
			elif evalElecs[i] == 'avg':
				elecKey = 'avg'
			avgValues['avgLFP'][q].append(lfpPopData[pop][elecKey]['avg'])
		q+=1 

	dfAvg = pd.DataFrame(avgValues['avgLFP'], index=pops)

	if verbose:
		return dfPeak, dfAvg, peakValues, avgValues, lfpPopData 
	else:
		return dfPeak, dfAvg


def plotDataFrames(dataFrame, electrodes=None, pops=None, cbarLabel=None, title=None, figSize=None, saveFig=True):
	### dataFrame: pandas dataFrame (can be obtained from getDataFrames function above)
	### electrodes: list, if none --> default: use all electrodes + 'avg'
	### pops: list, if none --> default: 
	### cbarLabel: str, label for color bar, optional
	### title: str, also optional 


	## Set label for color scalebar 
	if cbarLabel is None:
		cbarLabel = 'LFP amplitudes (mV)'

	## Create lists of electrode (columns and labels)
	if electrodes is None:
		electrodeColumns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
		electrodeLabels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 'avg']
	else:
		if 'avg' in electrodes:
			electrodeColumns = electrodes.copy()
			avgIndex = electrodeColumns.index('avg')
			electrodeColumns[avgIndex] = 20
		else:
			electrodeColumns = electrodes.copy() 
		electrodeLabels = electrodes.copy() 
		print('electrodeLabels' + str(electrodeLabels))

	## dataFrame subset according to electrodes specified in argument! 
	dataFrame = dataFrame[electrodeColumns]

	## Create list of cell populations 
	if pops is None:  ### POTENTIAL ISSUE WITH ORDERING OR NO???
		pops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 
		'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 
		'IT5B', 'CT5B', 'PT5B', 'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6']
	
	## dataFrame subset according to cell populations specified in argument! 
	dataFrame = dataFrame[dataFrame.index.isin(pops)]

	## TRANSPOSE DATA FRAME 
	pivotedDataFrame = dataFrame.T


	## Create lists of x and y axis labels 
	x_axis_labels = pops.copy() 
	y_axis_labels = electrodeLabels.copy() 


	## Set size of figure 
	if figSize is None:
		figSize = (12,6)
	plt.figure(figsize = figSize)


	## Set title of figure 
	if title is not None:
		plt.title(title)


	## Create heatmap! 
	ax = sns.heatmap(pivotedDataFrame, xticklabels=x_axis_labels, yticklabels=y_axis_labels, linewidth=0.4, cbar_kws={'label': cbarLabel})

	## Set labels on x and y axes 
	plt.xlabel('Cell populations')
	plt.xticks(rotation=45, fontsize=7)
	plt.ylabel('Electrodes')
	plt.yticks(rotation=0, fontsize=8)

	return ax



def evalPops(dataFrame):
	### dataFrame: pandas dataFrame --> can be gotten from getDataFrames

	includePopsRel = []
	includePopsAbs = []
	return includePopsRel, includePopsAbs

# def evalPopsRelative():
	# includePopsRel = []
	# return includePopsRel

# def evalPopsAbsolute():
# 	## CONDITIONS: 
# 	## (1) lfp signal avg'ed over all electrodes for each pop
# 	## (2) lfp signal for particular electrodes for each pop 
# 	## ADDRESS (2) LATER!!

# 	includePopsAbs = []
# 	return includePopsAbs




def getSpikeData(dataFile, graphType, pop, timeRange=None): 
	### dataFile: path to .pkl data file to load 
	### graphType: str --> either 'hist' or 'spect'
	### pop: list or str --> which pop to include 
	### timeRange: list --> e.g. [start, stop]

	sim.load(dataFile, instantiate=False)

	if type(pop) is str:
		popList = [pop]
	elif type(pop) is list:
		popList = pop

	if timeRange is None:
		timeRange = [0, sim.cfg.duration]

	if graphType is 'spect':
		spikeDict = sim.analysis.getRateSpectrogramData(include=popList, timeRange=timeRange)
	elif graphType is 'hist':
		spikeDict = sim.analysis.getSpikeHistData(include=popList, timeRange=timeRange, binSize=5, graphType='bar', measure='rate')

	return spikeDict 


def plotCombinedSpike(spectDict, histDict, timeRange, colorDict, pop, figSize=(10,7), fontSize=12, saveFig=True):
	### spectDict: dict --> can be gotten with getSpikeData(graphType='spect')
	### histDict: dict --> can be gotten with getSpikeData(graphType='hist')
	### colorDict: dict 
	### pop: str or list of length 1 --> population to include 
	### timeRange: tuple 
	### figSize: tuple
	### fontSize: int 

	# Get relevant pop
	if type(pop) is str:
		popToPlot = pop
	elif type(pop) is list:
		popToPlot = pop[0]

	# Create figure 
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figSize) #, constrained_layout=True)

	# Set font size
	fontsiz = fontSize
	plt.rcParams.update({'font.size': fontsiz})

	### SPECTROGRAM -- for top panel!!
	allSignal = spectDict['allSignal']
	allFreqs = spectDict['allFreqs']

	ax1 = plt.subplot(211)
	img = ax1.imshow(allSignal[0], extent=(np.amin(timeRange), np.amax(timeRange), np.amin(allFreqs[0]), np.amax(allFreqs[0])), origin='lower', 
		interpolation='None', aspect='auto', cmap=plt.get_cmap('viridis'))
	divider1 = make_axes_locatable(ax1)
	cax1 = divider1.append_axes('right', size='3%', pad = 0.2)
	plt.colorbar(img, cax = cax1, orientation='vertical', label='Power')
	ax1.set_title('Spike Rate Spectrogram for ' + popToPlot)
	# ax1.set_xlabel('Time (ms)', fontsize=fontsiz)
	ax1.set_ylabel('Hz', fontsize=fontsiz)
	ax1.set_xlim(left=timeRange[0], right=timeRange[1])
	# ax1.margins(x=0)


	### HISTOGRAM -- for bottom panel!! 
	histoT = histDict['histoT']
	histoCount = histDict['histoData']

	ax2 = plt.subplot(212)
	ax2.bar(histoT, histoCount[0], width = 5, color=colorDict[popToPlot], fill=True)
	divider2 = make_axes_locatable(ax2)
	cax2 = divider2.append_axes('right', size='3%', pad = 0.2)
	cax2.axis('off')
	ax2.set_title('Spike Rate Histogram for ' + popToPlot)
	ax2.set_xlabel('Time (ms)', fontsize=fontsiz)
	ax2.set_ylabel('Rate (Hz)', fontsize=fontsiz) # CLARIFY Y AXIS
	ax2.set_xlim(left=timeRange[0], right=timeRange[1])
	# ax2.margins(x=0)
	plt.show()


	# plt.suptitle(popToPlot)
	plt.tight_layout()

	if saveFig:
		prePath = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/popContribFigs/'
		fileName = pop + '_combinedSpike.png'
		pathToFile = prePath + fileName
		plt.savefig(pathToFile)


def getLFPDataDict(dataFile, pop, timeRange, plotType, filtFreq=None, electrodes=['avg']):
	### dataFile: str --> path to .pkl data file to load for analysis 
	### pop: str or list  
	### timeRange: list, e.g. [start, stop]
	### plotType: str or list --> 'spectrogram' or 'timeSeries'
	### filtFreq: list --> frequencies to bandpass filter lfp data 
	### electrodes: list --> default: ['avg'] --> # SINGLE ELEMENT LIST; correct this or improve it?!?

	# Load data file 
	sim.load(dataFile, instantiate=False)

	# Pops
	if type(pop) is str:
		popList = [pop]
	elif type(pop) is list:
		popList = pop

	# Set up which kind of data -- i.e. timeSeries or spectrogram 
	if type(plotType) is str:
		plots = [plotType]
	elif type(plotType) is list:
		plots = plotType

	lfpOutput = sim.analysis.getLFPData(pop=popList, timeRange=timeRange, electrodes=electrodes, filtFreq=filtFreq, plots=plots)


	return lfpOutput


def plotCombinedLFP(spectDict, timeSeriesDict, timeRange, pop, colorDict, ylim=None, figSize=(10,7), fontSize=12, titleSubset=None, saveFig=True): # electrode='avg',
	### spectDict: dict with spectrogram data
	### timeSeriesDict: dict with timeSeries data
	### timeRange: list, e.g. [start, stop]
	### colorDict: dict --> corresponds pop to color 
	# 	MAY NOT BE NECESSARY  --> ### electrode: str or int --> (LENGTH OF 1 -- FIX THIS LATER?) electrodes better??? 
	### pop: list or str --> relevant population to plot data for 
	### ylim: list --> [min, max]
	### figSize: tuple --> default is (10,7)
	### fontSize: int --> default is 12 

	# Get relevant pop
	if type(pop) is str:
		popToPlot = pop
	elif type(pop) is list:
		popToPlot = pop[0]


	## Create figure 
	figSize = (10,7)
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figSize)

	## Set font size 
	fontsiz = fontSize
	plt.rcParams.update({'font.size': fontsiz})



	##### SPECTROGRAM  --> TOP PANEL !! 
	# lfp = spectDict['LFP']  # #<--- UNUSED
	spec = spectDict['spec']  ### DO I NEED TO CHANGE THIS IF ELECTRODES...??? NO, DOES THAT IN lfp_orig.py I believe!!!

	S = spec[0].TFR
	F = spec[0].f   
	T = timeRange

	vmin = np.array([s.TFR for s in spec]).min()
	vmax = np.array([s.TFR for s in spec]).max()
	vc = [vmin, vmax]

	ax1 = plt.subplot(2, 1, 1)
	img = ax1.imshow(S, extent=(np.amin(T), np.amax(T), np.amin(F), np.amax(F)), origin='lower', interpolation='None', aspect='auto', vmin=vc[0], vmax=vc[1], cmap=plt.get_cmap('viridis'))
	divider1 = make_axes_locatable(ax1)
	cax1 = divider1.append_axes('right', size='3%', pad=0.2)
	plt.colorbar(img, cax = cax1, orientation='vertical', label='Power')
	if titleSubset is not None:
		spectTitle = 'LFP Spectrogram for ' + popToPlot + titleSubset
	else:
		spectTitle = 'LFP Spectrogram for ' + popToPlot
	ax1.set_title(spectTitle)	#('LFP Spectrogram for ' + popToPlot)
	ax1.set_ylabel('Hz')
	# ax1.set_xlabel('Time (ms)')
	ax1.set_xlim(left=timeRange[0], right=timeRange[1])
	if ylim is not None:
		ax1.set_ylim(ylim[0], ylim[1])




	##### TIME SERIES  ## ON BOTTOM PANEL !! 
	t = timeSeriesDict['t']
	lfp = timeSeriesDict['LFP'] # timeSeriesDict['lfpPlot'] #timeSeriesDict['LFP'] ### ELECTRODE FIX???

	separation = 1 
	ydisp = np.absolute(lfp).max() * separation
	# if electrode == 'avg' :
	# 	lfpPlot = np.mean(lfp, axis=1)
	# elif isinstance(electrode, Number):
	# 	lfpPlot = lfp[:, electrode]  ### NEED THIS SECOND ELEMENT IN LIST TO BE ELECTRODE NUM!!!

	### TEST LINE ### <-- should do same as above electrodes lines?? BUT I DID TAKE OUT ELECTRODE ARGUMENT!!!
	lfpPlot = timeSeriesDict['lfpPlot']

	lw = 1.0
	ax2 = plt.subplot(2, 1, 2)
	divider2 = make_axes_locatable(ax2)
	cax2 = divider2.append_axes('right', size='3%', pad=0.2)
	cax2.axis('off')
	ax2.plot(t[0:len(lfpPlot)], -lfpPlot+(1*ydisp), color=colorDict[popToPlot], linewidth=lw)  # plt.plot(t[0:len(lfpPlot)], -lfpPlot, color=colorDict[popToPlot], linewidth=lw)  # color=colorDict[includePops[0]] # -lfpPlot+(i*ydisp)
	if titleSubset is not None:
		timeSeriesTitle = 'LFP Signal for ' + popToPlot + titleSubset
	else:
		timeSeriesTitle = 'LFP Signal for ' + popToPlot
	ax2.set_title(timeSeriesTitle) 	#('LFP Signal for ' + popToPlot)
	ax2.set_xlabel('Time (ms)')
	ax2.set_xlim(left=timeRange[0], right=timeRange[1]) # plt.margins(x=0)
	ax2.set_ylabel('LFP Amplitudes, mV') ### UNITS uV or mV?? 

	# plt.suptitle(popToPlot)
	plt.tight_layout()
	plt.show()

	if saveFig:
		prePath = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/popContribFigs/'
		if titleSubset:
			elecInFileName = titleSubset[-2:]
		else:
			elecInFileName = 'avgElecs'
		fileName = pop + '_combinedLFP_' + 'elec' + elecInFileName + '.png'
		pathToFile = prePath + fileName
		plt.savefig(pathToFile)





### USEFUL VARIABLES ### 
## set layer bounds:
layerBounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550, 'L6': 2000}

## Cell populations: 
allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4',
'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B',
'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']#, 'IC']
L1pops = ['NGF1']
L2pops = ['IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2']
L3pops = ['IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3']
L4pops = ['ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4']
L5Apops = ['IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A']
L5Bpops = ['IT5B', 'PT5B', 'CT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B']
L6pops = ['IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6']
thalPops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']
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

## for COLORS in DATA PLOTTING!! 
colorList = [[0.42,0.67,0.84], [0.90,0.76,0.00], [0.42,0.83,0.59], [0.90,0.32,0.00],
            [0.34,0.67,0.67], [0.90,0.59,0.00], [0.42,0.82,0.83], [1.00,0.85,0.00],
            [0.33,0.67,0.47], [1.00,0.38,0.60], [0.57,0.67,0.33], [0.5,0.2,0.0],
            [0.71,0.82,0.41], [0.0,0.2,0.5], [0.70,0.32,0.10]]*3

colorDict = {}
for p in range(len(allpops)):
	colorDict[allpops[p]] = colorList[p]



#############################################
#### DATA FILES -- SET BOOLEANS HERE !! #####
#############################################

######### SET LOCAL BOOL	!!
local = 1								# if using local machine (1) 	# if using neurosim or other (0)
## Path to data files
if local:
	based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/'  # '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/miscRuns/shortRuns/'


######## SET WAVELET TO LOOK AT		!!

# wavelets = ['delta', 'beta', 'alpha', 'theta']

# for wavelet in wavelets:
# 	timeRange, dataFile = getWaveletInfo(wavelet, based)
# 	dfPeak, dfAvg = getDataFrames(dataFile=dataFile, timeRange=timeRange)
# 	peakTitle = 'Peak LFP Amplitudes of ' + wavelet + ' Wavelet'
# 	peakPlot = plotDataFrames(dfPeak, pops=ECortPops, title=peakTitle)
# 	plt.show(peakPlot)
# 	avgTitle = 'Avg LFP Amplitudes of ' + wavelet + ' Wavelet'
# 	avgPlot = plotDataFrames(dfAvg, pops=ECortPops, title=avgTitle)
# 	plt.show(avgPlot)

#########
delta = 0
beta = 	0
alpha = 0
theta = 1
# gamma = 0 

if delta:
	timeRange, dataFile = getWaveletInfo('delta', based)
	wavelet='delta' ### MOVE THESE EVENTUALLY -- BEING USED FOR peakTitle
elif beta:
	timeRange, dataFile = getWaveletInfo('beta', based)
	wavelet='beta'
elif alpha:
	timeRange, dataFile = getWaveletInfo('alpha', based) ## recall timeRange issue (see nb)
	wavelet='alpha'
elif theta:
	timeRange, dataFile = getWaveletInfo('theta', based)
	wavelet='theta'
# elif gamma:
# 	print('Cannot analyze gamma wavelet at this time')



########################
####### PLOTTING #######
########################

#### EVALUATING POPULATIONS TO CHOOSE #### 
## TO DO: Make a function that outputs list of pops vs. looking at it graphically (how many pops to include? avg or peak?)

evalPopsBool = 0

if evalPopsBool:
	dfPeak, dfAvg = getDataFrames(dataFile=dataFile, timeRange=timeRange)

	peakTitle = 'Peak LFP Amplitudes of ' + wavelet + ' Wavelet'
	peakPlot = plotDataFrames(dfPeak, pops=ECortPops, title=peakTitle)#title='Peak LFP Amplitudes') # pops=ECortPops,
	plt.show(peakPlot)

	avgTitle = 'Avg LFP Amplitudes of ' + wavelet + ' Wavelet'
	avgPlot = plotDataFrames(dfAvg, pops=ECortPops, title=avgTitle)#title='Avg LFP Amplitudes') # pops=ECortPops,
	plt.show(avgPlot)


########################
includePops = ['IT3', 'IT5A', 'IT5B', 'CT5B', 'PT5B']		# placeholder for now <-- will ideally come out of the function above once the pop LFP netpyne issues get resolved! 

###### COMBINED SPIKE DATA PLOTTING ######
## TO DO: 
## Smooth or mess with bin size to smooth out spectrogram for spiking data

plotSpikeData = 0

if plotSpikeData:
	for pop in includePops:
		print('Plotting spike data for ' + pop)

		## Get dictionaries with spiking data for spectrogram and histogram plotting 
		spikeSpectDict = getSpikeData(dataFile, graphType='spect', pop=pop, timeRange=timeRange)  # pop=includePops
		histDict = getSpikeData(dataFile, graphType='hist', pop=pop, timeRange=timeRange)		# pop=includePops


		## Then call plotting function 
		plotCombinedSpike(spectDict=spikeSpectDict, histDict=histDict, timeRange=timeRange, colorDict=colorDict, pop=pop, figSize=(10,7), fontSize=12)  # pop=includePops, 



###### COMBINED LFP PLOTTING ######
## TO DO: 
## [IN PROGRESS] Filter the timeRanged lfp data to the wavelet frequency band
## Could also compare the change in lfp amplitude from "baseline"  (e.g. some time window before the wavelet and then during the wavelet event) 

plotLFPCombinedData = 1

# ### BAND PASS FILTER LFP DATA -- MOVE THIS INTO WAVELET INFO 
# filtFreq: delta (0.5-4 Hz), theta (4-9 Hz), alpha (9-15 Hz), beta (15-29 Hz), gamma (30-80 Hz)
bandpassLFP = 0
if bandpassLFP:
	if delta:
		filtFreq = 5 #[0.5, 5] # Hz
	elif beta:
		filtFreq = [21, 40] # [15, 29]
	elif alpha:
		filtFreq = [11, 17]	#[9, 15]
	elif theta:
		filtFreq = 30 #[4, 9] #[5, 7.25] #[4, 9]
else:
	filtFreq = None


includePops = ['PT5B']	# ['IT3', 'IT5A', 'IT5B', 'CT5B', 'PT5B']		# placeholder for now <-- will ideally come out of the function above once the pop LFP netpyne issues get resolved! 

if plotLFPCombinedData:
	for pop in includePops:
		print('Plotting LFP spectrogram and timeSeries for ' + pop)

		if pop == 'IT3':
			electrodes = [1] #[1,9] #['avg']
		elif pop == 'IT5A':
			electrodes = [10]
		elif pop == 'CT5B':
			electrodes = [11]
		elif pop == 'PT5B':
			electrodes = [11]
		else:
			electrodes = ['avg']

		## Get dictionaries with LFP data for spectrogram and timeSeries plotting  
		LFPSpectOutput = getLFPDataDict(dataFile, pop=pop, timeRange=timeRange, plotType=['spectrogram'], electrodes=electrodes) 
		LFPtimeSeriesOutput = getLFPDataDict(dataFile, pop=pop, timeRange=timeRange, plotType=['timeSeries'], electrodes=electrodes) #filtFreq=filtFreq, 


		### Call plotting function 
		if delta:
			ylim = [1, 40]
		else:
			ylim=None
		if electrodes[0] == 'avg':
			plotTitleSubset= ' (averaged over all electrodes)'
		elif isinstance(electrodes[0], Number):
			plotTitleSubset = ', electrode ' + str(electrodes[0])
		plotCombinedLFP(spectDict=LFPSpectOutput, timeSeriesDict=LFPtimeSeriesOutput, timeRange=timeRange, pop=pop, colorDict=colorDict, ylim=ylim, figSize=(10,7), fontSize=12, titleSubset=plotTitleSubset)





######################################################
############ NOT BEING USED AT THE MOMENT ############
######################################################

# #### MUA PLOTTING ####
# MUA = 0				## bool (0 or 1) 
# MUApops = ['IT2', 'IT3'] # 0			## bool OR list of pops --> e.g. ['ITP4', 'ITS4'] # ECortPops.copy()

# if MUA: 
# 	if MUApops:
# 		plotMUA(dataFile, colorList, timeRange, pops=MUApops)
# 	else:
# 		plotMUA(dataFile, colorList, timeRange, pops=None)



# #### LFP CUSTOM TIME SERIES PLOTTING ####
# customLFPtimeSeries = 0						## bool (0 or 1)
# customLFPPops = ['ITP4', 'ITS4']	## list or bool (0)
# filtFreq = [13,30]
# customLFPelectrodes = [3, 4, 5, 6]

# if customLFPtimeSeries:
# 	plotCustomLFPTimeSeries(dataFile, colorList, filtFreq=filtFreq, electrodes=customLFPelectrodes, timeRange=timeRange, pops=customLFPPops) # showFig=1, saveFig=0, figsize=None 



# ### CSD, TRACES ### 
# CSD = 0  		### <-- Make this work like LFP plotting (for individual pops!) and make sure time error is not in this version of csd code! 
# # traces = 0 	### <-- NOT WORKING RIGHT NOW !!


# #### PLOT CSD or TRACES ##### 
# if CSD:  
# 	sim.load(dataFile, instantiate=False) 
# 	sim.analysis.plotCSD(spacing_um=100, timeRange=timeRange, overlay='CSD', hlines=0, layer_lines=1, layer_bounds=layerBounds, saveFig=0, figSize=(5,5), showFig=1) # LFP_overlay=True


# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------

#if __name__ == '__main__':







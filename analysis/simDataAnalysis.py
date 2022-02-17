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
import matplotlib.ticker  ## for colorbar 
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import netpyne
from numbers import Number
import seaborn as sns 
import pandas as pd 
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable



######################################################################
##### These functions are currently NOT BEING USED !!! #####
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
def getBandpassRange(freqBand):
	##### --> This function correlates bandpass frequency range for each frequency band; can be used for bandpassing LFP data 
	##### --> delta (0.5-4 Hz), theta (4-9 Hz), alpha (9-15 Hz), beta (15-29 Hz), gamma (30-80 Hz)
	### freqBand: str --> e.g. 'delta', 'alpha', 'beta', 'gamma', 'theta'

	if freqBand == 'delta':
		filtFreq = 5 #[0.5, 5] # Hz
	elif freqBand =='beta':
		filtFreq = [21, 40] # [15, 29]
	elif freqBand == 'alpha':
		filtFreq = [11, 17]	#[9, 15]
	elif freqBand == 'theta':
		filtFreq = 30 #[4, 9] #[5, 7.25]
	else:
		filtFreq = None

	return filtFreq
def getDataFiles(based):
	### based: str ("base directory")
	#### ---> This function returns a list of all the .pkl files (allDataFiles) in the 'based' directory

	allFiles = os.listdir(based)
	allDataFiles = []
	for file in allFiles:
		if '.pkl' in file:
			allDataFiles.append(file)
	return allDataFiles 
######################################################################

######################################################################
##### evalPops functions are NOT YET FINISHED!!! ##### 
def evalPopsRelative():
	includePopsRel = []
	return includePopsRel
def evalPopsAbsolute():
	## CONDITIONS: 
	## (1) lfp signal avg'ed over all electrodes for each pop
	## (2) lfp signal for particular electrodes for each pop 
	## ADDRESS (2) LATER!!

	includePopsAbs = []
	return includePopsAbs
######################################################################

######################################################################
#### FUNCTION(S) IN PROGRESS 
def evalFreqBand(freqBand, dlmsPklFile):
	## freqBand: str 	--> e.g. 'alpha', 'beta' ,'theta', 'delta', 'gamma'
	## dlmsPklFile: .pkl file 	--> from dlms.pkl file, saved from load.py 
	print('Evaluating all oscillation events in a given frequency band')

	# Load dlms file
	subjectDir = dlms.split('_dlms.pkl')[0]
	dlmsFullPath = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/wavelets/' + subjectDir + '/' + dlmsPklFile
	dlmsFile = open(dlmsFullPath, 'rb')
	dlms = pickle.load(dlmsFile)
	dlmsFile.close()
######################################################################



###################
#### FUNCTIONS ####
###################
def getWaveletInfo(freqBand, based, verbose=0): 
	## freqBand: str  --> e.g. 'delta', 'alpha', 'theta'
	## based: str --> path to directory with the .pkl data files 
	## verbose: bool --> if 0, default to only putting out timeRange and dataFile, if 1 --> include channel as well 

	waveletInfo = {
	'delta': {'dataFile': 'A1_v34_batch65_v34_batch65_0_0_data.pkl', 'timeRange': [1480, 2520], 'channel': 14},
	'beta': {'dataFile': 'A1_v34_batch67_v34_batch67_0_0_data.pkl',	'timeRange': [456, 572], 'channel': 14}, 
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

## Heatmaps ## 
def getDataFrames(dataFile, timeRange, verbose=0):
	#### -->  This function will return data frames of peak and average lfp amplitudes, for picking cell pops
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

				avgPopData = np.mean(popLFPdata, axis=1) 		## lfp data (from 1 pop) at each timepoint, averaged over all electrodes

				avgAvgLFP = np.average(avgPopData)
				lfpPopData[pop]['avg']['avg'] = avgAvgLFP 		## time-average of lfp data (from 1 pop) that has been averaged in space (over all electrodes)

				peakAvgLFP = np.amax(avgPopData)
				lfpPopData[pop]['avg']['peak'] = peakAvgLFP 	## highest datapoint of all the lfp data (from 1 pop) that has been averaged in space (over all electrodes)

			elif isinstance(elec, Number):
				elecKey = 'elec' + str(elec)
				lfpPopData[pop][elecKey] = {}
				lfpPopData[pop][elecKey]['avg'] = np.average(popLFPdata[:, elec]) 	## LFP data (from 1 pop) averaged in time, over 1 electrode 
				lfpPopData[pop][elecKey]['peak'] = np.amax(popLFPdata[:, elec]) 	## Maximum LFP value from 1 pop, over time, recorded at 1 electrode 


	#### PEAK LFP AMPLITUDES, DATA FRAME ####
	peakValues = {}
	peakValues['pops'] = []
	peakValues['peakLFP'] = [[] for i in range(len(pops))]  # should be 36! 
	p=0
	for pop in pops:
		peakValues['pops'].append(pop)
		for i, elec in enumerate(evalElecs): 
			if isinstance(elec, Number):
				elecKey = 'elec' + str(elec)
			elif elec == 'avg': 
				elecKey = 'avg'
			peakValues['peakLFP'][p].append(lfpPopData[pop][elecKey]['peak'])
		p+=1
	dfPeak = pd.DataFrame(peakValues['peakLFP'], index=pops)


	#### AVERAGE LFP AMPLITUDES, DATA FRAME ####
	avgValues = {}
	avgValues['pops'] = []
	avgValues['avgLFP'] = [[] for i in range(len(pops))]
	q=0
	for pop in pops:
		avgValues['pops'].append(pop)
		for i, elec in enumerate(evalElecs):
			if isinstance(elec, Number):
				elecKey = 'elec' + str(elec)
			elif elec == 'avg':
				elecKey = 'avg'
			avgValues['avgLFP'][q].append(lfpPopData[pop][elecKey]['avg'])
		q+=1
	dfAvg = pd.DataFrame(avgValues['avgLFP'], index=pops)


	if verbose:
		return dfPeak, dfAvg, peakValues, avgValues, lfpPopData 
	else:
		return dfPeak, dfAvg
def plotDataFrames(dataFrame, electrodes=None, pops=None, title=None, cbarLabel=None, figSize=None, savePath=None, saveFig=True):
	#### --> This function will plot a heatmap of the peak or average LFP amplitudes across electrodes & cell populations
	### dataFrame: pandas dataFrame  --> These can be obtained from getDataFrames function above)
	### electrodes: list 	--> DEFAULT: use all electrodes + 'avg'
	### pops: list 			--> DEFAULT: all cortical pops 
	### title: str  		--> Optional; title of the entire figure
	### cbarLabel: str 		--> DEFAULT: 'LFP amplitudes (mV)'  -->  (label on the color scale bar)
	### figSize: tuple 		--> DEFAULT: (12,6)
	### savePath: str, path to directory where figures should be saved  --> DEFAULT: '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/'
	### saveFig: bool 		-->  DEFAULT: True 


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
			electrodeColumns[avgIndex] = 20  				## <-- Ensures that the correct index is used to access the 'avg' data
		else:
			electrodeColumns = electrodes.copy() 
		electrodeLabels = electrodes.copy() 
		# print('electrodeLabels' + str(electrodeLabels)) 	## <-- TESTING LINE (to make sure electrode labels for the plot are coming out as intended) 


	## dataFrame subset, according to the electrodes specified in the 'electrodes' argument! 
	dataFrame = dataFrame[electrodeColumns]

	## Create list of cell populations 
	if pops is None:  ### POTENTIAL ISSUE WITH ORDERING OR NO???
		pops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 
		'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 
		'IT5B', 'CT5B', 'PT5B', 'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6']


	## dataFrame subset according to cell populations specified in argument! 
	dataFramePops = dataFrame[dataFrame.index.isin(pops)]

	## TRANSPOSE DATA FRAME 
	pivotedDataFrame = dataFramePops.T

	## Set Font Sizes for the Heatmap Plot 
	titleFontSize = 20
	labelFontSize = 15
	tickFontSize = 10


	## Create lists of x and y axis labels 
	x_axis_labels = pops.copy() 
	y_axis_labels = electrodeLabels.copy() 


	## Set size of figure 
	if figSize is None:
		figSize = (12,6) 
	plt.figure(figsize = figSize)


	## Set title of figure 
	if title is not None:
		plt.title(title, fontsize=titleFontSize)


	## Create heatmap! 
	ax = sns.heatmap(pivotedDataFrame, xticklabels=x_axis_labels, yticklabels=y_axis_labels, linewidth=0.4, cbar_kws={'label': cbarLabel})

	## Set labels on x and y axes 
	plt.xlabel('Cell populations', fontsize=labelFontSize)
	plt.xticks(rotation=45, fontsize=tickFontSize)
	plt.ylabel('Electrodes', fontsize=labelFontSize)
	plt.yticks(rotation=0, fontsize=tickFontSize) 

	if saveFig:
		if savePath is None:
			prePath = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/LFP_heatmaps/'
		else:
			prePath = savePath
		fileName = 'heatmap.png'
		pathToFile = prePath + fileName
		plt.savefig(pathToFile, dpi=300)

	return ax

## Return top 5 & bottom 5 pop-electrode pairs ## 
def getPopElectrodeLists(evalPopsDict, verbose=0):
	## This function returns a list of lists, 2 elements long, with first element being a list of pops to include
	## 			and the second being the corresponding list of electrodes 
	### evalPopsDict: dict 	--> can be returned from def evalPops() below 
	### verbose: bool 		--> Determines PopElecLists is returned, or PopElecLists + includePops + electrodes 

	includePops = []
	electrodes = []
	for key in evalPopsDict:
		includePops.append(evalPopsDict[key]['pop'])
		electrodes.append(evalPopsDict[key]['electrode'])

	PopElecLists = [[]] * 2
	PopElecLists[0] = includePops.copy()
	PopElecLists[1] = electrodes.copy()

	if verbose:
		return PopElecLists, includePops, electrodes
	else:
		return PopElecLists
def evalPops(dataFrame):
	## NOTE: Add functionality to this such that near-same pop/electrode pairs are not included (e.g. IT3 electrode 10, IT3 electrode 11)
	### dataFrame: pandas dataFrame --> can be gotten from getDataFrames

	## MAXIMUM VALUES ## 
	maxPops = dataFrame.idxmax()
	maxPopsDict = dict(maxPops)
	maxPopsDict['avg'] = maxPopsDict.pop(20)

	maxValues = dataFrame.max()
	maxValuesDict = dict(maxValues)
	maxValuesDict['avg'] = maxValuesDict.pop(20)

	maxValuesDict_sorted = sorted(maxValuesDict.items(), key=lambda kv: kv[1], reverse=True)
	#############
	## ^^ This will result in something like:
	### [(9, 0.5670525182931198), (1, 0.3420748960387809), (10, 0.33742019248236954), 
	### (13, 0.32119689278509755), (12, 0.28783570785551094), (15, 0.28720528519895633), 
	### (11, 0.2639944261574631), (8, 0.22602826003777302), (5, 0.2033219839190444), 
	### (14, 0.1657402764440641), (16, 0.15516847639341205), (19, 0.107089151806042), 
	### (17, 0.10295759810425918), (18, 0.07873739475515538), (6, 0.07621437123298459), 
	### (3, 0.06367696913556453), (2, 0.06276483442249459), (4, 0.06113624787605265), 
	### (7, 0.057098300660935186), (0, 0.027993550732642168), ('avg', 0.016416523477252705)]
	
	## Each element of this ^^ is a tuple -- (electrode, value)
	## dict_sorted[0][0] -- electrode corresponding to the highest value
	## dict_sorted[0][1] -- highest LFP amplitude in the dataFrame 

	## dict_sorted[1][0] -- electrode corresponding to 2nd highest value
	## dict_sorted[1][1] -- second highest LFP amplitude in the dataFrame

	## dict_sorted[2][0] -- electrode corresponding to 3rd highest value
	## dict_sorted[2][1] -- third highest LFP amplitude in the dataFrame

	## maxPopsDict[dict_sorted[0][0]] -- pop associated w/ the electrode that corresponds to the highest Value 
	## maxPopsDict[dict_sorted[1][0]]
	## maxPopsDict[dict_sorted[2][0]]
	#############


	## MINIMUM VALUES ##  
	minPops = dataFrame.idxmin()
	minPopsDict = dict(minPops)				# minPopsList = list(minPops)
	minPopsDict['avg'] = minPopsDict.pop(20)

	minValues = dataFrame.min()
	minValuesDict = dict(minValues)			# minValuesList = list(minValues)
	minValuesDict['avg'] = minValuesDict.pop(20)

	minValuesDict_sorted = sorted(minValuesDict.items(), key=lambda kv: kv[1], reverse=False)



	### Get the pop / electrode pairing for top 5 and bottom 5 pops ### 
	popRank = ['first', 'second', 'third', 'fourth', 'fifth']
	popInfo = ['pop', 'electrode', 'lfpValue']

	top5pops = {}
	bottom5pops = {}


	for i in range(len(popRank)):
		top5pops[popRank[i]] = {}
		bottom5pops[popRank[i]] = {}
		for infoType in popInfo:
			if infoType == 'pop':
				top5pops[popRank[i]][infoType] = maxPopsDict[maxValuesDict_sorted[i][0]]
				bottom5pops[popRank[i]][infoType] = minPopsDict[minValuesDict_sorted[i][0]]
			elif infoType == 'electrode':
				top5pops[popRank[i]][infoType] = maxValuesDict_sorted[i][0]
				bottom5pops[popRank[i]][infoType] = minValuesDict_sorted[i][0]
			elif infoType == 'lfpValue':
				top5pops[popRank[i]][infoType] = maxValuesDict_sorted[i][1]
				bottom5pops[popRank[i]][infoType] = minValuesDict_sorted[i][1]

	return top5pops, bottom5pops

## Spike Activity: data and plotting ## 
def getSpikeData(dataFile, pop, graphType, timeRange): 
	### dataFile: path to .pkl data file to load 
	### pop: list or str --> which pop to include 
	### graphType: str --> either 'hist' or 'spect'
	### timeRange: list --> e.g. [start, stop]

	# Load data file
	sim.load(dataFile, instantiate=False)

	# Pops
	if type(pop) is str:
		popList = [pop]
	elif type(pop) is list:
		popList = pop

	# Set up which kind of data -- i.e. spectrogram or histogram 
	if graphType is 'spect':
		spikeDict = sim.analysis.getRateSpectrogramData(include=popList, timeRange=timeRange)
	elif graphType is 'hist':
		spikeDict = sim.analysis.getSpikeHistData(include=popList, timeRange=timeRange, binSize=5, graphType='bar', measure='rate')

	return spikeDict 
def plotCombinedSpike(spectDict, histDict, timeRange, pop, colorDict, figSize=(10,7), colorMap='jet', maxFreq=None, vmaxContrast=None, savePath=None, saveFig=True):
	### spectDict: dict --> can be gotten with getSpikeData(graphType='spect')
	### histDict: dict  --> can be gotten with getSpikeData(graphType='hist')
	### timeRange: list --> e.g. [start, stop]
	### pop: str or list of length 1 --> population to include 
	### colorDict: dict --> dict that corresponds pops to colors 
	### figSize: tuple 	--> DEFAULT: (10,7)
	### colorMap: str 	--> DEFAULT: 'jet' 	--> cmap for ax.imshow lines --> Options are currently 'jet' or 'viridis' 
	### maxFreq: int --> whole number that determines the maximum frequency plotted on the spectrogram 
			### --> NOTE --> ### NOT IMPLEMENTED YET !! minFreq: int --> whole number that determines the minimum frequency plotted on the spectrogram
	### vmaxContrast: float or int --> Denominator This will help with color contrast if desired!!!, e.g. 1.5 or 3
	### savePath: str   --> Path to directory where fig should be saved; DEFAULT: '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/popContribFigs/'
	### saveFig: bool 	--> DEFAULT: True
 
	# Get relevant pop
	if type(pop) is str:
		popToPlot = pop
	elif type(pop) is list:
		popToPlot = pop[0]

	# Create figure 
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figSize)

	# Set font sizes
	labelFontSize = 12
	titleFontSize = 20


	### SPECTROGRAM -- for top panel!!
	allSignal = spectDict['allSignal']
	allFreqs = spectDict['allFreqs']

	# Set frequencies to be plotted 
	if maxFreq is None:
		maxFreq = np.amax(allFreqs[0])
		imshowSignal = allSignal[0]
	else:
		if type(maxFreq) is not int:
			maxFreq = round(maxFreq)
		imshowSignal = allSignal[0][:maxFreq]

	# Set color contrast parameters 
	if vmaxContrast is None:
		vmin = None
		vmax = None
	else:
		vmin = np.amin(imshowSignal)
		vmax = np.amax(imshowSignal) / vmaxContrast

	# Plot spectrogram 
	ax1 = plt.subplot(211)
	img = ax1.imshow(imshowSignal, extent=(np.amin(timeRange), np.amax(timeRange), np.amin(allFreqs[0]), maxFreq), origin='lower', 
			interpolation='None', aspect='auto', cmap=plt.get_cmap(colorMap), vmin=vmin, vmax=vmax)
	divider1 = make_axes_locatable(ax1)
	cax1 = divider1.append_axes('right', size='3%', pad = 0.2)
	fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)		## fmt lines are for colorbar to be in scientific notation
	fmt.set_powerlimits((0,0))
	plt.colorbar(img, cax = cax1, orientation='vertical', label='Power', format=fmt)
	ax1.set_title('Spike Rate Spectrogram for ' + popToPlot, fontsize=titleFontSize)
	ax1.set_ylabel('Frequency (Hz)', fontsize=labelFontSize)
	ax1.set_xlim(left=timeRange[0], right=timeRange[1])


	### HISTOGRAM -- for bottom panel!! 
	histoT = histDict['histoT']
	histoCount = histDict['histoData']

	ax2 = plt.subplot(212)
	ax2.bar(histoT, histoCount[0], width = 5, color=colorDict[popToPlot], fill=True)
	divider2 = make_axes_locatable(ax2)
	cax2 = divider2.append_axes('right', size='3%', pad = 0.2)
	cax2.axis('off')
	ax2.set_title('Spike Rate Histogram for ' + popToPlot, fontsize=titleFontSize)
	ax2.set_xlabel('Time (ms)', fontsize=labelFontSize)
	ax2.set_ylabel('Rate (Hz)', fontsize=labelFontSize) # CLARIFY Y AXIS
	ax2.set_xlim(left=timeRange[0], right=timeRange[1])
	plt.show()

	plt.tight_layout()

	if saveFig:
		if savePath is None:
			prePath = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/popContribFigs/' 	# popContribFigs_cmapJet/'
		else:
			prePath = savePath
		fileName = pop + '_combinedSpike.png'
		pathToFile = prePath + fileName
		plt.savefig(pathToFile, dpi=300)

## LFP: data and plotting ## 
def getLFPDataDict(dataFile, pop, plotType, timeRange, electrode):
	### dataFile: str 			--> path to .pkl data file to load for analysis 
	### pop: str or list  		--> cell population to get the LFP data for 
	### plotType: str or list 	--> 'spectrogram' or 'timeSeries'
	### timeRange: list  		-->  e.g. [start, stop]
	### electrode: list or int or str designating the electrode of interest --> e.g. 10, [10], 'avg'
		### NOT IN USE CURRENTLY: filtFreq: list --> DEFAULT: None; frequencies to bandpass filter lfp data  <-- supposed to come from def getBandpassRange() 

	# Load data file 
	sim.load(dataFile, instantiate=False)

	# Pops
	if type(pop) is str:
		popList = [pop]
	elif type(pop) is list:
		popList = pop

	# Electrodes
	if type(electrode) is not list: 
		electrodeList = [electrode]
	else:
		electrodeList = electrode


	# Set up which kind of data -- i.e. timeSeries or spectrogram 
	if type(plotType) is str:
		plots = [plotType]
	elif type(plotType) is list:
		plots = plotType

	lfpOutput = sim.analysis.getLFPData(pop=popList, timeRange=timeRange, electrodes=electrodeList, plots=plots) # filtFreq=filtFreq (see above; in args)

	return lfpOutput
def plotCombinedLFP(spectDict, timeSeriesDict, timeRange, pop, colorDict, figSize=(10,7), colorMap='jet', maxFreq=None, vmaxContrast=None, titleElectrode=None, savePath=None, saveFig=True): # electrode='avg',
	### spectDict: dict with spectrogram data
	### timeSeriesDict: dict with timeSeries data
	### timeRange: list 	--> [start, stop]
	### pop: list or str 	--> relevant population to plot data for 
	### colorDict: dict 	--> corresponds pop to color 
	### figSize: tuple 		--> DEFAULT: (10,7)
	### colorMap: str 		--> DEFAULT: 'jet' 	--> cmap for ax.imshow lines --> Options are currently 'jet' or 'viridis' 
	### maxFreq: int 		--> whole number that determines the maximum frequency plotted on the spectrogram 
			### --> NOTE --> ### NOT IMPLEMENTED YET !! minFreq: int --> whole number that determines the minimum frequency plotted on the spectrogram
	### vmaxContrast: float or int 			--> Denominator This will help with color contrast if desired!!!, e.g. 1.5 or 3
	### titleElectrode: str or (1-element) list	-->  FOR USE IN PLOT TITLES !! --> This is for the electrode that will appear in the title 
	### savePath: str 	  	--> Path to directory where fig should be saved; DEFAULT: '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/popContribFigs/'
	### saveFig: bool 		--> DEFAULT: True 


	# Get relevant pop
	if type(pop) is str:
		popToPlot = pop
	elif type(pop) is list:
		popToPlot = pop[0]


	## Create figure 
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figSize)

	## Set font size 
	labelFontSize = 15 		# NOTE: in plotCombinedSpike, labelFontSize = 12
	titleFontSize = 20

	## Set up titles for spectrogram and timeSeries plots:
	spectTitlePreamble = 'LFP Spectrogram for ' + popToPlot
	timeSeriesTitlePreamble = 'LFP Signal for ' + popToPlot 

	if titleElectrode is not None:
		if type(titleElectrode) is list:
			titleElectrode = titleElectrode[0]
		if titleElectrode == 'avg':
			elecTitleSubset = ', averaged over all electrodes'
		else:
			elecTitleSubset = ', electrode ' + str(titleElectrode)
		spectTitle = spectTitlePreamble + elecTitleSubset
		timeSeriesTitle = timeSeriesTitlePreamble + elecTitleSubset
		figFilename = pop + '_combinedLFP_elec_' + str(titleElectrode) + '.png'
	else:
		spectTitle = spectTitlePreamble
		timeSeriesTitle = timeSeriesTitlePreamble
		figFilename = pop + '_combinedLFP.png'



	##### SPECTROGRAM  --> TOP PANEL !! #####
	spec = spectDict['spec']

	S = spec[0].TFR
	F = spec[0].f
	T = timeRange

	## Set up vmin / vmax color contrasts 
	vmin = np.array([s.TFR for s in spec]).min()
	# print('vmin: ' + str(vmin)) ### COLOR MAP CONTRAST TESTING LINES 
	if vmaxContrast is None:
		vmax = np.array([s.TFR for s in spec]).max()
	else:
		preVmax = np.array([s.TFR for s in spec]).max()
		# print('original vmax: ' + str(preVmax))		### COLOR MAP CONTRAST TESTING LINES 
		vmax = preVmax / vmaxContrast 
		# print('new vmax: ' + str(vmax)) 				### COLOR MAP CONTRAST TESTING LINES 
	vc = [vmin, vmax]

	## Plot Spectrogram 
	ax1 = plt.subplot(2, 1, 1)
	img = ax1.imshow(S, extent=(np.amin(T), np.amax(T), np.amin(F), np.amax(F)), origin='lower', interpolation='None', aspect='auto', 
		vmin=vc[0], vmax=vc[1], cmap=plt.get_cmap(colorMap))
	divider1 = make_axes_locatable(ax1)
	cax1 = divider1.append_axes('right', size='3%', pad=0.2)
	fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)		## fmt lines are for colorbar scientific notation
	fmt.set_powerlimits((0,0))
	plt.colorbar(img, cax = cax1, orientation='vertical', label='Power', format=fmt)
	ax1.set_title(spectTitle, fontsize=titleFontSize)
	ax1.set_ylabel('Frequency (Hz)', fontsize=labelFontSize)
	ax1.set_xlim(left=timeRange[0], right=timeRange[1])
	if maxFreq is not None:
		ax1.set_ylim(1, maxFreq) 	## TO DO: turn '1' into minFreq



	##### TIME SERIES  --> BOTTOM PANEL !! #####
	t = timeSeriesDict['t']
	lfpPlot = timeSeriesDict['lfpPlot']

	lw = 1.0
	ax2 = plt.subplot(2, 1, 2)
	divider2 = make_axes_locatable(ax2)
	cax2 = divider2.append_axes('right', size='3%', pad=0.2)
	cax2.axis('off')
	ax2.plot(t[0:len(lfpPlot)], lfpPlot, color=colorDict[popToPlot], linewidth=lw)
	ax2.set_title(timeSeriesTitle, fontsize=titleFontSize)
	ax2.set_xlabel('Time (ms)', fontsize=labelFontSize)
	ax2.set_xlim(left=timeRange[0], right=timeRange[1])
	ax2.set_ylabel('LFP Amplitudes (mV)', fontsize=labelFontSize)

	plt.tight_layout()
	plt.show()

	## Save figure 
	if saveFig:
		if savePath is None:
			prePath = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/popContribFigs/' 		# popContribFigs_cmapJet/'
		else:
			prePath = savePath
		pathToFile = prePath + figFilename
		plt.savefig(pathToFile, dpi=300)

## PSD: Get most powerful frequency from LFP data w/ option to plot the PSD ## 
def getPSDinfo(dataFile, pop, timeRange, electrode, plotPSD=False):
	### dataFile: str 			--> path to .pkl data file to load for analysis 
	### pop: str or list  		--> cell population to get the LFP data for
	### timeRange: list  		-->  e.g. [start, stop]
	### electrode: list or int or str designating the electrode of interest --> e.g. 10, [10], 'avg'
	### plotPSD: bool 			--> Determines whether or not to plot the PSD signals 	--> DEFAULT: False


	# Load data file 
	sim.load(dataFile, instantiate=False)

	# Get LFP data 				--> ### NOTE: make sure electrode list / int is fixed 
	outputData = sim.analysis.getLFPData(pop=pop, timeRange=timeRange, electrodes=electrode, plots=['PSD'])

	# Get signal & frequency data
	signalList = outputData['allSignal']
	signal = signalList[0]
	freqsList = outputData['allFreqs']
	freqs = freqsList[0]

	# print(str(signal.shape))
	# print(str(freqs.shape))

	maxSignalIndex = np.where(signal==np.amax(signal))
	maxPowerFrequency = freqs[maxSignalIndex]
	print(pop + ' max power frequency in LFP signal at electrode ' + str(electrode[0]) + ': ' + str(maxPowerFrequency))

	# Create PSD plots, if specified 
	if plotPSD:
		sim.analysis.plotLFP(pop=pop, timeRange=timeRange, electrodes=electrode, plots=['PSD'])

	return maxPowerFrequency


##########################
#### USEFUL VARIABLES ####
##########################
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
ECortPops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'CT5B', 'PT5B', 'IT6', 'CT6']
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

#########
delta = 0
beta = 	0
alpha = 0
theta = 1
# gamma = 0 

if delta:
	timeRange, dataFile = getWaveletInfo('delta', based)
	wavelet='delta' ### MOVE THESE EVENTUALLY -- BEING USED FOR peakTitle
	ylim = [1, 40]
	maxFreq = ylim[1]  ## maxFreq is for use in plotCombinedLFP, for the spectrogram plot 
elif beta:
	timeRange, dataFile = getWaveletInfo('beta', based)
	wavelet='beta'
	maxFreq=None
elif alpha:
	timeRange, dataFile = getWaveletInfo('alpha', based) ## recall timeRange issue (see nb)
	wavelet='alpha'
	maxFreq=None
elif theta:
	timeRange, dataFile = getWaveletInfo('theta', based)
	wavelet='theta'
	maxFreq=None
elif gamma:
	print('Cannot analyze gamma wavelet at this time')



########################
####### PLOTTING #######
########################

#### EVALUATING POPULATIONS TO CHOOSE #### 
evalPopsBool = 1

if evalPopsBool:
	print('timeRange: ' + str(timeRange))
	print('dataFile: ' + str(dataFile))
	dfPeak, dfAvg = getDataFrames(dataFile=dataFile, timeRange=timeRange)			# dfPeak, dfAvg, peakValues, avgValues, lfpPopData = getDataFrames(dataFile=dataFile, timeRange=timeRange, verbose=1)

	#### PEAK LFP Amplitudes ####
	top5popsPeak, bottom5popsPeak = evalPops(dfPeak) 
	MaxPeakLists = getPopElectrodeLists(top5popsPeak)
	MinPeakLists = getPopElectrodeLists(bottom5popsPeak)

	# peakTitle = 'Peak LFP Amplitudes of ' + wavelet + ' Wavelet'
	# peakPlot = plotDataFrames(dfPeak, pops=ECortPops, title=peakTitle)
	# plt.show(peakPlot)


	#### AVP LFP Amplitudes ####
	top5popsAvg, bottom5popsAvg = evalPops(dfAvg)
	MaxAvgLists = getPopElectrodeLists(top5popsAvg)
	MinAvgLists = getPopElectrodeLists(bottom5popsAvg)

	# avgTitle = 'Avg LFP Amplitudes of ' + wavelet + ' Wavelet'   # 'Avg LFP Amplitudes of Theta Wavelet' 
	# avgPlot = plotDataFrames(dfAvg, pops=ECortPops, title=avgTitle)
	# plt.show(avgPlot)


###################################
###### COMBINED LFP PLOTTING ######
###################################

plotLFPCombinedData = 0

# includePops = ['CT5B']#['IT3', 'IT5A', 'PT5B']	# placeholder for now <-- will ideally come out of the function above once the pop LFP netpyne issues get resolved! 
includePops = includePopsMaxPeak.copy()


if plotLFPCombinedData:
	for i in range(len(includePops)):
		pop = includePops[i]
		electrode = [electrodesMaxPeak[i]]

		print('Plotting LFP spectrogram and timeSeries for ' + pop + ' at electrode ' + str(electrode))

		## Get dictionaries with LFP data for spectrogram and timeSeries plotting  
		LFPSpectOutput = getLFPDataDict(dataFile, pop=pop, timeRange=timeRange, plotType=['spectrogram'], electrode=electrode) 
		LFPtimeSeriesOutput = getLFPDataDict(dataFile, pop=pop, timeRange=timeRange, plotType=['timeSeries'], electrode=electrode) #filtFreq=filtFreq, 


		plotCombinedLFP(spectDict=LFPSpectOutput, timeSeriesDict=LFPtimeSeriesOutput, timeRange=timeRange, pop=pop, colorDict=colorDict, maxFreq=maxFreq, 
			figSize=(10,7), titleElectrode=electrode, saveFig=0)

		### Get the strongest frequency in the LFP signal ### 
		# maxPowerFrequencyGETLFP = getPSDinfo(dataFile=dataFile, pop=pop, timeRange=timeRange, electrode=electrodes, plotPSD=1)



# if plotLFPCombinedData:
# 	for pop in includePops:
# 		print('Plotting LFP spectrogram and timeSeries for ' + pop)

# 		# ## Get electrodes associated with each pop   #### <-- THESE SHOULD BE RETURNED AUTOMATICALLY FROM EVAL POPS FX
# 		# if pop == 'IT3':  ## COULD ADD THESE INTO DICT SOMEWHERE ELSE FOR ACCESS!!! 
# 		# 	electrodes = [1] 
# 		# elif pop == 'IT5A':
# 		# 	electrodes = [10]
# 		# elif pop == 'PT5B':
# 		# 	electrodes = [11]
# 		# elif pop == 'IT5B':
# 		# 	electrodes = [11]
# 		# elif pop == 'CT5B':
# 		# 	electrodes = [11]
# 		# else:
# 		# 	electrodes = ['avg']

# 		## Get dictionaries with LFP data for spectrogram and timeSeries plotting  
# 		LFPSpectOutput = getLFPDataDict(dataFile, pop=pop, timeRange=timeRange, plotType=['spectrogram'], electrode=electrodes) 
# 		LFPtimeSeriesOutput = getLFPDataDict(dataFile, pop=pop, timeRange=timeRange, plotType=['timeSeries'], electrode=electrodes) #filtFreq=filtFreq, 


# 		plotCombinedLFP(spectDict=LFPSpectOutput, timeSeriesDict=LFPtimeSeriesOutput, timeRange=timeRange, pop=pop, colorDict=colorDict, maxFreq=maxFreq, 
# 			figSize=(10,7), titleElectrode=electrodes, saveFig=1)

# 		### Get the strongest frequency in the LFP signal ### 
# 		# maxPowerFrequencyGETLFP = getPSDinfo(dataFile=dataFile, pop=pop, timeRange=timeRange, electrode=electrodes, plotPSD=1)

## TO DO: 
## (1) [IN PROGRESS] Filter the timeRanged lfp data to the wavelet frequency band
## (2) Compare the change in lfp amplitude from "baseline"  (e.g. some time window before the wavelet and then during the wavelet event) 





##########################################
###### COMBINED SPIKE DATA PLOTTING ######
##########################################

plotSpikeData = 0

# includePops = includePopsMaxPeak.copy()		# ['PT5B']	#['IT3', 'IT5A', 'PT5B']	# placeholder for now <-- will ideally come out of the function above once the pop LFP netpyne issues get resolved! 

if plotSpikeData:
	for pop in includePops:
		print('Plotting spike data for ' + pop)

		## Get dictionaries with spiking data for spectrogram and histogram plotting 
		spikeSpectDict = getSpikeData(dataFile, graphType='spect', pop=pop, timeRange=timeRange)
		histDict = getSpikeData(dataFile, graphType='hist', pop=pop, timeRange=timeRange)

		## Then call plotting function 
		plotCombinedSpike(spectDict=spikeSpectDict, histDict=histDict, timeRange=timeRange, colorDict=colorDict,
		pop=pop, figSize=(10,7), colorMap='jet', vmaxContrast=None, maxFreq=None, saveFig=1)


 # ---> ## TO DO: Smooth or mess with bin size to smooth out spectrogram for spiking data






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







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
from netpyne.analysis import csd
from numbers import Number
import seaborn as sns 
import pandas as pd 
import pickle
import morlet
from morlet import MorletSpec, index2ms
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
def getPSDinfo(dataFile, pop, timeRange, electrode, lfpData=None, plotPSD=False):
	##### SHOULD TURN THIS INTO PLOT PSD 
	### dataFile: str 			--> path to .pkl data file to load for analysis 
	### pop: str or list  		--> cell population to get the LFP data for
	### timeRange: list  		-->  e.g. [start, stop]
	### electrode: list or int or str designating the electrode of interest --> e.g. 10, [10], 'avg'
	### lfpData: input LFP data to use instead of loading sim.load(dataFile)
	### plotPSD: bool 			--> Determines whether or not to plot the PSD signals 	--> DEFAULT: False

	# Load data file 
	sim.load(dataFile, instantiate=False)

	if lfpData is None:
		# Get LFP data 				--> ### NOTE: make sure electrode list / int is fixed 
		outputData = getLFPData(pop=pop, timeRange=timeRange, electrodes=electrode, plots=['PSD'])  # sim.analysis.getLFPData

	elif lfpData is not None:  ### THIS IS FOR SUMMED LFP DATA!!! 
		outputData = getLFPData(inputLFP=lfpData, timeRange=None, electrodes=None, plots=['PSD'])

	# Get signal & frequency data
	signalList = outputData['allSignal']
	signal = signalList[0]
	freqsList = outputData['allFreqs']
	freqs = freqsList[0]

	# print(str(signal.shape))
	# print(str(freqs.shape))

	maxSignalIndex = np.where(signal==np.amax(signal))
	maxPowerFrequency = freqs[maxSignalIndex]
	if electrode is None:
		print('max power frequency in LFP signal: ' + str(maxPowerFrequency))
	else:
		print(pop + ' max power frequency in LFP signal at electrode ' + str(electrode[0]) + ': ' + str(maxPowerFrequency))

	# Create PSD plots, if specified 
	if plotPSD:
		# plotLFP(pop=pop, timeRange=timeRange, electrodes=electrode, plots=['PSD']) # sim.analysis.plotLFP


		# freqs = allFreqs[i]
		# signal = allSignal[i]
		maxFreq=100
		lineWidth=1.0
		color='black'
		plt.figure(figsize=(10,7))
		plt.plot(freqs[freqs<maxFreq], signal[freqs<maxFreq], linewidth=lineWidth, color=color) #, label='Electrode %s'%(str(elec)))
		## max freq testing lines !! ##
		# print('type(freqs): ' + str(type(freqs)))
		# print('max freq: ' + str(np.amax(freqs)))
		# print('type(signal): ' + str(type(signal)))
		# print('max signal: ' + str(np.amax(signal)))
		# print('signal[0]: ' + str(signal[0]))
		# ###
		plt.xlim([0, maxFreq])

		# plt.ylabel(ylabel, fontsize=fontSize)

		# format plot
		plt.xticks(np.arange(0, maxFreq, step=5))
		fontSize=12
		plt.xlabel('Frequency (Hz)', fontsize=fontSize)
		plt.tight_layout()
		plt.suptitle('LFP Power Spectral Density', fontsize=fontSize, fontweight='bold') # add yaxis in opposite side
		plt.show()

	return maxPowerFrequency
def evalPopsOLD(dataFrame):
	###### --> Return top 5 & bottom 5 pop-electrode pairs ## 
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
def getPopElectrodeLists(evalPopsDict, verbose=0):
	## This function returns a list of lists, 2 elements long, with first element being a list of pops to include
	## 			and the second being the corresponding list of electrodes 
	### evalPopsDict: dict 	--> can be returned from def evalPops(), above 
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
def getCSDdataOLD(dataFile=None, outputType=['timeSeries', 'spectrogram'], timeRange=None, electrode=None, dt=None, sampr=None, pop=None, spacing_um=100, minFreq=1, maxFreq=100, stepFreq=1):
	#### Outputs a dict with CSD and other relevant data for plotting! 
	## dataFile: str     					--> .pkl file with recorded simulation 
	## outputType: list of strings 			--> options are 'timeSeries' +/- 'spectrogram' --> OR could be empty, if want csdData from all electrodes!! 
	## timeRange: list 						--> e.g. [start, stop]
	## electrode: list or None 				--> e.g. [8], None
	## dt: time step of the simulation 		--> (usually --> sim.cfg.recordStep)
	## sampr: sampling rate (Hz) 			--> (usually --> 1/(dt/1000))
	## pop: str or list 					--> e.g. 'ITS4' or ['ITS4']
	## spacing_um: int 						--> 100 by DEFAULT (spacing between electrodes in MICRONS)
	## minFreq: float / int 				--> DEFAULT: 1 Hz  
	## maxFreq: float / int 				--> DEFAULT: 100 Hz 
	## stepFreq: float / int 				--> DEFAULT: 1 Hz 
			## TO DO: 
			###  --> Should I also have an lfp_input option so that I can get the CSD data of summed LFP data...?
			###  --> Should I make it such that output can include both timeSeries and spectrogram so don't have to run this twice? test this!! 

	## load .pkl simulation file 
	if dataFile:
		sim.load(dataFile, instantiate=False)
	else:
		print('No dataFile; will use data from dataFile already loaded elsewhere!')

	## Determine timestep, sampling rate, and electrode spacing 
	dt = sim.cfg.recordStep
	sampr = 1.0/(dt/1000.0) 	# divide by 1000.0 to turn denominator from units of ms to s
	spacing_um = spacing_um		# 100um by default # 

	## Get LFP data   # ----> NOTE: SHOULD I MAKE AN LFP INPUT OPTION?????? FOR SUMMED LFP DATA??? (also noted above in arg descriptions)
	if pop is None:
		lfpData = sim.allSimData['LFP']
	else:
		if type(pop) is list:
			pop = pop[0]
		lfpData = sim.allSimData['LFPPops'][pop]


	## Step 1 in segmenting CSD data --> ALL electrodes, ALL timepoints
	csdData_allElecs_allTime = csd.getCSD(LFP_input_data=lfpData, dt=dt, sampr=sampr, spacing_um=spacing_um, vaknin=True)


	## Step 2 in segmenting CSD data --> ALL electrodes, SPECIFIED (OR ALL) timepoints
	if timeRange is None:
		timeRange = [0, sim.cfg.duration]			# this is for use later, in the outputType if statements below 
		csdData_allElecs = csdData_allElecs_allTime
	else:
		csdData_allElecs = csdData_allElecs_allTime[:,int(timeRange[0]/dt):int(timeRange[1]/dt)] ## NOTE: this assumes timeRange is in ms!! 


	## Step 3 in segmenting CSD data --> SPECIFIED (OR ALL) electrode(s), SPECIFIED (OR ALL) timepoints  
	if electrode is None:
		csdData = csdData_allElecs
		print('Outputting CSD data for ALL electrodes!')
	else:
		# electrode = list(electrode)  # if electrodes is int, this will turn it into a list; if it's a list, won't change anything. 
		if len(electrode) > 1:
			## NOTE: at some point, change this so the correct electrode-specific CSD data is provided!!! 
			print('More than one electrode listed!! --> outputData[\'csd\'] will contain CSD data from ALL electrodes!')
			csdData = csdData_allElecs
		elif len(electrode) == 1:
			elec = electrode[0]
			if elec == 'avg':
				csdData = np.mean(csdData_allElecs, axis=0)
			elif isinstance(elec, Number):
				csdData = csdData_allElecs[elec, :]


	outputData = {'csd': csdData}


	# timeSeries --------------------------------------------
	if 'timeSeries' in outputType:
		print('Returning timeSeries data')

		## original way of determining t 
		t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)   # time array for x-axis 
		## more like load.py
		# t = 

		outputData.update({'t': t})


	# spectrogram -------------------------------------------
	if 'spectrogram' in outputType:
		print('Returning spectrogram data')

		spec = []
		freqList = None

		## Determine electrode(s) to loop over: 
		if electrode is None: 
			print('No electrode specified; returning spectrogram data for ALL electrodes')
			electrode = []
			electrode.extend(list(range(int(sim.net.recXElectrode.nsites))))

		print('Channels considered for spectrogram data: ' + str(electrode))


		## Spectrogram Data Calculations! 
		if len(electrode) > 1:
			for i, elec in enumerate(electrode):
				csdDataSpect_allElecs = np.transpose(csdData_allElecs)  # Transposing this data may not be necessary!!! 
				csdDataSpect = csdDataSpect_allElecs[:, elec]
				fs = int(1000.0 / sim.cfg.recordStep)
				t_spec = np.linspace(0, morlet.index2ms(len(csdDataSpect), fs), len(csdDataSpect)) 
				spec.append(MorletSpec(csdDataSpect, fs, freqmin=minFreq, freqmax=maxFreq, freqstep=stepFreq, lfreq=freqList))

		elif len(electrode) == 1: 	# use csdData, as determined above the timeSeries and spectrogram 'if' statements (already has correct electrode-specific CSD data!)
			fs = int(1000.0 / sim.cfg.recordStep)
			t_spec = np.linspace(0, morlet.index2ms(len(csdData), fs), len(csdData)) # Seems this is only used for the fft circumstance...? 
			spec.append(MorletSpec(csdData, fs, freqmin=minFreq, freqmax=maxFreq, freqstep=stepFreq, lfreq=freqList))


		## Get frequency list 
		f = freqList if freqList is not None else np.array(range(minFreq, maxFreq+1, stepFreq))   # only used as output for user

		## vmin, vmax --> vc = [vmin, vmax]
		vmin = np.array([s.TFR for s in spec]).min()
		vmax = np.array([s.TFR for s in spec]).max()
		vc = [vmin, vmax]

		## T (timeRange)
		T = timeRange 

		## F, S
		for i, elec in enumerate(electrode):   # works for electrode of length 1 or greater! No need for if statement regarding length. 
			F = spec[i].f
			# if normSpec:  #### THIS IS FALSE BY DEFAULT, SO COMMENTING IT OUT HERE 
				# spec[i].TFR = spec[i].TFR / vmax
				# S = spec[i].TFR
				# vc = [0, 1]
			S = spec[i].TFR


		outputData.update({'T': T, 'F': F, 'S': S, 'vc': vc})  ### All the things necessary for plotting!! 
		outputData.update({'spec': spec, 't': t_spec*1000.0, 'freqs': f[f<=maxFreq]}) 
			### This is at the end of the plotLFP and getLFPdata functions, but not sure what purpose it will serve; keep this in for now!! 
			### ^^ ah, well, could derive F and S from spec. not sure I need 't' or 'freqs' though? hmmm. 

	return outputData
######################################################################

######################################################################
##### FUNCTIONS IN PROGRESS ##### 
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
def evalWaveletsByBand(based, dfPklFile):
	## NOT IN USE RIGHT NOW --> ## freqBand: str 			--> e.g. 'alpha', 'beta' ,'theta', 'delta', 'gamma'
	## NOT IN USE RIGHT NOW --> dlmsPklFile: .pkl file 	--> from dlms.pkl file, saved from load.py 
	## based: str 				--> Beginning of path to the .pkl files 
	## dfPklFile: .pkl file 	--> from df.pkl file, saved from load.py 

	print('Evaluating all oscillation events in a given frequency band')

	based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/wavelets/' ### COULD make this an arg!! 
	subjectDir = dfPklFile.split('_df.pkl')[0]

	# Load df file 
	dfFullPath = based + subjectDir + '/' + dfPklFile
	df = pd.read_pickle(dfFullPath)

	# # Load dlms file
	# dlmsFullPath = based + subjectDir + '/' + dlmsPklFile
	# dlmsFile = open(dlmsFullPath, 'rb')
	# dlms = pickle.load(dlmsFile)
	# dlmsFile.close()


	return df
## plotting functions
def plotTimeSeries(timeSeriesDict, dataType=['LFP', 'CSD']):
	### dataType: list of str 		--> Indicates which dataType will be input / plotted
	### timeSeriesDict: dict 		--> Output of ____ (which exact funtions)
	print('Plot time series for LFP or CSD data')
def plotSpectrogram(spectDict, dataType=['LFP', 'CSD', 'Spike']):
	### dataType: list of str 		--> Indicates which dataType will be input / plotted
	### spectDict: dict 			--> output of ____ which functions?
	print('Plot spectrogram for LFP, CSD, or Spiking data')
def plotHistogram(histDict):
	### histDict: dict 			--> output of ___which function exactly? 
	print('Plot histogram of spiking data')
######################################################################




#################################################################################################################
######### NetPyNE Functions that have been modified for use here!! ##############################################
#################################################################################################################

### SPIKE DATA ### 
def getRateSpectrogramData(include=['allCells', 'eachPop'], timeRange=None, binSize=5, minFreq=1, maxFreq=100, stepFreq=1, NFFT=256, noverlap=128, smooth=0, transformMethod = 'morlet', norm=False):
	"""
	include : list
		<Short description of include>
		**Default:** ``['allCells', 'eachPop']``
		**Options:** ``<option>`` <description of option>

	timeRange : <``None``?>
		<Short description of timeRange>
		**Default:** ``None``
		**Options:** ``<option>`` <description of option>

	binSize : int
		<Short description of binSize>
		**Default:** ``5``
		**Options:** ``<option>`` <description of option>

	minFreq : int
		<Short description of minFreq>
		**Default:** ``1``
		**Options:** ``<option>`` <description of option>

	maxFreq : int
		<Short description of maxFreq>
		**Default:** ``100``
		**Options:** ``<option>`` <description of option>

	stepFreq : int
		<Short description of stepFreq>
		**Default:** ``1``
		**Options:** ``<option>`` <description of option>

	NFFT : int
		<Short description of NFFT>
		**Default:** ``256``
		**Options:** ``<option>`` <description of option>

	noverlap : int
		<Short description of noverlap>
		**Default:** ``128``
		**Options:** ``<option>`` <description of option>

	smooth : int
		<Short description of smooth>
		**Default:** ``0``
		**Options:** ``<option>`` <description of option>

	transformMethod : str
		<Short description of transformMethod>
		**Default:** ``'morlet'``
		**Options:** ``<option>`` <description of option>

	norm : bool
		<Short description of norm>
		**Default:** ``False``
		**Options:** ``<option>`` <description of option> 
		"""
	print('Getting firing rate spectrogram data ...')

	# Replace 'eachPop' with list of pops
	if 'eachPop' in include:
		include.remove('eachPop')
		for pop in sim.net.allPops: include.append(pop)

	# time range
	if timeRange is None:
		timeRange = [0,sim.cfg.duration]

	histData = []

	allSignal, allFreqs = [], []

	# Plot separate line for each entry in include
	for iplot,subset in enumerate(include):
		from netpyne.analysis.utils import getCellsInclude
		cells, cellGids, netStimLabels = getCellsInclude([subset])
		numNetStims = 0

		# Select cells to include
		if len(cellGids) > 0:
			try:
				spkinds,spkts = list(zip(*[(spkgid,spkt) for spkgid,spkt in zip(sim.allSimData['spkid'],sim.allSimData['spkt']) if spkgid in cellGids]))
			except:
				spkinds,spkts = [],[]
		else:
			spkinds,spkts = [],[]


		# Add NetStim spikes
		spkts, spkinds = list(spkts), list(spkinds)
		numNetStims = 0
		if 'stims' in sim.allSimData:
			for netStimLabel in netStimLabels:
				netStimSpks = [spk for cell,stims in sim.allSimData['stims'].items() \
					for stimLabel,stimSpks in stims.items() for spk in stimSpks if stimLabel == netStimLabel]
				if len(netStimSpks) > 0:
					lastInd = max(spkinds) if len(spkinds)>0 else 0
					spktsNew = netStimSpks
					spkindsNew = [lastInd+1+i for i in range(len(netStimSpks))]
					spkts.extend(spktsNew)
					spkinds.extend(spkindsNew)
					numNetStims += 1

		histo = np.histogram(spkts, bins = np.arange(timeRange[0], timeRange[1], binSize))
		histoT = histo[1][:-1]+binSize/2
		histoCount = histo[0]
		histoCount = histoCount * (1000.0 / binSize) / (len(cellGids)+numNetStims) # convert to rates

		histData.append(histoCount)

		# Morlet wavelet transform method
		if transformMethod == 'morlet':
			from morlet import MorletSpec, index2ms

			Fs = 1000.0 / binSize

			morletSpec = MorletSpec(histoCount, Fs, freqmin=minFreq, freqmax=maxFreq, freqstep=stepFreq)
			freqs = morletSpec.f
			spec = morletSpec.TFR
			ylabel = 'Power'
			allSignal.append(spec)
			allFreqs.append(freqs)

	# plotting
	T = timeRange

	# save figure data
	figData = {'histData': histData, 'histT': histoT, 'include': include, 'timeRange': timeRange, 'binSize': binSize}

	return {'allSignal': allSignal, 'allFreqs':allFreqs}
def getSpikeHistData(include=['eachPop', 'allCells'], timeRange=None, binSize=5, graphType='bar', measure='rate', norm=False, smooth=None, filtFreq=None, filtOrder=3, axis=True, **kwargs):
	"""
	include : list
		Populations and cells to include in the plot.
		**Default:**
		``['eachPop', 'allCells']`` plots histogram for each population and overall average
		**Options:**
		``['all']`` plots all cells and stimulations,
		``['allNetStims']`` plots just stimulations,
		``['popName1']`` plots a single population,
		``['popName1', 'popName2']`` plots multiple populations,
		``[120]`` plots a single cell,
		``[120, 130]`` plots multiple cells,
		``[('popName1', 56)]`` plots a cell from a specific population,
		``[('popName1', [0, 1]), ('popName2', [4, 5, 6])]``, plots cells from multiple populations

	timeRange : list [start, stop]
		Time range to plot.
		**Default:**
		``None`` plots entire time range
		**Options:** ``<option>`` <description of option>

	binSize : int
		Size of bin in ms to use for spike histogram.
		**Default:** ``5``
		**Options:** ``<option>`` <description of option>

	graphType : str
		Show histograms as line graphs or bar plots.
		**Default:** ``'bar'``
		**Options:** ``'line'``

	measure : str
		Whether to plot spike freguency (rate) or spike count.
		**Default:** ``'rate'``
		**Options:** ``'count'``

	norm : bool
		Whether to normalize the data or not.
		**Default:** ``False`` does not normalize the data
		**Options:** ``<option>`` <description of option>

	smooth : int
		Window width for smoothing.
		**Default:** ``None`` does not smooth the data
		**Options:** ``<option>`` <description of option>

	filtFreq : int or list
		Frequency for low-pass filter (int) or frequencies for bandpass filter in a list: [low, high]
		**Default:** ``None`` does not filter the data
		**Options:** ``<option>`` <description of option>

	filtOrder : int
		Order of the filter defined by `filtFreq`.
		**Default:** ``3``
		**Options:** ``<option>`` <description of option>

	axis : bool
		Whether to include a labeled axis on the figure.
		**Default:** ``True`` includes a labeled axis
		**Options:** ``False`` includes a scale bar

	kwargs : <type>
		<Short description of kwargs>
		**Default:** *required*
	"""

	# from .. import sim  ### <--- SHOULD ALREADY HAVE THIS

	print('Getting spike histogram data...')

	# Replace 'eachPop' with list of pops
	if 'eachPop' in include:
		include.remove('eachPop')
		for pop in sim.net.allPops: include.append(pop)


	# time range
	if timeRange is None:
		timeRange = [0, sim.cfg.duration]

	histoData = []

	# Plot separate line for each entry in include
	for iplot,subset in enumerate(include):
		from netpyne.analysis.utils import getCellsInclude
		if isinstance(subset, list):
			cells, cellGids, netStimLabels = getCellsInclude(subset)
		else:
			cells, cellGids, netStimLabels = getCellsInclude([subset])
		numNetStims = 0

		# Select cells to include
		if len(cellGids) > 0:
			try:
				spkinds,spkts = list(zip(*[(spkgid,spkt) for spkgid,spkt in zip(sim.allSimData['spkid'],sim.allSimData['spkt']) if spkgid in cellGids]))
			except:
				spkinds,spkts = [],[]
		else:
			spkinds,spkts = [],[]

		# Add NetStim spikes
		spkts, spkinds = list(spkts), list(spkinds)
		numNetStims = 0
		if 'stims' in sim.allSimData:
			for netStimLabel in netStimLabels:
				netStimSpks = [spk for cell,stims in sim.allSimData['stims'].items() \
				for stimLabel,stimSpks in stims.items() for spk in stimSpks if stimLabel == netStimLabel]
				if len(netStimSpks) > 0:
					lastInd = max(spkinds) if len(spkinds)>0 else 0
					spktsNew = netStimSpks
					spkindsNew = [lastInd+1+i for i in range(len(netStimSpks))]
					spkts.extend(spktsNew)
					spkinds.extend(spkindsNew)
					numNetStims += 1

		histo = np.histogram(spkts, bins = np.arange(timeRange[0], timeRange[1], binSize))
		histoT = histo[1][:-1]+binSize/2
		histoCount = histo[0]

		if measure == 'rate':
			histoCount = histoCount * (1000.0 / binSize) / (len(cellGids)+numNetStims) # convert to firing rate

		if filtFreq:
			from scipy import signal
			fs = 1000.0/binSize
			nyquist = fs/2.0
			if isinstance(filtFreq, list): # bandpass
				Wn = [filtFreq[0]/nyquist, filtFreq[1]/nyquist]
				b, a = signal.butter(filtOrder, Wn, btype='bandpass')
			elif isinstance(filtFreq, Number): # lowpass
				Wn = filtFreq/nyquist
				b, a = signal.butter(filtOrder, Wn)
			histoCount = signal.filtfilt(b, a, histoCount)

		if norm:
			histoCount /= max(histoCount)

		if smooth:
			histoCount = _smooth1d(histoCount, smooth)[:len(histoT)]  ## get smooth1d from netpyne.analysis.utils if necessary

		histoData.append(histoCount)

	# save figure data
	figData = {'histoData': histoData, 'histoT': histoT, 'include': include, 'timeRange': timeRange, 'binSize': binSize}

	return {'include': include, 'histoData': histoData, 'histoT': histoT, 'timeRange': timeRange}

### LFP ### 
def getLFPData(pop=None, timeRange=None, electrodes=['avg', 'all'], plots=['timeSeries', 'spectrogram'], inputLFP=None, NFFT=256, noverlap=128, nperseg=256, 
	minFreq=1, maxFreq=100, stepFreq=1, smooth=0, separation=1.0, logx=False, logy=False, normSignal=False, normSpec=False, filtFreq=False, filtOrder=3, detrend=False, 
	transformMethod='morlet'):
	"""
	Function for plotting local field potentials (LFP)

	Parameters

	pop: str (NOTE: for now) 
		Population to plot lfp data for (sim.allSimData['LFPPops'][pop] <-- requires LFP pop saving!)
		``None`` plots overall LFP data 

	timeRange : list [start, stop]
		Time range to plot.
		**Default:**
		``None`` plots entire time range

	electrodes : list
		List of electrodes to include; ``'avg'`` is the average of all electrodes; ``'all'`` is each electrode separately.
		**Default:** ``['avg', 'all']``

	plots : list
		List of plot types to show.
		**Default:** ``['timeSeries', 'spectrogram']`` <-- added 'PSD' (EYG, 2/10/2022) <-- 'PSD' taken out on 3/16/22

	NFFT : int (power of 2)
		Number of data points used in each block for the PSD and time-freq FFT.
		**Default:** ``256``

	noverlap : int (<nperseg)
		Number of points of overlap between segments for PSD and time-freq.
		**Default:** ``128``

	nperseg : int
		Length of each segment for time-freq.
		**Default:** ``256``

	minFreq : float
		Minimum frequency shown in plot for PSD and time-freq.
		**Default:** ``1``

	maxFreq : float
		Maximum frequency shown in plot for PSD and time-freq.
		**Default:** ``100``

	stepFreq : float
		Step frequency.
		**Default:** ``1``

	smooth : int
		Window size for smoothing LFP; no smoothing if ``0``
		**Default:** ``0``

	separation : float
		Separation factor between time-resolved LFP plots; multiplied by max LFP value.
		**Default:** ``1.0``

	logx : bool
		Whether to make x-axis logarithmic
		**Default:** ``False``
		**Options:** ``<option>`` <description of option>

	logy : bool
		Whether to make y-axis logarithmic
		**Default:** ``False``

	normSignal : bool
		Whether to normalize the signal.
		**Default:** ``False``

	normSpec : bool
		Needs documentation.
		**Default:** ``False``

	filtFreq : int or list
		Frequency for low-pass filter (int) or frequencies for bandpass filter in a list: [low, high]
		**Default:** ``False`` does not filter the data

	filtOrder : int
		Order of the filter defined by `filtFreq`.
		**Default:** ``3``

	detrend : bool
		Whether to detrend.
		**Default:** ``False``

	transformMethod : str
		Transform method.
		**Default:** ``'morlet'``
		**Options:** ``'fft'``

	Returns
	-------
	(dict)
		A tuple consisting of the Matplotlib figure handles and a dictionary containing the plot data
	"""

	# from .. import sim   ### <-- should already be there!!!! in imports!! 

	print('Getting LFP data ...')


	# time range
	if timeRange is None:
		timeRange = [0,sim.cfg.duration]

	# populations
	#### CLEAN THIS UP.... go through all the possibilities implied by these if statements and make sure all are accounted for! 
	if inputLFP is None: 
		if pop is None:
			if timeRange is None:
				lfp = np.array(sim.allSimData['LFP'])
			elif timeRange is not None: 
				lfp = np.array(sim.allSimData['LFP'])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]

		elif pop is not None:
			if type(pop) is str:
				popToPlot = pop
			elif type(pop) is list and len(pop)==1:
				popToPlot = pop[0]
			# elif type(pop) is list and len(pop) > 1:  #### USE THIS AS JUMPING OFF POINT TO EXPAND FOR LIST OF MULTIPLE POPS?? 
			lfp = np.array(sim.allSimData['LFPPops'][popToPlot])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]

	#### MAKE SURE THAT THIS ADDITION DOESN'T MAKE ANYTHING ELSE BREAK!! 
	elif inputLFP is not None:  ### DOING THIS FOR PSD FOR SUMMED LFP SIGNAL !!! 
		lfp = inputLFP 

		# if timeRange is None:
		#     lfp = inputLFP 
		# elif timeRange is not None:
		#     lfp = inputLFP[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:] ### hmm. 


	if filtFreq:
		from scipy import signal
		fs = 1000.0/sim.cfg.recordStep
		nyquist = fs/2.0
		if isinstance(filtFreq, list): # bandpass
			Wn = [filtFreq[0]/nyquist, filtFreq[1]/nyquist]
			b, a = signal.butter(filtOrder, Wn, btype='bandpass')
		elif isinstance(filtFreq, Number): # lowpass
			Wn = filtFreq/nyquist
			b, a = signal.butter(filtOrder, Wn)
		for i in range(lfp.shape[1]):
			lfp[:,i] = signal.filtfilt(b, a, lfp[:,i])

	if detrend:
		from scipy import signal
		for i in range(lfp.shape[1]):
			lfp[:,i] = signal.detrend(lfp[:,i])

	if normSignal:
		for i in range(lfp.shape[1]):
			offset = min(lfp[:,i])
			if offset <= 0:
				lfp[:,i] += abs(offset)
			lfp[:,i] /= max(lfp[:,i])

	# electrode selection
	print('electrodes: ' + str(electrodes))
	if electrodes is None:
		print('electrodes is None -- improve this')
	elif type(electrodes) is list:
		if 'all' in electrodes:
			electrodes.remove('all')
			electrodes.extend(list(range(int(sim.net.recXElectrode.nsites))))

	data = {'lfp': lfp}  # returned data


	# time series -----------------------------------------
	if 'timeSeries' in plots:
		ydisp = np.absolute(lfp).max() * separation
		offset = 1.0*ydisp
		t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)


		for i,elec in enumerate(electrodes):
			if elec == 'avg':
				lfpPlot = np.mean(lfp, axis=1)
				color = 'k'
				lw=1.0
			elif isinstance(elec, Number) and (inputLFP is not None or elec <= sim.net.recXElectrode.nsites):
				lfpPlot = lfp[:, elec]
				color = 'k' #colors[i%len(colors)]
				lw = 1.0

			if len(t) < len(lfpPlot):
				lfpPlot = lfpPlot[:len(t)]


		data['lfpPlot'] = lfpPlot
		data['ydisp'] =  ydisp
		data['t'] = t

	# Spectrogram ------------------------------
	if 'spectrogram' in plots:
		import matplotlib.cm as cm
		numCols = 1 #np.round(len(electrodes) / maxPlots) + 1

		# Morlet wavelet transform method
		if transformMethod == 'morlet':
			spec = []
			freqList = None
			if logy:
				freqList = np.logspace(np.log10(minFreq), np.log10(maxFreq), int((maxFreq-minFreq)/stepFreq))

			for i,elec in enumerate(electrodes):
				if elec == 'avg':
					lfpPlot = np.mean(lfp, axis=1)
				elif isinstance(elec, Number) and (inputLFP is not None or elec <= sim.net.recXElectrode.nsites):
					lfpPlot = lfp[:, elec]
				fs = int(1000.0 / sim.cfg.recordStep)
				t_spec = np.linspace(0, morlet.index2ms(len(lfpPlot), fs), len(lfpPlot))
				spec.append(MorletSpec(lfpPlot, fs, freqmin=minFreq, freqmax=maxFreq, freqstep=stepFreq, lfreq=freqList))

			f = freqList if freqList is not None else np.array(range(minFreq, maxFreq+1, stepFreq))   # only used as output for user

			vmin = np.array([s.TFR for s in spec]).min()
			vmax = np.array([s.TFR for s in spec]).max()

			for i,elec in enumerate(electrodes):
				T = timeRange
				F = spec[i].f
				if normSpec:
					spec[i].TFR = spec[i].TFR / vmax
					S = spec[i].TFR
					vc = [0, 1]
				else:
					S = spec[i].TFR
					vc = [vmin, vmax]


		# FFT transform method
		elif transformMethod == 'fft':

			from scipy import signal as spsig
			spec = []

			for i,elec in enumerate(electrodes):
				if elec == 'avg':
					lfpPlot = np.mean(lfp, axis=1)
				elif isinstance(elec, Number) and elec <= sim.net.recXElectrode.nsites:
					lfpPlot = lfp[:, elec]
				# creates spectrogram over a range of data
				# from: http://joelyancey.com/lfp-python-practice/
				fs = int(1000.0/sim.cfg.recordStep)
				f, t_spec, x_spec = spsig.spectrogram(lfpPlot, fs=fs, window='hanning',
				detrend=mlab.detrend_none, nperseg=nperseg, noverlap=noverlap, nfft=NFFT,  mode='psd')
				x_mesh, y_mesh = np.meshgrid(t_spec*1000.0, f[f<maxFreq])
				spec.append(10*np.log10(x_spec[f<maxFreq]))

			vmin = np.array(spec).min()
			vmax = np.array(spec).max()


	outputData = {'LFP': lfp, 'lfpPlot': lfpPlot, 'electrodes': electrodes, 'timeRange': timeRange}
	### Added lfpPlot to this, because that usually has the post-processed electrode-based lfp data

	if 'timeSeries' in plots:
		outputData.update({'t': t})

	if 'spectrogram' in plots:
		outputData.update({'spec': spec, 't': t_spec*1000.0, 'freqs': f[f<=maxFreq]})

	# if 'PSD' in plots:
	# 	outputData.update({'allFreqs': allFreqs, 'allSignal': allSignal})


	return outputData
def plotLFP(pop=None, timeRange=None, electrodes=['avg', 'all'], plots=['timeSeries', 'PSD', 'spectrogram', 'locations'], inputLFP=None, NFFT=256, noverlap=128, nperseg=256, minFreq=1, maxFreq=100, stepFreq=1, smooth=0, separation=1.0, includeAxon=True, logx=False, logy=False, normSignal=False, normPSD=False, normSpec=False, filtFreq=False, filtOrder=3, detrend=False, transformMethod='morlet', maxPlots=8, overlay=False, colors=None, figSize=(8, 8), fontSize=14, lineWidth=1.5, dpi=200, saveData=None, saveFig=None, showFig=True):
	"""
	Parameters

	pop: str (NOTE: for now) 
		Population to plot lfp data for (sim.allSimData['LFPPops'][pop] <-- requires LFP pop saving!)
		``None`` plots overall LFP data 

	timeRange : list [start, stop]
		Time range to plot.
		**Default:**
		``None`` plots entire time range

	electrodes : list
		List of electrodes to include; ``'avg'`` is the average of all electrodes; ``'all'`` is each electrode separately.
		**Default:** ``['avg', 'all']``

	plots : list
		List of plot types to show.
		**Default:** ``['timeSeries', 'PSD', 'spectrogram', 'locations']``

	NFFT : int (power of 2)
		Number of data points used in each block for the PSD and time-freq FFT.
		**Default:** ``256``

	noverlap : int (<nperseg)
		Number of points of overlap between segments for PSD and time-freq.
		**Default:** ``128``

	nperseg : int
		Length of each segment for time-freq.
		**Default:** ``256``

	minFreq : float
		Minimum frequency shown in plot for PSD and time-freq.
		**Default:** ``1``

	maxFreq : float
		Maximum frequency shown in plot for PSD and time-freq.
		**Default:** ``100``

	stepFreq : float
		Step frequency.
		**Default:** ``1``

	smooth : int
		Window size for smoothing LFP; no smoothing if ``0``
		**Default:** ``0``

	separation : float
		Separation factor between time-resolved LFP plots; multiplied by max LFP value.
		**Default:** ``1.0``

	includeAxon : bool
		Whether to show the axon in the location plot.
		**Default:** ``True``

	logx : bool
		Whether to make x-axis logarithmic
		**Default:** ``False``
		**Options:** ``<option>`` <description of option>

	logy : bool
		Whether to make y-axis logarithmic
		**Default:** ``False``

	normSignal : bool
		Whether to normalize the signal.
		**Default:** ``False``

	normPSD : bool
		Whether to normalize the power spectral density.
		**Default:** ``False``

	normSpec : bool
		Needs documentation.
		**Default:** ``False``

	filtFreq : int or list
		Frequency for low-pass filter (int) or frequencies for bandpass filter in a list: [low, high]
		**Default:** ``False`` does not filter the data

	filtOrder : int
		Order of the filter defined by `filtFreq`.
		**Default:** ``3``

	detrend : bool
		Whether to detrend.
		**Default:** ``False``

	transformMethod : str
		Transform method.
		**Default:** ``'morlet'``
		**Options:** ``'fft'``

	maxPlots : int
		Maximum number of subplots. Currently unused.
		**Default:** ``8``

	overlay : bool
		Whether to overlay plots or use subplots.
		**Default:** ``False`` overlays plots.

	colors : list
		List of normalized RGB colors to use.
		**Default:** ``None`` uses standard colors

	figSize : list [width, height]
		Size of figure in inches.
		**Default:** ``(10, 8)``

	fontSize : int
		Font size on figure.
		**Default:** ``14``

	lineWidth : int
		Line width.
		**Default:** ``1.5``

	dpi : int
		Resolution of figure in dots per inch.
		**Default:** ``100``

	saveData : bool or str
		Whether and where to save the data used to generate the plot.
		**Default:** ``False``
		**Options:** ``True`` autosaves the data,
		``'/path/filename.ext'`` saves to a custom path and filename, valid file extensions are ``'.pkl'`` and ``'.json'``

	saveFig : bool or str
		Whether and where to save the figure.
		**Default:** ``False``
		**Options:** ``True`` autosaves the figure,
		``'/path/filename.ext'`` saves to a custom path and filename, valid file extensions are ``'.png'``, ``'.jpg'``, ``'.eps'``, and ``'.tiff'``

	showFig : bool
		Shows the figure if ``True``.
		**Default:** ``True``

	Returns
	-------
	(figs, dict)
		A tuple consisting of the Matplotlib figure handles and a dictionary containing the plot data

	"""

	# from .. import sim  ### <-- should already be here 
	# from ..support.scalebar import add_scalebar
	import testScalebar

	print('Plotting LFP ...')

	if not colors: colors = colorList

	# set font size
	plt.rcParams.update({'font.size': fontSize})

	# time range
	if timeRange is None:
		timeRange = [0,sim.cfg.duration]

	# populations
	if pop is None:
		if inputLFP is not None:
			lfp = inputLFP #[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]
		else:
			lfp = np.array(sim.allSimData['LFP'])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]
	elif pop is not None:
		lfp = np.array(sim.allSimData['LFPPops'][pop])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]


	if filtFreq:
		from scipy import signal
		fs = 1000.0/sim.cfg.recordStep
		nyquist = fs/2.0
		if isinstance(filtFreq, list): # bandpass
			Wn = [filtFreq[0]/nyquist, filtFreq[1]/nyquist]
			b, a = signal.butter(filtOrder, Wn, btype='bandpass')
		elif isinstance(filtFreq, Number): # lowpass
			Wn = filtFreq/nyquist
			b, a = signal.butter(filtOrder, Wn)
		for i in range(lfp.shape[1]):
			lfp[:,i] = signal.filtfilt(b, a, lfp[:,i])

	if detrend:
		from scipy import signal
		for i in range(lfp.shape[1]):
			lfp[:,i] = signal.detrend(lfp[:,i])

	if normSignal:
		for i in range(lfp.shape[1]):
			offset = min(lfp[:,i])
			if offset <= 0:
				lfp[:,i] += abs(offset)
			lfp[:,i] /= max(lfp[:,i])

	# electrode selection
	if electrodes is None:
		print('electrodes is None: improve this -- ') ### FOR PSD PLOTTING / INFO FOR SUMMED LFP SIGNAL!!
	elif type(electrodes) is list:
		if 'all' in electrodes:
			electrodes.remove('all')
			electrodes.extend(list(range(int(sim.net.recXElectrode.nsites))))

	# plotting
	figs = []

	data = {'lfp': lfp}  # returned data


	# time series -----------------------------------------
	if 'timeSeries' in plots:
		ydisp = np.absolute(lfp).max() * separation
		offset = 1.0*ydisp
		t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)

		if figSize:
			figs.append(plt.figure(figsize=figSize))

		for i,elec in enumerate(electrodes):
			if elec == 'avg':
				lfpPlot = np.mean(lfp, axis=1)
				color = 'k'
				lw=1.0
			elif isinstance(elec, Number) and (inputLFP is not None or elec <= sim.net.recXElectrode.nsites):
				lfpPlot = lfp[:, elec]
				color = colors[i%len(colors)]
				lw = 1.0

			if len(t) < len(lfpPlot):
				lfpPlot = lfpPlot[:len(t)]

			plt.plot(t[0:len(lfpPlot)], -lfpPlot+(i*ydisp), color=color, linewidth=lw)
			if len(electrodes) > 1:
				plt.text(timeRange[0]-0.07*(timeRange[1]-timeRange[0]), (i*ydisp), elec, color=color, ha='center', va='top', fontsize=fontSize, fontweight='bold')

		ax = plt.gca()

		data['lfpPlot'] = lfpPlot
		data['ydisp'] =  ydisp
		data['t'] = t

		# format plot
		if len(electrodes) > 1:
			plt.text(timeRange[0]-0.14*(timeRange[1]-timeRange[0]), (len(electrodes)*ydisp)/2.0, 'LFP electrode', color='k', ha='left', va='bottom', fontSize=fontSize, rotation=90)
			plt.ylim(-offset, (len(electrodes))*ydisp)
		else:
			if pop is None:
				plt.suptitle('LFP Signal', fontSize=fontSize, fontweight='bold')
			elif pop is not None:
				timeSeriesTitle = 'LFP Signal of ' + pop + ' population'
				plt.suptitle(timeSeriesTitle, fontSize=fontSize, fontweight='bold')
		ax.invert_yaxis()
		plt.xlabel('time (ms)', fontsize=fontSize)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['left'].set_visible(False)
		plt.subplots_adjust(bottom=0.1, top=1.0, right=1.0)

		# calculate scalebar size and add scalebar
		round_to_n = lambda x, n, m: int(np.ceil(round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1)) / m)) * m
		scaley = 1000.0  # values in mV but want to convert to uV
		m = 10.0
		sizey = 100/scaley
		while sizey > 0.25*ydisp:
			try:
				sizey = round_to_n(0.2*ydisp*scaley, 1, m) / scaley
			except:
				sizey /= 10.0
			m /= 10.0
		labely = '%.3g $\mu$V'%(sizey*scaley)#)[1:]
		if len(electrodes) > 1:
			add_scalebar(ax,hidey=True, matchy=False, hidex=False, matchx=False, sizex=0, sizey=-sizey, labely=labely, unitsy='$\mu$V', scaley=scaley, loc=3, pad=0.5, borderpad=0.5, sep=3, prop=None, barcolor="black", barwidth=2)
		else:
			add_scalebar(ax, hidey=True, matchy=False, hidex=True, matchx=True, sizex=None, sizey=-sizey, labely=labely, unitsy='$\mu$V', scaley=scaley, unitsx='ms', loc=3, pad=0.5, borderpad=0.5, sep=3, prop=None, barcolor="black", barwidth=2)
		# save figure
		if saveFig:
			if isinstance(saveFig, basestring):
				filename = saveFig
			else:
				filename = sim.cfg.filename + '_LFP_timeseries.png'
			plt.savefig(filename, dpi=dpi)

	# Spectrogram ------------------------------
	if 'spectrogram' in plots:
		import matplotlib.cm as cm
		numCols = 1 #np.round(len(electrodes) / maxPlots) + 1
		figs.append(plt.figure(figsize=(figSize[0]*numCols, figSize[1])))

		# Morlet wavelet transform method
		if transformMethod == 'morlet':
			spec = []
			freqList = None
			if logy:
				freqList = np.logspace(np.log10(minFreq), np.log10(maxFreq), int((maxFreq-minFreq)/stepFreq))

			for i,elec in enumerate(electrodes):
				if elec == 'avg':
					lfpPlot = np.mean(lfp, axis=1)
				elif isinstance(elec, Number) and (inputLFP is not None or elec <= sim.net.recXElectrode.nsites):
					lfpPlot = lfp[:, elec]
				fs = int(1000.0 / sim.cfg.recordStep)
				t_spec = np.linspace(0, morlet.index2ms(len(lfpPlot), fs), len(lfpPlot))
				spec.append(MorletSpec(lfpPlot, fs, freqmin=minFreq, freqmax=maxFreq, freqstep=stepFreq, lfreq=freqList))

			f = freqList if freqList is not None else np.array(range(minFreq, maxFreq+1, stepFreq))   # only used as output for user

			vmin = np.array([s.TFR for s in spec]).min()
			vmax = np.array([s.TFR for s in spec]).max()

			for i,elec in enumerate(electrodes):
				plt.subplot(np.ceil(len(electrodes) / numCols), numCols, i + 1)
				T = timeRange
				F = spec[i].f
				if normSpec:
					spec[i].TFR = spec[i].TFR / vmax
					S = spec[i].TFR
					vc = [0, 1]
				else:
					S = spec[i].TFR
					vc = [vmin, vmax]

				plt.imshow(S, extent=(np.amin(T), np.amax(T), np.amin(F), np.amax(F)), origin='lower', interpolation='None', aspect='auto', vmin=vc[0], vmax=vc[1], cmap=plt.get_cmap('viridis'))
				if normSpec:
					plt.colorbar(label='Normalized power')
				else:
					plt.colorbar(label='Power')
				plt.ylabel('Hz')
				plt.tight_layout()
				if len(electrodes) > 1:
					plt.title('Electrode %s' % (str(elec)), fontsize=fontSize - 2)

		# FFT transform method
		elif transformMethod == 'fft':

			from scipy import signal as spsig
			spec = []

			for i,elec in enumerate(electrodes):
				if elec == 'avg':
					lfpPlot = np.mean(lfp, axis=1)
				elif isinstance(elec, Number) and elec <= sim.net.recXElectrode.nsites:
					lfpPlot = lfp[:, elec]
				# creates spectrogram over a range of data
				# from: http://joelyancey.com/lfp-python-practice/
				fs = int(1000.0/sim.cfg.recordStep)
				f, t_spec, x_spec = spsig.spectrogram(lfpPlot, fs=fs, window='hanning',
				detrend=mlab.detrend_none, nperseg=nperseg, noverlap=noverlap, nfft=NFFT,  mode='psd')
				x_mesh, y_mesh = np.meshgrid(t_spec*1000.0, f[f<maxFreq])
				spec.append(10*np.log10(x_spec[f<maxFreq]))

			vmin = np.array(spec).min()
			vmax = np.array(spec).max()

			for i,elec in enumerate(electrodes):
				plt.subplot(np.ceil(len(electrodes)/numCols), numCols, i+1)
				plt.pcolormesh(x_mesh, y_mesh, spec[i], cmap=cm.viridis, vmin=vmin, vmax=vmax)
				plt.colorbar(label='dB/Hz', ticks=[np.ceil(vmin), np.floor(vmax)])
				if logy:
					plt.yscale('log')
					plt.ylabel('Log-frequency (Hz)')
					if isinstance(logy, list):
						yticks = tuple(logy)
						plt.yticks(yticks, yticks)
				else:
					plt.ylabel('(Hz)')
				if len(electrodes) > 1:
					plt.title('Electrode %s'%(str(elec)), fontsize=fontSize-2)

		plt.xlabel('time (ms)', fontsize=fontSize)
		plt.tight_layout()
		if pop is None:
			plt.suptitle('LFP spectrogram', size=fontSize, fontweight='bold')
		elif pop is not None:
			spectTitle = 'LFP spectrogram of ' + pop + ' population'
			plt.suptitle(spectTitle, size=fontSize, fontweight='bold')

		plt.subplots_adjust(bottom=0.08, top=0.90)

		# save figure
		if saveFig:
			if isinstance(saveFig, basestring):
				filename = saveFig
			else:
				filename = sim.cfg.filename + '_LFP_timefreq.png'
			plt.savefig(filename, dpi=dpi)

	# locations ------------------------------
	if 'locations' in plots:
		try:
			cvals = [] # used to store total transfer resistance

			for cell in sim.net.compartCells:
				trSegs = list(np.sum(sim.net.recXElectrode.getTransferResistance(cell.gid)*1e3, axis=0)) # convert from Mohm to kilohm
				if not includeAxon:
					i = 0
					for secName, sec in cell.secs.items():
						nseg = sec['hObj'].nseg #.geom.nseg
						if 'axon' in secName:
							for j in range(i,i+nseg): del trSegs[j]
						i+=nseg
				cvals.extend(trSegs)

			includePost = [c.gid for c in sim.net.compartCells]
			fig = sim.analysis.plotShape(includePost=includePost, showElectrodes=electrodes, cvals=cvals, includeAxon=includeAxon, dpi=dpi,
			fontSize=fontSize, saveFig=saveFig, showFig=showFig, figSize=figSize)[0]
			figs.append(fig)
		except:
			print('  Failed to plot LFP locations...')



	outputData = {'LFP': lfp, 'electrodes': electrodes, 'timeRange': timeRange, 'saveData': saveData, 'saveFig': saveFig, 'showFig': showFig}

	if 'timeSeries' in plots:
		outputData.update({'t': t})

	# if 'PSD' in plots:
	# 	outputData.update({'allFreqs': allFreqs, 'allSignal': allSignal})

	if 'spectrogram' in plots:
		outputData.update({'spec': spec, 't': t_spec*1000.0, 'freqs': f[f<=maxFreq]})

	#save figure data
	if saveData:
		figData = outputData
		_saveFigData(figData, saveData, 'lfp')

	# show fig
	if showFig: plt.show() #_showFigure()


	return figs, outputData
#################################################################################################################



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

## Heatmaps for LFP data ## 
def getDataFrames(dataFile, timeRange, verbose=0):  ### Make this work with arbitrary input data, not just LFP so can look at CSD as well!!!! 
	## This function will return data frames of peak and average LFP amplitudes, for picking cell pops
	### dataFile: str 		--> .pkl file to load, with data from the whole recording
	### timeRange: list 	--> e.g. [start, stop]
	### verbose: bool 		--> if 0, return only the data frames; if 1 - return all lists and dataframes 

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
		cbarLabel = 'LFP amplitude (mV)'
	elif cbarLabel is 'CSD':
		cbarLabel = 'CSD amplitude (' + r'$\frac{mV}{mm^2}$' + ')'

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
	plt.ylabel('Channel', fontsize=labelFontSize)
	plt.yticks(rotation=0, fontsize=tickFontSize) 

	if saveFig:
		if savePath is None:
			prePath = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/LFP_heatmaps/'
		else:
			prePath = savePath
		fileName = 'heatmap.png'
		pathToFile = prePath + fileName
		plt.savefig(pathToFile, dpi=300)

	plt.show()

	return ax

## Pops with the highest contributions to LFP / CSD signal at a particular electrode ## 
def evalPops(dataFrame, electrode, verbose=0):
	##### -->  This function outputs a dict with the pops & values from the main & adjacent electrodes specified in the 'electrode' arg 
	## dataFrame: pandas dataFrame --> with peak or avg LFP or CSD data of each pop at each electrode --> output of getDataFrames
	## electrode: int 	--> electrode where the oscillation event of interest occurred - focus on data from this electrode plus the ones immediately above and below it
	## verbose: bool 	--> Default: 0 (False), only returns maxPopsValues. True returns dict + main electrode data frames. 

	#### SUBSET THE DATAFRAME ACCORDING TO SPECIFIED ELECTRODE & ADJACENT ELECTRODE(S) ####  
	if electrode == 0:									# if electrode is 0, then this is topmost, so only include 0, 1  	# dataFrameSubset = dataFrame[[elec, bottomElec]]
		# Determine electrodes to use for subsetting 
		elec = electrode
		bottomElec = electrode + 1
		topElec = None
		# Subset the dataFrame & create another with absolute values of the subset 
		dataFrameSubsetBottomElec = dataFrame[[bottomElec]]
		dataFrameSubsetBottomElecAbs = dataFrameSubsetBottomElec.abs()

	elif electrode == 19:	# if electrode is 19, this is bottom-most, so only include 18, 19 	# dataFrameSubset = dataFrame[[topElec, elec]]
		# Determine electrodes to use for subsetting 
		elec = electrode
		bottomElec = None
		topElec = electrode - 1
		# Subset the dataFrame & create another with absolute values of the subset 
		dataFrameSubsetTopElec = dataFrame[[topElec]]
		dataFrameSubsetTopElecAbs = dataFrameSubsetTopElec.abs()

	elif electrode > 0 and electrode < 19:														# dataFrameSubset = dataFrame[[topElec, elec, bottomElec]]
		# Determine electrodes to use for subsetting 
		elec = electrode
		bottomElec = electrode + 1
		topElec = electrode - 1
		# Subset the dataFrame & create another with absolute values of the subset 
		dataFrameSubsetTopElec = dataFrame[[topElec]]
		dataFrameSubsetTopElecAbs = dataFrameSubsetTopElec.abs()
		dataFrameSubsetBottomElec = dataFrame[[bottomElec]]
		dataFrameSubsetBottomElecAbs = dataFrameSubsetBottomElec.abs()

	dataFrameSubsetElec = dataFrame[[elec]]
	dataFrameSubsetElecAbs = dataFrameSubsetElec.abs()

	# So now I have: 
	## dataframes with electrode subset and adjacent electrode(s) subset (e.g. dataFrameSubsetElec, dataFrameSubsetBottomElec, dataFrameSubsetTopElec)
	## dataframes with absolute values of electrode subset and adjacent electrode(s) subset (e.g. dataFrameSubsetElecAbs, dataFrameSubsetBottomElecAbs)


	#### MAIN ELECTRODE #### 
	## Make dict for storage 
	maxPopsValues = {}
	maxPopsValues['elec'] = {}
	maxPopsValues['elec']['electrodeNumber'] = electrode
	## Values / pops associated with main electrode 
	dfElecAbs = dataFrameSubsetElecAbs.sort_values(by=[elec], ascending=False) # elec OR maxPopsValues['elec']['electrodeNumber']
		#### SUBSET VIA HARDCODED THRESHOLD --> 
	# dfElecSub = dfElecAbs[dfElecAbs[elec]>0.1]
		#### OR SUBSET BY TAKING ANY VALUE > (MAX VALUE / 3) ---> 
	# maxValueElec = list(dfElecAbs.max())[0]
	# dfElecSub = dfElecAbs[dfElecAbs[elec]>(maxValueElec/3)]
		#### OR SUBSET BY TAKING THE TOP 5 CONTRIBUTORS ----> 
	dfElecSub = dfElecAbs.head(5)
	elecPops = list(dfElecSub.index)
	## Find the non-absolute values of the above, and store this info along with the associated populations, into the storage dict:
	for pop in elecPops:
		maxPopsValues['elec'][pop] = list(dataFrameSubsetElec.loc[pop])[0]

	#### BOTTOM ADJACENT ELECTRODE #### 
	if bottomElec is not None:
		## Dict for storage 
		maxPopsValues['bottomElec'] = {}
		maxPopsValues['bottomElec']['electrodeNumber'] = bottomElec
		## Values / pops associated with bottom adjacent electrode 
		dfBottomElecAbs = dataFrameSubsetBottomElecAbs.sort_values(by=[bottomElec], ascending=False) # bottomElec OR maxPopsValues['bottomElec']['electrodeNumber']
		dfBottomElecSub = dfBottomElecAbs.head(5)		# dfBottomElecSub = dfBottomElecAbs[dfBottomElecAbs[bottomElec]>0.1]
		bottomElecPops = list(dfBottomElecSub.index)
		## Find the non-absolute values of the above, and store this info along with the associated populations, into the storage dict:
		for pop in bottomElecPops:
			maxPopsValues['bottomElec'][pop] = list(dataFrameSubsetBottomElec.loc[pop])[0]

	#### TOP ADJACENT ELECTRODE #### 
	if topElec is not None:
		## Dict for storage 
		maxPopsValues['topElec'] = {}
		maxPopsValues['topElec']['electrodeNumber'] = topElec
		## Values / pops associated with top adjacent electrode 
		dfTopElecAbs = dataFrameSubsetTopElecAbs.sort_values(by=[topElec], ascending=False) # topElec OR maxPopsValues['topElec']['electrodeNumber']
		dfTopElecSub = dfTopElecAbs.head(5) 		# dfTopElecSub = dfTopElecAbs[dfTopElecAbs[topElec]>0.1]
		topElecPops = list(dfTopElecSub.index)
		## Find the non-absolute values of the above, and store this info along with the associated populations, into the storage dict:
		for pop in topElecPops:
			maxPopsValues['topElec'][pop] = list(dataFrameSubsetTopElec.loc[pop])[0]

	if verbose:
		return maxPopsValues, dfElecSub, dataFrameSubsetElec
	else:
		return maxPopsValues

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
		spikeDict = getRateSpectrogramData(include=popList, timeRange=timeRange)   ## sim.analysis.getRateSpectrogramData
	elif graphType is 'hist':
		spikeDict = getSpikeHistData(include=popList, timeRange=timeRange, binSize=5, graphType='bar', measure='rate') ## sim.analysis.getSpikeHistData

	return spikeDict 
def plotCombinedSpike(timeRange, pop, colorDict, plotTypes=['spectrogram', 'histogram'], spectDict=None, histDict=None, figSize=(10,7), colorMap='jet', minFreq=None, maxFreq=None, vmaxContrast=None, savePath=None, saveFig=True):
	### timeRange: list 				--> e.g. [start, stop]
	### pop: str or list of length 1 	--> population to include 
	### colorDict: dict 				--> dict that corresponds pops to colors 
	### plotTypes: list 				--> ['spectrogram', 'histogram']
	### spectDict: dict 				--> can be gotten with getSpikeData(graphType='spect')
	### histDict: dict  				--> can be gotten with getSpikeData(graphType='hist')
	### figSize: tuple 					--> DEFAULT: (10,7)
	### colorMap: str 					--> DEFAULT: 'jet' 	--> cmap for ax.imshow lines --> Options are currently 'jet' or 'viridis' 
	### minFreq: int 					--> whole number that determines the minimum frequency plotted on the spectrogram 
	### maxFreq: int 					--> whole number that determines the maximum frequency plotted on the spectrogram 
	### vmaxContrast: float or int 		--> Denominator This will help with color contrast if desired!!!, e.g. 1.5 or 3
	### savePath: str   				--> Path to directory where fig should be saved; DEFAULT: '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/popContribFigs/'
	### saveFig: bool 					--> DEFAULT: True

 
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

	#### SPECTROGRAM CALCULATIONS ####-------------------------------------------------------
	if 'spectrogram' in plotTypes:
		allSignal = spectDict['allSignal']
		allFreqs = spectDict['allFreqs']


		# Set frequencies to be plotted (minFreq / maxFreq)
		if minFreq is None:
			minFreq = np.amin(allFreqs[0])			# DEFAULT: 1
		if maxFreq is None:
			maxFreq = np.amax(allFreqs[0])			# DEFAULT: 100
		# elif maxFreq is not None:					# maxFreq must be an integer!! 
		# 	if type(maxFreq) is not int:
		# 		maxFreq = round(maxFreq)


		# Set up imshowSignal
		imshowSignal = allSignal[0][minFreq:maxFreq+1]  ## Can't tell if the +1 is necessary or not here - leave it in for now!! 


		# Set color contrast parameters (vmin / vmax)
		if vmaxContrast is None:
			vmin = np.amin(imshowSignal)		# None
			vmax = np.amax(imshowSignal)		# None
		else:
			vmin = np.amin(imshowSignal)
			vmax = np.amax(imshowSignal) / vmaxContrast


	#### HISTOGRAM CALCULATIONS ####-------------------------------------------------------
	if 'histogram' in plotTypes:
		histoT = histDict['histoT']
		histoCount = histDict['histoData']

	#### PLOTTING ####-------------------------------------------------------
	if 'spectrogram' in plotTypes and 'histogram' in plotTypes:

		# Plot Spectrogram 
		ax1 = plt.subplot(211)
		img = ax1.imshow(imshowSignal, extent=(np.amin(timeRange), np.amax(timeRange), minFreq, maxFreq), origin='lower', 
				interpolation='None', aspect='auto', cmap=plt.get_cmap(colorMap), vmin=vmin, vmax=vmax)
		divider1 = make_axes_locatable(ax1)
		cax1 = divider1.append_axes('right', size='3%', pad = 0.2)
		fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)		## fmt lines are for colorbar to be in scientific notation
		fmt.set_powerlimits((0,0))
		plt.colorbar(img, cax = cax1, orientation='vertical', label='Power', format=fmt)
		ax1.set_title('Spike Rate Spectrogram for ' + popToPlot, fontsize=titleFontSize)
		ax1.set_ylabel('Frequency (Hz)', fontsize=labelFontSize)
		ax1.set_xlim(left=timeRange[0], right=timeRange[1])
		# ax1.set_ylim(minFreq, maxFreq) ## Redundant if ax1.imshow extent arg uses minFreq and maxFreq (and if imshowSignal is sliced as well)

		# Plot Histogram 
		ax2 = plt.subplot(212)
		ax2.bar(histoT, histoCount[0], width = 5, color=colorDict[popToPlot], fill=True)
		divider2 = make_axes_locatable(ax2)
		cax2 = divider2.append_axes('right', size='3%', pad = 0.2)
		cax2.axis('off')
		ax2.set_title('Spike Rate Histogram for ' + popToPlot, fontsize=titleFontSize)
		ax2.set_xlabel('Time (ms)', fontsize=labelFontSize)
		ax2.set_ylabel('Rate (Hz)', fontsize=labelFontSize) # CLARIFY Y AXIS
		ax2.set_xlim(left=timeRange[0], right=timeRange[1])

		# For potential figure saving
		figFilename = popToPlot + '_combinedSpike.png'


	elif 'spectrogram' in plotTypes and 'histogram' not in plotTypes:
		# Plot Spectrogram 
		ax1 = plt.subplot(111)
		img = ax1.imshow(imshowSignal, extent=(np.amin(timeRange), np.amax(timeRange), minFreq, maxFreq), origin='lower', 
				interpolation='None', aspect='auto', cmap=plt.get_cmap(colorMap), vmin=vmin, vmax=vmax) ## CHANGE maxFreq // np.amax(allFreqs[0])
		divider1 = make_axes_locatable(ax1)
		cax1 = divider1.append_axes('right', size='3%', pad = 0.2)
		fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)		## fmt lines are for colorbar to be in scientific notation
		fmt.set_powerlimits((0,0))
		plt.colorbar(img, cax = cax1, orientation='vertical', label='Power', format=fmt)
		ax1.set_title('Spike Rate Spectrogram for ' + popToPlot, fontsize=titleFontSize)
		ax1.set_ylabel('Frequency (Hz)', fontsize=labelFontSize)
		ax1.set_xlim(left=timeRange[0], right=timeRange[1])
		# ax1.set_ylim(minFreq, maxFreq)

		# For potential figure saving
		figFilename = popToPlot + '_spike_spectrogram.png'

	elif 'spectrogram' not in plotTypes and 'histogram' in plotTypes:
		# Plot Histogram 
		ax2 = plt.subplot(111)
		ax2.bar(histoT, histoCount[0], width = 5, color=colorDict[popToPlot], fill=True)
		divider2 = make_axes_locatable(ax2)
		cax2 = divider2.append_axes('right', size='3%', pad = 0.2)
		cax2.axis('off')
		ax2.set_title('Spike Rate Histogram for ' + popToPlot, fontsize=titleFontSize)
		ax2.set_xlabel('Time (ms)', fontsize=labelFontSize)
		ax2.set_ylabel('Rate (Hz)', fontsize=labelFontSize) # CLARIFY Y AXIS
		ax2.set_xlim(left=timeRange[0], right=timeRange[1])

		# For potential figure saving
		figFilename = popToPlot + '_spike_histogram.png'

	plt.tight_layout()

	## Save figure
	if saveFig:
		if savePath is None:
			prePath = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/popContribFigs/' 	# popContribFigs_cmapJet/'
		else:
			prePath = savePath
		# fileName = pop + '_combinedSpike.png'
		pathToFile = prePath + figFilename	# fileName
		plt.savefig(pathToFile, dpi=300)

	## Show figure
	plt.show()

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

	lfpOutput = getLFPData(pop=popList, timeRange=timeRange, electrodes=electrodeList, plots=plots) # sim.analysis.getLFPData # filtFreq=filtFreq (see above; in args)

	return lfpOutput
def plotCombinedLFP(timeRange, pop, colorDict, plotTypes=['spectrogram', 'timeSeries'], spectDict=None, timeSeriesDict=None, figSize=(10,7), colorMap='jet', minFreq=None, maxFreq=None, vmaxContrast=None, electrode=None, savePath=None, saveFig=True): # electrode='avg',
	### timeRange: list 						--> [start, stop]
	### pop: list or str 						--> relevant population to plot data for 
	### colorDict: dict 						--> corresponds pop to color 
	### plotTypes: list 						--> DEFAULT: ['spectrogram', 'timeSeries'] 
	### spectDict: dict 						--> Contains spectrogram data; output of getLFPDataDict
	### timeSeriesDict: dict 					--> Contains timeSeries data; output of getLFPDataDict
	### figSize: tuple 							--> DEFAULT: (10,7)
	### colorMap: str 							--> DEFAULT: 'jet' 	--> cmap for ax.imshow lines --> Options are currently 'jet' or 'viridis' 
	### maxFreq: int 							--> whole number that determines the maximum frequency plotted on the spectrogram 
	### minFreq: int 							--> whole number that determines the minimum frequency plotted on the spectrogram
	### vmaxContrast: float or int 				--> Denominator This will help with color contrast if desired!!!, e.g. 1.5 or 3
	### electrode: int 					--> electrode at which to plot the CSD data 
		# ^^^ REPLACEMENT ### titleElectrode: str or (1-element) list	-->  FOR USE IN PLOT TITLES !! --> This is for the electrode that will appear in the title 
	### savePath: str 	  						--> Path to directory where fig should be saved; DEFAULT: '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/popContribFigs/'
	### saveFig: bool 							--> DEFAULT: True 
		### NOTE: RE-WORK HOW THE ELECTRODE GETS FACTORED INTO THE TITLE AND SAVING!! 

	# Get relevant pop
	if type(pop) is str:
		popToPlot = pop
	elif type(pop) is list:
		popToPlot = pop[0]

	## Create figure 
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figSize)

	# Set electrode variable, for plot title(s)
	if type(electrode) is list:
		electrode = electrode[0]


	## Set font size 
	labelFontSize = 15 		# NOTE: in plotCombinedSpike, labelFontSize = 12
	titleFontSize = 20


	#### SPECTROGRAM CALCULATIONS ####-------------------------------------------------------
	if 'spectrogram' in plotTypes:
		spec = spectDict['spec']

		S = spec[0].TFR
		F = spec[0].f
		T = timeRange

		## Set up vmin / vmax color contrasts 
		orig_vmin = np.array([s.TFR for s in spec]).min()
		orig_vmax = np.array([s.TFR for s in spec]).max()

		if vmaxContrast is None:
			vmin = orig_vmin
			vmax = orig_vmax
		else:
			vmin = orig_vmin
			vmax = orig_vmax / vmaxContrast 
			print('original vmax: ' + str(orig_vmax))
			print('new vmax: ' + str(vmax))

		vc = [vmin, vmax]


		## Set up minFreq and maxFreq for spectrogram
		if minFreq is None:
			minFreq = np.amin(F) 	# 1
		if maxFreq is None:
			maxFreq = np.amax(F)	# 100
		print('minFreq: ' + str(minFreq) + ' Hz')
		print('maxFreq: ' + str(maxFreq) + ' Hz')

		## Set up imshowSignal
		imshowSignal = S[minFreq:maxFreq+1] ## NOTE: Is the +1 necessary here or not? Same question for spiking data. Leave it in for now. 



	#### TIME SERIES CALCULATIONS ####-------------------------------------------------------
	if 'timeSeries' in plotTypes:
		t = timeSeriesDict['t']
		lfpPlot = timeSeriesDict['lfpPlot']



	#### PLOTTING ####-------------------------------------------------------
	# Plot both 
	if 'spectrogram' in plotTypes and 'timeSeries' in plotTypes:

		### PLOT SPECTROGRAM ###
		# spectrogram title #
		spectTitle = 'LFP Spectrogram for ' + popToPlot + ', electrode ' + str(electrode)   ## NOTE: make condition for avg elec? 
		# plot and format # 
		ax1 = plt.subplot(2, 1, 1)
		img = ax1.imshow(imshowSignal, extent=(np.amin(T), np.amax(T), minFreq, maxFreq), origin='lower', interpolation='None', aspect='auto', 
			vmin=vc[0], vmax=vc[1], cmap=plt.get_cmap(colorMap))	 # imshowSignal --> S 	 	# minFreq --> np.amin(F)  	# maxFreq --> np.amax(F) 
		divider1 = make_axes_locatable(ax1)
		cax1 = divider1.append_axes('right', size='3%', pad=0.2)
		fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)		## fmt lines are for colorbar scientific notation
		fmt.set_powerlimits((0,0))
		plt.colorbar(img, cax = cax1, orientation='vertical', label='Power', format=fmt)
		ax1.set_title(spectTitle, fontsize=titleFontSize)
		ax1.set_ylabel('Frequency (Hz)', fontsize=labelFontSize)
		ax1.set_xlim(left=timeRange[0], right=timeRange[1])
		# ax1.set_ylim(minFreq, maxFreq)		## Redundant if ax1.imshow extent reflects minFreq and maxFreq


		### PLOT TIME SERIES ###
		# timeSeries title # 
		timeSeriesTitle = 'LFP Signal for ' + popToPlot + ', electrode ' + str(electrode)
		# plot and format #
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

		# For potential figure saving
		figFilename = popToPlot + '_combinedLFP_elec_' + str(electrode) + '.png'

	# Plot only spectrogram 
	elif 'spectrogram' in plotTypes and 'timeSeries' not in plotTypes:
		### PLOT SPECTROGRAM ###
		# spectrogram title # 
		spectTitle = 'LFP Spectrogram for ' + popToPlot + ', electrode ' + str(electrode)
		# plot and format # 
		ax1 = plt.subplot(1, 1, 1)
		img = ax1.imshow(imshowSignal, extent=(np.amin(T), np.amax(T), minFreq, maxFreq), origin='lower', interpolation='None', aspect='auto', 
			vmin=vc[0], vmax=vc[1], cmap=plt.get_cmap(colorMap))  # imshowSignal --> S 	 	# minFreq --> np.amin(F)  	# maxFreq --> np.amax(F) 
		divider1 = make_axes_locatable(ax1)
		cax1 = divider1.append_axes('right', size='3%', pad=0.2)
		fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)		## fmt lines are for colorbar scientific notation
		fmt.set_powerlimits((0,0))
		plt.colorbar(img, cax = cax1, orientation='vertical', label='Power', format=fmt)
		ax1.set_title(spectTitle, fontsize=titleFontSize)
		ax1.set_ylabel('Frequency (Hz)', fontsize=labelFontSize)
		ax1.set_xlim(left=timeRange[0], right=timeRange[1])
		# ax1.set_ylim(minFreq, maxFreq)		## Redundant if ax1.imshow extent reflects minFreq and maxFreq

		# For potential figure saving
		figFilename = popToPlot + '_LFP_spectrogram_elec_' + str(electrode) + '.png'

	# Plot only timeSeries 
	elif 'spectrogram' not in plotTypes and 'timeSeries' in plotTypes:
		### PLOT TIME SERIES ###
		# timeSeries title # 
		timeSeriesTitle = 'LFP Signal for ' + popToPlot + ', electrode ' + str(electrode)
		# plot and format # 
		lw = 1.0
		ax2 = plt.subplot(1, 1, 1)
		divider2 = make_axes_locatable(ax2)
		cax2 = divider2.append_axes('right', size='3%', pad=0.2)
		cax2.axis('off')
		ax2.plot(t[0:len(lfpPlot)], lfpPlot, color=colorDict[popToPlot], linewidth=lw)
		ax2.set_title(timeSeriesTitle, fontsize=titleFontSize)
		ax2.set_xlabel('Time (ms)', fontsize=labelFontSize)
		ax2.set_xlim(left=timeRange[0], right=timeRange[1])
		ax2.set_ylabel('LFP Amplitudes (mV)', fontsize=labelFontSize)

		# For potential figure saving
		figFilename = popToPlot + '_LFP_timeSeries_elec_' + str(electrode) + '.png'

	plt.tight_layout()

	## Save figure 
	if saveFig:
		if savePath is None:
			prePath = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/popContribFigs/' 		# popContribFigs_cmapJet/'
		else:
			prePath = savePath
		pathToFile = prePath + figFilename
		plt.savefig(pathToFile, dpi=300)

	## Show figure 
	plt.show()

def getSumLFP(dataFile, pops, elecs=False, timeRange=None, showFig=False):
	# THIS FUNCTON ALLOWS YOU TO ADD TOGETHER LFP CONTRIBUTIONS FROM ARBITRARY POPS AT SPECIFIED ELECTRODES
	### dataFile: str --> .pkl file to load w/ simulation data 
	### pops: list OR dict --> list if just a list of pops, dict if including electrodes (e.g. {'IT3': 1, 'IT5A': 10, 'PT5B': 11})
			### popElecDict: dict --> e.g. {'IT3': 1, 'IT5A': 10, 'PT5B': 11}
	### timeRange: list --> e.g. [start, stop]
	### showFig: bool --> Determines whether or not plt.show() will be called 
	### elecs: bool ---> False by default; means that electrodes will NOT be taken into account and output will be the total LFP.

	print('Calculating combined LFP signal')

	sim.load(dataFile, instantiate=False)  ### Figure out efficiency here!!  

	if timeRange is None:   	# if timeRange is None, use the whole simulation! 
		timeRange = [0, sim.cfg.duration]


	lfpData = {} 	# dictionary to store lfp data 

	## Calculating sum of LFP signals at specified electrodes! 
	if elecs:
		if type(pops) is list:  	# We need pops w/ corresponding electrodes if we want to calculate summed LFP at specfic electrodes! 
			print('pops argument needs to be a dict with pop as key & corresponding electrode as value, e.g. {\'IT3\': 1, \'IT5A\': 10, \'PT5B\': 11}')
			lfpData = None
		elif type(pops) is dict:
			for pop in pops:
				lfpData[pop] = {}
				elec = pops[pop]		# electrode that corresponds with the current pop being evaluated! 
				popLFPdata = np.array(sim.allSimData['LFPPops'][pop])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]
				lfpData[pop]['elec'] = popLFPdata[:, elec]

			popList = list(pops.keys()) 	## Make a list from the 'pops' dict keys -- this will be a list of the relevant populations 
			lfpData['sum'] = np.zeros(lfpData[popList[0]]['elec'].shape) 	## have to establish an array of zeros first so the below 'for' loop functions properly
			for i in range(len(pops)):
				lfpData['sum'] += lfpData[popList[i]]['elec']
	## Calculating sum of LFP signals over ALL electrodes! 
	else:
		print('Calculating summed LFP signal from ' + str(pops) + ' over all electrodes')
		for pop in pops:
			lfpData[pop] = np.array(sim.allSimData['LFPPops'][pop])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]
		lfpData['sum'] = np.zeros(lfpData[pops[0]].shape)  ## have to establish an array of zeros first so the below 'for' loop functions properly
		for i in range(len(pops)):
			lfpData['sum'] += lfpData[pops[i]]

	### PLOTTING ### 
	if showFig:
		t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)
		plt.figure(figsize = (12,7))
		plt.xlabel('Time (ms)')
		### Create title of plot ### 
		titlePreamble = 'Summed LFP signal from'
		popsInTitle = ''
		if elecs and type(pops) is dict:
			for pop in pops:
				popsInTitle += ' ' + pop + '(elec ' + str(pops[pop]) + ')'
			plt.ylabel('LFP Amplitude (mV)')	### y axis label 
			plt.plot(t, lfpData['sum'])   #### PLOT HERE 
			lfpSumTitle = titlePreamble + popsInTitle 
			plt.title(lfpSumTitle)
			plt.legend()
			plt.show()
		elif not elecs:
			print('Use plotLFP')
			# for pop in pops:
			# 	popsInTitle += ' ' + pop + ' '
			# for i in range(lfpData['sum'].shape[1]):  ## number of electrodes 
			# 	lfpPlot = lfpData['sum'][:, i]
			# 	lw = 1.0 ## line-width
			# 	ydisp = np.absolute(lfpPlot).max() 
			# 	plt.plot(t, -lfpPlot+(i*ydisp), linewidth=lw, label='electrode ' + str(i))

	return lfpData 

## CSD: data and plotting ## 
def getCSDdata(dataFile=None, outputType=['timeSeries', 'spectrogram'], oscEventInfo=None, dt=None, sampr=None, pop=None, spacing_um=100, minFreq=1, maxFreq=100, stepFreq=1):
	#### Outputs a dict with CSD and other relevant data for plotting! 
	## dataFile: str     					--> .pkl file with recorded simulation 
	## outputType: list of strings 			--> options are 'timeSeries' +/- 'spectrogram' --> OR could be empty, if want csdData from all electrodes!! 
	## oscEventInfo: dict 					---> dict w/ chan, left, right, minT, maxT, alignoffset
		# DEPRECATED ## timeRange: list 						--> e.g. [start, stop]
		# DEPRECATED ## channel: list or None 				--> e.g. [8], None
	## dt: time step of the simulation 		--> (usually --> sim.cfg.recordStep)
	## sampr: sampling rate (Hz) 			--> (usually --> 1/(dt/1000))
	## pop: str or list 					--> e.g. 'ITS4' or ['ITS4']
	## spacing_um: int 						--> 100 by DEFAULT (spacing between electrodes in MICRONS)
	## minFreq: float / int 				--> DEFAULT: 1 Hz  
	## maxFreq: float / int 				--> DEFAULT: 100 Hz 
	## stepFreq: float / int 				--> DEFAULT: 1 Hz 
			## TO DO: 
			###  --> Should I also have an lfp_input option so that I can get the CSD data of summed LFP data...?
			###  --> Should I make it such that output can include both timeSeries and spectrogram so don't have to run this twice? test this!! 

	## load .pkl simulation file 
	if dataFile:
		sim.load(dataFile, instantiate=False)
	else:
		print('No dataFile; will use data from dataFile already loaded elsewhere!')


	## Extract oscillation event info 
	if oscEventInfo is not None:
		# chan, left, right, minT, maxT, alignoffset
		# RECALL: HAVE TO CORRECT FOR 6_11 !!! THIS WORKS FOR 0_6 AS IS!!! 
		chan = oscEventInfo['chan']
		print('channel: ' + str(chan))
		left = oscEventInfo['left']
		print('left: ' + str(left))
		right = oscEventInfo['right']
		print('right: ' + str(right))
		minT = oscEventInfo['minT']
		print('minT: ' + str(minT))
		maxT = oscEventInfo['maxT']
		print('maxT: ' + str(maxT))
		alignoffset = oscEventInfo['alignoffset']
		print('alignoffset: ' + str(alignoffset))
		w2 = oscEventInfo['w2']
		print('w2: ' + str(w2))
	else:
		chan=None
		print('No oscillation event data detected!')



	## Determine timestep, sampling rate, and electrode spacing 
	dt = sim.cfg.recordStep  	# or should I divide by 1000.0 up here, and then just do 1.0/dt below for sampr? 
	sampr = 1.0/(dt/1000.0) 	# divide by 1000.0 to turn denominator from units of ms to s
	spacing_um = spacing_um		# 100um by default # 


	## Get LFP data   # ----> NOTE: SHOULD I MAKE AN LFP INPUT OPTION?????? FOR SUMMED LFP DATA??? (also noted above in arg descriptions)
	if pop is None:
		lfpData = sim.allSimData['LFP']
	else:
		if type(pop) is list:
			pop = pop[0]
		lfpData = sim.allSimData['LFPPops'][pop]



	## Get CSD data for ALL channels, over ALL timepoints
	csdData = csd.getCSD(LFP_input_data=lfpData, dt=dt, sampr=sampr, spacing_um=spacing_um, vaknin=True)
	# Input full CSD data into outputData dict 
	outputData = {'csd': csdData}  # REMEMBER: ALL CHANS, ALL TIMEPOINTS!! 



	# timeSeries --------------------------------------------
	if 'timeSeries' in outputType and oscEventInfo is not None:  ### make case for when it IS None
		print('Returning timeSeries data')

		##############################
		#### BEFORE THE OSC EVENT ####
		idx0_before = max(0,left - w2)
		idx1_before = left 
		# (1) Calculate CSD before osc event
		csdBefore = csdData[chan,idx0_before:idx1_before]
		# (2) Calculate timepoint data 
		beforeT = (maxT-minT) * (idx1_before - idx0_before) / (right - left + 1)
		tt_before = np.linspace(minT-beforeT,minT,len(csdBefore)) + alignoffset
		# (3) Input time and CSD data for before osc event into outputData dict 
		outputData.update({'beforeT': beforeT})
		outputData.update({'tt_before': tt_before})
		outputData.update({'csdBefore': csdBefore})

		##############################
		#### DURING THE OSC EVENT #### 			--> # Need left, right, minT, maxT, alignoffset
		# (1) Calculate CSD during osc event
		csdDuring = csdData[chan,left:right]
		# (2) Calculate timepoint data 
		tt_during = np.linspace(minT,maxT,len(csdDuring)) + alignoffset				# 		duringT = np.linspace(minT,maxT,len(csdDuring)) + alignoffset 
		# (3) Input time and CSD data for during osc event into outputData dict 
		outputData.update({'tt_during': tt_during})
		outputData.update({'csdDuring': csdDuring})

		#############################
		#### AFTER THE OSC EVENT #### 
		idx0_after = int(right)
		idx1_after = min(idx0_after + w2,max(csdData.shape[0],csdData.shape[1]))
		# (1) Calculate CSD after osc event
		csdAfter = csdData[chan,idx0_after:idx1_after]
		# (2) Calculate timepoint data 
		afterT = (maxT-minT) * (idx1_after - idx0_after) / (right - left + 1)		#		print('afterT: ' + str(afterT))
		tt_after = np.linspace(maxT,maxT+afterT,len(csdAfter)) + alignoffset
		# (3) Input time and CSD data for after osc event into outputData dict 
		outputData.update({'afterT': afterT})
		outputData.update({'tt_after': tt_after})
		outputData.update({'csdAfter': csdAfter})

		# Update outputData with the channel info as well
		outputData.update({'chan': chan})

		# Get xlim for plotting
		xl = (minT-beforeT + alignoffset, maxT+afterT + alignoffset)
		outputData.update({'xl': xl})

	# spectrogram -------------------------------------------
	if 'spectrogram' in outputType:
		print('Returning spectrogram data')

		spec = []
		freqList = None

		## Determine electrode(s) to loop over: 
		if electrode is None: 
			print('No electrode specified; returning spectrogram data for ALL electrodes')
			electrode = []
			electrode.extend(list(range(int(sim.net.recXElectrode.nsites))))

		print('Channels considered for spectrogram data: ' + str(electrode))


		## Spectrogram Data Calculations! 
		if len(electrode) > 1:
			for i, elec in enumerate(electrode):
				csdDataSpect_allElecs = np.transpose(csdData_allElecs)  # Transposing this data may not be necessary!!! 
				csdDataSpect = csdDataSpect_allElecs[:, elec]
				fs = int(1000.0 / sim.cfg.recordStep)
				t_spec = np.linspace(0, morlet.index2ms(len(csdDataSpect), fs), len(csdDataSpect)) 
				spec.append(MorletSpec(csdDataSpect, fs, freqmin=minFreq, freqmax=maxFreq, freqstep=stepFreq, lfreq=freqList))

		elif len(electrode) == 1: 	# use csdData, as determined above the timeSeries and spectrogram 'if' statements (already has correct electrode-specific CSD data!)
			fs = int(1000.0 / sim.cfg.recordStep)
			t_spec = np.linspace(0, morlet.index2ms(len(csdData), fs), len(csdData)) # Seems this is only used for the fft circumstance...? 
			spec.append(MorletSpec(csdData, fs, freqmin=minFreq, freqmax=maxFreq, freqstep=stepFreq, lfreq=freqList))


		## Get frequency list 
		f = freqList if freqList is not None else np.array(range(minFreq, maxFreq+1, stepFreq))   # only used as output for user

		## vmin, vmax --> vc = [vmin, vmax]
		vmin = np.array([s.TFR for s in spec]).min()
		vmax = np.array([s.TFR for s in spec]).max()
		vc = [vmin, vmax]

		## T (timeRange)
		T = timeRange 

		## F, S
		for i, elec in enumerate(electrode):   # works for electrode of length 1 or greater! No need for if statement regarding length. 
			F = spec[i].f
			# if normSpec:  #### THIS IS FALSE BY DEFAULT, SO COMMENTING IT OUT HERE 
				# spec[i].TFR = spec[i].TFR / vmax
				# S = spec[i].TFR
				# vc = [0, 1]
			S = spec[i].TFR


		outputData.update({'T': T, 'F': F, 'S': S, 'vc': vc})  ### All the things necessary for plotting!! 
		outputData.update({'spec': spec, 't': t_spec*1000.0, 'freqs': f[f<=maxFreq]}) 
			### This is at the end of the plotLFP and getLFPdata functions, but not sure what purpose it will serve; keep this in for now!! 
			### ^^ ah, well, could derive F and S from spec. not sure I need 't' or 'freqs' though? hmmm. 

	return outputData
def plotCombinedCSD(pop, electrode, colorDict, timeSeriesDict=None, spectDict=None, alignoffset=None, vmaxContrast=None, colorMap='jet', figSize=(10,7), minFreq=None, maxFreq=None, plotTypes=['timeSeries', 'spectrogram'], hasBefore=1, hasAfter=1, savePath=None, saveFig=True):
	### pop: list or str 				--> relevant population to plot data for 
	### electrode: int 					--> electrode at which to plot the CSD data 
	### colorDict: dict 				--> corresponds pop to color 
	### timeSeriesDict: dict 			--> output of getCSDdata (with outputType=['timeSeries'])
	### spectDict: dict 				--> output of getCSDdata (with outputType=['spectrogram'])
	### alignoffset: float 				!! --> Can be gotten from plotWavelets.py in a1dat [TO DO: ELABORATE ON THIS / LINK CALCULATIONS]
	### vmaxContrast: float or int 		--> Denominator; This will help with color contrast if desired!, e.g. 1.5 or 3 (DEFAULT: None)
	### colorMap: str 					--> DEFAULT: jet; cmap for ax.imshow lines --> Options are currently 'jet' or 'viridis'
	### figSize: tuple 					--> DEFAULT: (10,7)
	### minFreq: int 					--> DEFAULT: None
	### maxFreq: int 					--> DEFAULT: None
	### plotTypes: list 				--> DEFAULT: ['timeSeries', 'spectrogram']
	### hasBefore: bool 				--> DEFAULT: 1 (before osc event data will be plotted)
	### hasAfter: bool  				--> DEFAULT: 1 (after osc event data will be plotted)
	### savePath: str    				--> Path to directory where fig should be saved; DEFAULT: '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/popContribFigs/'
	### saveFig: bool 					--> DEFAULT: True
		### TO DO: CHANGE ALL INSTANCES OF "ELECTRODE" TO "CHANNEL"

	# Get relevant pop
	if type(pop) is str:
		popToPlot = pop
	elif type(pop) is list:
		popToPlot = pop[0]

	# Create figure 
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figSize)

	# Set font sizes
	labelFontSize = 12  ## NOTE: spike has this as 12, lfp plotting has this as 15 
	titleFontSize = 20

	# Set electrode variable, for plot title(s)
	if type(electrode) is list:
		electrode = electrode[0]


	#### SPECTROGRAM CALCULATIONS ####-------------------------------------------------------
	if 'spectrogram' in plotTypes:

		# These lines will work as long as the getCSDdata function that retrieves the spectDict had only 1 electrode in electrode arg!!
		S = spectDict['S']
		F = spectDict['F']
		T = spectDict['T']				# timeRange

		# vmin and vmax  -- adjust for color contrast purposes, if desired 
		vc = spectDict['vc']
		orig_vmin = vc[0]
		orig_vmax = vc[1]

		if vmaxContrast is None:
			vmin = orig_vmin
			vmax = orig_vmax
		else:
			print('original vmax: ' + str(orig_vmax))
			vmin = orig_vmin
			vmax = orig_vmax / vmaxContrast
			print('new vmax: ' + str(vmax))
			vc = [vmin, vmax]


		## Set up minFreq and maxFreq for spectrogram
		if minFreq is None:
			minFreq = np.amin(F) 	# 1
		if maxFreq is None:
			maxFreq = np.amax(F)	# 100

		## Set up imshowSignal
		imshowSignal = S[minFreq:maxFreq+1] ## NOTE: Is the +1 necessary here or not? Same question for spiking data. Leave it in for now. 


	#### TIME SERIES CALCULATIONS ####-------------------------------------------------------
	if 'timeSeries' in plotTypes:  ### NOTE!!!! --> should I do "hasbefore" and 'hasafter' as args?!??! 
		# time (x-axis)
		tt_during = timeSeriesDict['tt_during']

		# CSD (y-axis)
		csdDuring = timeSeriesDict['csdDuring']

		if hasBefore:
			# time (x-axis)
			tt_before = timeSeriesDict['tt_before']
			# CSD (y-axis)
			csdBefore = timeSeriesDict['csdBefore']

		if hasAfter:
			# time (x-axis)
			tt_after = timeSeriesDict['tt_after']
			# CSD (y-axis)
			csdAfter = timeSeriesDict['csdAfter']

	#### PLOTTING ####-------------------------------------------------------
	if 'spectrogram' in plotTypes and 'timeSeries' in plotTypes:

		### PLOT SPECTROGRAM ### 
		# spectrogram title
		spectTitle = 'CSD Spectrogram for ' + popToPlot + ', channel ' + str(electrode)
		# plot and format 
		ax1 = plt.subplot(2, 1, 1)
		img = ax1.imshow(imshowSignal, extent=(np.amin(T), np.amax(T), minFreq, maxFreq), origin='lower', interpolation='None', aspect='auto', 
			vmin=vc[0], vmax=vc[1], cmap=plt.get_cmap(colorMap))  # minFreq --> np.amin(F)  	# maxFreq --> np.amax(F)	# imshowSignal --> S
		divider1 = make_axes_locatable(ax1)
		cax1 = divider1.append_axes('right', size='3%', pad=0.2)
		fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)		## fmt lines are for colorbar scientific notation
		fmt.set_powerlimits((0,0))
		plt.colorbar(img, cax = cax1, orientation='vertical', label='Power', format=fmt)
		ax1.set_title(spectTitle, fontsize=titleFontSize)
		ax1.set_ylabel('Frequency (Hz)', fontsize=labelFontSize)
		ax1.set_xlim(left=T[0], right=T[1]) 			# ax1.set_xlim(left=timeRange[0], right=timeRange[1])
		# ax1.set_ylim(minFreq, maxFreq)				# Uncomment this if using the commented-out ax1.imshow (with S, and with np.amin(F) etc.)

		### PLOT TIMESERIES ###
		# timeSeries title
		timeSeriesTitle = 'CSD Signal for ' + popToPlot + ', channel ' + str(timeSeriesDict['chan'])

		# y-axis label
		timeSeriesYAxis = 'CSD Amplitude (' + r'$\frac{mV}{mm^2}$' + ')'

		# Format timeSeries plot 
		lw = 1.0
		ax2 = plt.subplot(2, 1, 2)
		divider2 = make_axes_locatable(ax2)
		cax2 = divider2.append_axes('right', size='3%', pad=0.2)
		cax2.axis('off')
		# Plot CSD before Osc Event
		if hasBefore: 
			ax2.plot(tt_before, csdBefore, color='k', linewidth=1.0)
		# Plot CSD during Osc Event
		ax2.plot(tt_during, csdDuring, color=colorDict[popToPlot], linewidth=lw) 		# ax2.plot(t, csdTimeSeries, color=colorDict[popToPlot], linewidth=lw) # 	# ax2.plot(t[0:len(lfpPlot)], lfpPlot, color=colorDict[popToPlot], linewidth=lw)
		# Plot CSD after Osc Event 
		if hasAfter:
			ax2.plot(tt_after, csdAfter, color='k', linewidth=1.0)
		# Set title and x-axis label
		ax2.set_title(timeSeriesTitle, fontsize=titleFontSize)
		ax2.set_xlabel('Time (ms)', fontsize=labelFontSize)
		# Set x-axis limits 
		if hasBefore and hasAfter:
			xl = timeSeriesDict['xl']
			ax2.set_xlim((xl))
		else:
			ax2.set_xlim(left=tt_during[0], right=tt_during[-1]) 		# ax2.set_xlim(left=t[0], right=t[-1]) 			# ax2.set_xlim(left=timeRange[0], right=timeRange[1])
		# Set y-axis label 
		ax2.set_ylabel(timeSeriesYAxis, fontsize=labelFontSize)

		# For potential saving 
		figFilename = popToPlot + '_combinedCSD_chan_' + str(electrode) + '.png'

	elif 'spectrogram' in plotTypes and 'timeSeries' not in plotTypes:

		### PLOT SPECTROGRAM ### 
		# spectrogram title
		spectTitle = 'CSD Spectrogram for ' + popToPlot + ', channel ' + str(electrode)
		# plot and format 
		ax1 = plt.subplot(1, 1, 1)
		# img = ax1.imshow(S, extent=(np.amin(T), np.amax(T), np.amin(F), np.amax(F)), origin='lower', interpolation='None', aspect='auto', 
		# 	vmin=vc[0], vmax=vc[1], cmap=plt.get_cmap(colorMap)) ## ANSWER: NO --> NOTE: instead of np.amax(F) should I be doing maxFreq? (similarly for minFreq // np.amin(F))
		img = ax1.imshow(imshowSignal, extent=(np.amin(T), np.amax(T), minFreq, maxFreq), origin='lower', interpolation='None', aspect='auto', 
			vmin=vc[0], vmax=vc[1], cmap=plt.get_cmap(colorMap)) 
		divider1 = make_axes_locatable(ax1)
		cax1 = divider1.append_axes('right', size='3%', pad=0.2)
		fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)		## fmt lines are for colorbar scientific notation
		fmt.set_powerlimits((0,0))
		plt.colorbar(img, cax = cax1, orientation='vertical', label='Power', format=fmt)
		ax1.set_title(spectTitle, fontsize=titleFontSize)
		ax1.set_xlabel('Time (ms)', fontsize=labelFontSize)
		ax1.set_ylabel('Frequency (Hz)', fontsize=labelFontSize)
		ax1.set_xlim(left=T[0], right=T[1]) 			# ax1.set_xlim(left=timeRange[0], right=timeRange[1])
		# ax1.set_ylim(minFreq, maxFreq) 				# Uncomment this if using the commented-out ax1.imshow (with S, and with np.amin(F) etc.)

		# For potential saving 
		figFilename = popToPlot + '_CSD_spectrogram_chan_' + str(electrode) + '.png'

	elif 'spectrogram' not in plotTypes and 'timeSeries' in plotTypes:

		### PLOT TIME SERIES ### 
		# timeSeries title
		timeSeriesTitle = 'CSD Signal for ' + popToPlot + ', channel ' + str(timeSeriesDict['chan'])

		# y-axis label
		timeSeriesYAxis = 'CSD Amplitude (' + r'$\frac{mV}{mm^2}$' + ')'

		# Format timeSeries plot 
		lw = 1.0
		ax2 = plt.subplot(1, 1, 1)
		divider2 = make_axes_locatable(ax2)
		cax2 = divider2.append_axes('right', size='3%', pad=0.2)
		cax2.axis('off')
		# Plot CSD before Osc Event
		if hasBefore: 
			ax2.plot(tt_before, csdBefore, color='k', linewidth=1.0)
		# Plot CSD during Osc Event 			# 		ax2.plot(t, csdTimeSeries, color=colorDict[popToPlot], linewidth=lw) # 	# ax2.plot(t[0:len(lfpPlot)], lfpPlot, color=colorDict[popToPlot], linewidth=lw)
		ax2.plot(tt_during, csdDuring, color=colorDict[popToPlot], linewidth=lw) # 	# ax2.plot(t[0:len(lfpPlot)], lfpPlot, color=colorDict[popToPlot], linewidth=lw)
		# Plot CSD after Osc Event 
		if hasAfter:
			ax2.plot(tt_after, csdAfter, color='k', linewidth=1.0)
		# Set title and x-axis label
		ax2.set_title(timeSeriesTitle, fontsize=titleFontSize)
		ax2.set_xlabel('Time (ms)', fontsize=labelFontSize)
		# Set x-axis limits 
		if hasBefore and hasAfter:
			xl = timeSeriesDict['xl']
			ax2.set_xlim((xl))
		elif hasBefore and not hasAfter:
			ax2.set_xlim(left=tt_before[0], right=tt_during[-1])
		elif not hasBefore and hasAfter:
			ax2.set_xlim(left=tt_during[0], right=tt_after[-1])
		else:
			ax2.set_xlim(left=tt_during[0], right=tt_during[-1])	# ax2.set_xlim(left=t[0], right=t[-1]) 	# ax2.set_xlim(left=timeRange[0], right=timeRange[1])
		# Set y-axis label 
		ax2.set_ylabel(timeSeriesYAxis, fontsize=labelFontSize)

		# For potential saving 
		figFilename = popToPlot + '_CSD_timeSeries_chan_' + str(electrode) + '.png'
 
	plt.tight_layout()

	## Save figure 
	if saveFig:
		if savePath is None:
			prePath = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/popContribFigs/' 		# popContribFigs_cmapJet/'
		else:
			prePath = savePath
		pathToFile = prePath + figFilename
		plt.savefig(pathToFile, dpi=600)

	## Show figure
	plt.show()
# Heatmaps for CSD data ##    NOTE: SHOULD COMBINE THIS WITH LFP DATA HEATMAP FUNCTIONS IN THE FUTURE!!
def getCSDDataFrames(dataFile, timeRange=None, verbose=0):
	## This function will return data frames of peak and average CSD amplitudes, for picking cell pops
	### dataFile: str 		--> .pkl file to load, with data from the whole recording
	### timeRange: list 	--> e.g. [start, stop]
	### verbose: bool 
		### TO DO: Make evalElecs an argument, so don't have to do all of them, if that's not what we want! 

	# Load .pkl data file...? Is this necessary? Yes if I end up commenting this out for the getCSDdata function! 
	sim.load(dataFile, instantiate=False)
	# Get args for getCSDdata
	dt = sim.cfg.recordStep
	sampr = 1.0/(dt/1000.0) 	# divide by 1000.0 to turn denominator from units of ms to s
	spacing_um = 100 

	# Get all cell pops (cortical)
	thalPops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']
	allPops = list(sim.net.allPops.keys())
	pops = [pop for pop in allPops if pop not in thalPops] 			## exclude thal pops 

	## Get all electrodes 	## SEE TO-DO ABOVE!! (make an if statement)
	evalElecs = []
	evalElecs.extend(list(range(int(sim.net.recXElectrode.nsites))))
	### add 'avg' to electrode list 
	evalElecs.append('avg')


	## get CSD data
	csdPopData = {}

	for pop in pops:  ## do this for ALL THE CELL POPULATIONS -- the pop selection will occur in plotting 
		print('Calculating CSD for pop ' + pop)

		csdPopData[pop] = {}

		popCSDdataFULL_origShape_dict = getCSDdata(dt=dt, sampr=sampr, dataFile=None, pop=pop, spacing_um=spacing_um, outputType=[]) 	# popCSDdataFULL_origShape = getCSDdata(dataFile, pop=pop) 
		popCSDdataFULL_origShape = popCSDdataFULL_origShape_dict['csd']
		popCSDdataFULL = np.transpose(popCSDdataFULL_origShape)	### TRANSPOSE THIS so (20,230000) --> (230000, 20)

		if timeRange is None:
			popCSDdata = popCSDdataFULL.copy()
		else:
			popCSDdata = popCSDdataFULL[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]


		for i, elec in enumerate(evalElecs): ### HOW IS THIS GOING TO WORK WITH CSD VS LFP?? HM -- VAKNIN GOOD ENOUGH TO SOLVE THIS PROBLEM? 
			if elec == 'avg':
				csdPopData[pop]['avg'] = {}

				avgPopData = np.mean(popCSDdata, axis=1)	## csd data (from 1 pop) at each timepoint, averaged over all electrodes

				avgAvgCSD = np.average(avgPopData)
				csdPopData[pop]['avg']['avg'] = avgAvgCSD	## time-average of CSD data (from 1 pop) that has been averaged in space (over all electrodes)

				peakAvgCSD = np.amax(avgPopData)
				csdPopData[pop]['avg']['peak'] = peakAvgCSD	## highest datapoint of all the CSD data (from 1 pop) that has been averaged in space (over all electrodes)

			elif isinstance(elec, Number):
				elecKey = 'elec' + str(elec)
				csdPopData[pop][elecKey] = {}
				csdPopData[pop][elecKey]['avg'] = np.average(popCSDdata[:, elec])	## CSD data from 1 pop, averaged in time, over 1 electrode 
				csdPopData[pop][elecKey]['peak'] = np.amax(popCSDdata[:, elec])		## Maximum CSD value from 1 pop, over time, recorded at 1 electrode 



	#### PEAK CSD AMPLITUDES, DATA FRAME ####
	peakValues = {}
	peakValues['pops'] = []
	peakValues['peakCSD'] = [[] for i in range(len(pops))]  # should be 36? 
	p=0
	for pop in pops:
		peakValues['pops'].append(pop)
		for i, elec in enumerate(evalElecs): 
			if isinstance(elec, Number):
				elecKey = 'elec' + str(elec)
			elif elec == 'avg': 
				elecKey = 'avg'
			peakValues['peakCSD'][p].append(csdPopData[pop][elecKey]['peak'])
		p+=1
	dfPeak = pd.DataFrame(peakValues['peakCSD'], index=pops)


	#### AVERAGE LFP AMPLITUDES, DATA FRAME ####
	avgValues = {}
	avgValues['pops'] = []
	avgValues['avgCSD'] = [[] for i in range(len(pops))]
	q=0
	for pop in pops:
		avgValues['pops'].append(pop)
		for i, elec in enumerate(evalElecs):
			if isinstance(elec, Number):
				elecKey = 'elec' + str(elec)
			elif elec == 'avg':
				elecKey = 'avg'
			avgValues['avgCSD'][q].append(csdPopData[pop][elecKey]['avg'])
		q+=1
	dfAvg = pd.DataFrame(avgValues['avgCSD'], index=pops)


	# return csdPopData
	if verbose:
		return dfPeak, dfAvg, peakValues, avgValues, csdPopData 
	else:
		return dfPeak, dfAvg

## PSD: data and plotting ## 
def getPSDdata(dataFile, inputData, minFreq=1, maxFreq=100, stepFreq=1, transformMethod='morlet'):
	## Look at the power spectral density of a given data set (e.g. CSD, LFP, summed LFP, etc.)
	### dataFile --> .pkl file with simulation recording 
	### inputData --> data to be analyzed 
	### minFreq
	### maxFreq
	### stepFreq
	### transformMethod --> str; options are 'morlet' or 'fft'


	# load simulation .pkl file 
	sim.load(dataFile, instantiate=False)  ## Loading this just to get the sim.cfg.recordStep !! 

	allFreqs = []
	allSignal = []


	# Morlet wavelet transform method
	if transformMethod == 'morlet':

		Fs = int(1000.0/sim.cfg.recordStep)

		morletSpec = MorletSpec(inputData, Fs, freqmin=minFreq, freqmax=maxFreq, freqstep=stepFreq)
		freqs = F = morletSpec.f
		spec = morletSpec.TFR
		signal = np.mean(spec, 1)
		ylabel = 'Power'

	# # FFT transform method
	# elif transformMethod == 'fft':
	# 	Fs = int(1000.0/sim.cfg.recordStep)
	# 	power = mlab.psd(lfpPlot, Fs=Fs, NFFT=NFFT, detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=noverlap, pad_to=None, sides='default', scale_by_freq=None)
	# 	if smooth:
	# 		signal = _smooth1d(10*np.log10(power[0]), smooth)
	# 	else:
	# 		signal = 10*np.log10(power[0])
	# 	freqs = power[1]
	# 	ylabel = 'Power (dB/Hz)'


	allFreqs.append(freqs)
	allSignal.append(signal)

	### To print out frequency w/ max power: 
	maxSignalIndex = np.where(signal==np.amax(signal))
	maxPowerFrequency = freqs[maxSignalIndex]
	print('max power frequency in signal: ' + str(maxPowerFrequency))

	psdData = {'allFreqs': allFreqs, 'allSignal': allSignal}

	return psdData 
def plotPSD(psdData, minFreq=0, maxFreq=100, freqStep=5, lineWidth=1.0, fontSize=12, color='k', figSize=(10,7)):
	### 	----> NOTE: MAKE OVERLAY OPTION POSSIBLE? 
	### This function should plot the PSD data 
	### psdData -->  output of def getPSD()
	### maxFreq --> 
	### lineWidth --> 
	### fontSize --> 
	### color --> Default: 'k' (black)
	### figSize --> Default: (10,7)

	# Get signal & frequency data
	signalList = psdData['allSignal']
	signal = signalList[0]
	freqsList = psdData['allFreqs']
	freqs = freqsList[0]


	plt.figure(figsize=figSize)
	plt.plot(freqs[freqs<maxFreq], signal[freqs<maxFreq], linewidth=lineWidth, color=color)

	# format plot
	plt.xlim([minFreq, maxFreq])
	plt.xticks(np.arange(minFreq, maxFreq, step=freqStep))
	plt.xlabel('Frequency (Hz)', fontsize=fontSize)
	# plt.ylabel(ylabel, fontsize=fontSize)
	plt.tight_layout()
	plt.suptitle('LFP Power Spectral Density', fontsize=fontSize, fontweight='bold')
	plt.show()



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
	timeRange, dataFile, waveletElectrode = getWaveletInfo('delta', based=based, verbose=1)
	wavelet='delta' ### MOVE THESE EVENTUALLY -- BEING USED FOR peakTitle
	# ylim = [1, 40]
	# maxFreq = ylim[1]  ## maxFreq is for use in plotCombinedLFP, for the spectrogram plot 
elif beta:
	timeRange, dataFile, waveletElectrode = getWaveletInfo('beta', based=based, verbose=1)
	wavelet='beta'
	# maxFreq=None
elif alpha:
	timeRange, dataFile, waveletElectrode = getWaveletInfo('alpha', based=based, verbose=1) ## recall timeRange issue (see nb)
	wavelet='alpha'
	# maxFreq=None
elif theta:
	# timeRange, dataFile = getWaveletInfo('theta', based)
	timeRange, dataFile, waveletElectrode = getWaveletInfo('theta', based=based, verbose=1)
	wavelet='theta'
	# maxFreq=None
elif gamma:
	print('Cannot analyze gamma wavelet at this time')




#################################################
####### Evaluating Pops by Frequency Band #######
#################################################

evalWaveletsByBandBool = 0
if evalWaveletsByBandBool:
	basedPkl = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/figs/wavelets/'
	dlmsPklFile = 'v34_batch57_3_4_data_timeRange_0_6_dlms.pkl'
	dfPklFile = 'v34_batch57_3_4_data_timeRange_0_6_df.pkl'   ### AUTOMATE / CONDENSE THIS SOMEHOW... 
	# dlmsData, dfData = evalWaveletsByBand(based=basedPkl, dlmsPklFile=dlmsPklFile, dfPklFile=dfPklFile)
	dfData = evalWaveletsByBand(based=basedPkl, dfPklFile=dfPklFile)




########################
####### PLOTTING #######
########################

#### EVALUATING POPULATIONS TO CHOOSE #### 
## Can plot LFP dataframes (heatmaps) from here as well 
evalPopsBool = 0
if evalPopsBool:
	print('timeRange: ' + str(timeRange))
	print('dataFile: ' + str(dataFile))
	print('channel: ' + str(waveletElectrode))

	# Get data frames for LFP and CSD data
	### dfPeak_LFP, dfAvg_LFP = getDataFrames(dataFile=dataFile, timeRange=timeRange)			# dfPeak, dfAvg, peakValues, avgValues, lfpPopData = getDataFrames(dataFile=dataFile, timeRange=timeRange, verbose=1)
	dfPeak_CSD, dfAvg_CSD = getCSDDataFrames(dataFile=dataFile, timeRange=timeRange)

	# Get the pops with the max contributions 
	maxPopsValues_peakCSD = evalPops(dataFrame=dfPeak_CSD, electrode=waveletElectrode)
	maxPopsValues_avgCSD = evalPops(dataFrame=dfAvg_CSD, electrode=waveletElectrode)


	# maxPopsValues_avgCSD['elec']


###################################
###### COMBINED LFP PLOTTING ######
###################################

plotLFPCombinedData = 0
if plotLFPCombinedData:
	includePops = ['IT3'] #, 'IT5A', 'PT5B']	# placeholder for now <-- will ideally come out of the function above once the pop LFP netpyne issues get resolved! 
	# includePops = includePopsMaxPeak.copy()  ### <-- getting an error about this!! 
	for i in range(len(includePops)):
		pop = includePops[i]
		electrode = [10] #[electrodesMaxPeak[i]]

		print('Plotting LFP spectrogram and timeSeries for ' + pop + ' at electrode ' + str(electrode))

		## Get dictionaries with LFP data for spectrogram and timeSeries plotting  
		LFPSpectOutput = getLFPDataDict(dataFile, pop=pop, timeRange=timeRange, plotType=['spectrogram'], electrode=electrode) 
		LFPtimeSeriesOutput = getLFPDataDict(dataFile, pop=pop, timeRange=timeRange, plotType=['timeSeries'], electrode=electrode) #filtFreq=filtFreq, 

		plotCombinedLFP(timeRange=timeRange, pop=pop, colorDict=colorDict, plotTypes=['timeSeries'], 
			spectDict=LFPSpectOutput, timeSeriesDict=LFPtimeSeriesOutput, figSize=(10,7), colorMap='jet', 
			minFreq=15, maxFreq=None, vmaxContrast=None, electrode=electrode, savePath=None, saveFig=True)

		# plotCombinedLFP(spectDict=LFPSpectOutput, timeSeriesDict=LFPtimeSeriesOutput, timeRange=timeRange, pop=pop, colorDict=colorDict, maxFreq=maxFreq, 
		# 	figSize=(10,7), titleElectrode=electrode, saveFig=0)

		### Get the strongest frequency in the LFP signal ### 
		# maxPowerFrequencyGETLFP = getPSDinfo(dataFile=dataFile, pop=pop, timeRange=timeRange, electrode=electrode, plotPSD=1)


###### COMBINING TOP 3 LFP SIGNAL !! 
summedLFP = 0 #1
if summedLFP: 
	includePops = ['IT3', 'IT5A', 'PT5B']
	popElecDict = {'IT3': 1, 'IT5A': 10, 'PT5B': 11}
	lfpDataTEST_fullElecs = getSumLFP(dataFile=dataFile, pops=includePops, elecs=False, timeRange=timeRange, showFig=True)
	# lfpDataTEST = getSumLFP2(dataFile=dataFile, pops=popElecDict, elecs=True, timeRange=timeRange, showFig=False)	# getSumLFP(dataFile=dataFile, popElecDict=popElecDict, timeRange=timeRange, showFig=False)


######## LFP PSD ########   ---> 	### GET PSD INFO OF SUMMED LFP SIGNAL!!! 
# maxPowerFrequency = getPSDinfo(dataFile=dataFile, pop=None, timeRange=None, electrode=None, lfpData=lfpDataTEST['sum'], plotPSD=True)
lfpPSD = 0 
if lfpPSD: 
	psdData = getPSDdata(dataFile=dataFile, inputData = lfpDataTEST['sum'])
	plotPSD(psdData)



#####################
######## CSD ########
#####################

## Combined Plotting 
plotCSDCombinedData = 1
if plotCSDCombinedData:
	print('Plotting Combined CSD data')
	electrode=[8]
	includePops=['ITS4']#, 'ITP4', 'IT5A'] # ['IT3', 'ITS4', 'ITP4', 'IT5A', 'PT5B']
	for pop in includePops:
		# timeSeriesDict = getCSDdata(dataFile=dataFile, outputType=['timeSeries'], timeRange=None, electrode=None, pop=pop, maxFreq=40)
		# spectDict = getCSDdata(dataFile=dataFile, outputType=['spectrogram'], timeRange=timeRange, electrode=electrode, pop=pop, maxFreq=40)
		thetaOscEventInfo = {'chan': 8, 'minT': 2785.22321038684, 
							'maxT': 3347.9278996316607, 'alignoffset':-3086.95, 'left': 55704, 'right':66958,
							'w2': 3376}
		timeSeriesDict = getCSDdata(dataFile=dataFile, outputType=['timeSeries'], oscEventInfo=thetaOscEventInfo, pop=pop, maxFreq=40)

		# ### time axis testing lines ###
		# minT = 2785.22321038684
		# maxT = 3347.9278996316607
		# alignoffset = -3086.95
		# left = 55704
		# right = 66958
		# w2 = 3376  # w2 = int(w2*0.6)  # 3376 is after this!!  ## This is just for the before / after 

		# # CSD 
		# CSD = timeSeriesDict['csd']  # NOTE: already segmented by relevant channel (8) 

		# # ## Calculate beforeT
		# idx0_before = max(0,left - w2)
		# idx1_before = left 
		# beforeT = (maxT-minT) * (idx1_before - idx0_before) / (right - left + 1)
		# print('beforeT: ' + str(beforeT))
		# sig_before = CSD[chan,idx0_before:idx1_before]

		## Calculate tt for during:
		# chan=8
		# CSD_orig = timeSeriesDict['csd'] ## already segmented by channel, if specified in the timeSeriesDict getCSDData args!! 
		# timeRangeX=[0,6] ## RECALL THAT THIS IS IN SECONDS  !! 
		# dt = 0.05 # (should load this from somewhere, but i know this is dt so keep this for now)
		# dtX = dt/1000.0 ### HAVE TO CHANGE THIS TO SECONDS 
		# CSD = CSD_orig[int(timeRangeX[0]/dtX):int(timeRangeX[1]/dtX)] # [:,int(timeRangeX[0]/dtX):int(timeRangeX[1]/dtX)]
		## ^^ not even necessary!! 
		# sig_during0 = CSD[chan,left:right] # CSD[chan,left:right] 
		# sig_during = CSD[left:right]
		# tt_during = np.linspace(minT,maxT,len(sig_during)) + alignoffset  

		# chan, left, right, minT, maxT, alignoffset


		# # ## Calculate afterT 
		# idx0_after = int(right)
		# idx1_after = min(idx0_after + w2,max(CSD.shape[0],CSD.shape[1]))
		# afterT = (maxT-minT) * (idx1_after - idx0_after) / (right - left + 1)
		# print('afterT: ' + str(afterT))
		# sig_after = CSD[chan,idx0_after:idx1_after]


		plotCombinedCSD(timeSeriesDict=timeSeriesDict, spectDict=None, colorDict=colorDict, pop=pop, electrode=electrode, 
			minFreq=1, maxFreq=40, vmaxContrast=None, colorMap='jet', figSize=(10,7), plotTypes=['timeSeries'], 
			hasBefore=0, hasAfter=1, saveFig=True) # maxFreq=100


## CSD heatmaps
plotCSDheatmaps = 0
if plotCSDheatmaps:
	# figSize=(10,7)
	# figSize=(7,7)  # <-- good for when 4 electrodes 
	# electrodes=None
	# electrodes=[8,9,10]
	dfCSDPeak, dfCSDAvg = getCSDDataFrames(dataFile=dataFile, timeRange=timeRange)
	# peakCSDPlot = plotDataFrames(dfCSDPeak, electrodes=None, pops=ECortPops, title='Peak CSD Values', cbarLabel='CSD', figSize=(10,7), savePath=None, saveFig=False)
	avgCSDPlot = plotDataFrames(dfCSDAvg, electrodes=None, pops=ECortPops, title='Avg CSD Values', cbarLabel='CSD', figSize=(10,7), savePath=None, saveFig=True)
	# maxPopsValues, dfElecSub, dataFrameSubsetElec = evalPops(dataFrame=dfCSDAvg, electrode=waveletElectrode , verbose=1)

## LFP heatmaps for comparison
plotLFPheatmaps = 0
if plotLFPheatmaps:
	dfLFPPeak, dfLFPAvg = getDataFrames(dataFile=dataFile, timeRange=timeRange)
	peakLFPPlot = plotDataFrames(dfLFPPeak, electrodes=None, pops=ECortPops, title='Peak LFP Values', cbarLabel='LFP', figSize=(10,7), savePath=None, saveFig=False)
	avgLFPPlot = plotDataFrames(dfLFPAvg, electrodes=None, pops=ECortPops, title='Avg LFP Values', cbarLabel='LFP', figSize=(10,7), savePath=None, saveFig=False)


## CSD PSD 
csdPSD = 0
if csdPSD:
	csdDataDict = getCSDdata(dataFile=dataFile, outputType=[], timeRange=timeRange, electrode=[9], pop='NGF3') # pop=None, spacing_um=100, minFreq=1, maxFreq=100, stepFreq=1)
	csdData = csdDataDict['csd']
	psdData = getPSDdata(dataFile=dataFile, inputData=csdData, minFreq=1, maxFreq=50, stepFreq=0.5)
	plotPSD(psdData)


##########################################
###### COMBINED SPIKE DATA PLOTTING ######
##########################################

plotCombinedSpikeData = 0	# includePopsMaxPeak.copy()		# ['PT5B']	#['IT3', 'IT5A', 'PT5B']	# placeholder for now <-- will ideally come out of the function above once the pop LFP netpyne issues get resolved! 
if plotCombinedSpikeData:
	includePops=['ITS4', 'ITP4', 'IT5A']# ['IT3', 'ITS4', 'IT5A']
	for pop in includePops:
		print('Plotting spike data for ' + pop)

		## Get dictionaries with spiking data for spectrogram and histogram plotting 
		spikeSpectDict = getSpikeData(dataFile, graphType='spect', pop=pop, timeRange=timeRange)
		histDict = getSpikeData(dataFile, graphType='hist', pop=pop, timeRange=timeRange)

		## Then call plotting function 
		plotCombinedSpike(spectDict=spikeSpectDict, histDict=histDict, timeRange=timeRange, colorDict=colorDict,
		pop=pop, figSize=(10,7), colorMap='jet', vmaxContrast=2.5, maxFreq=None, saveFig=1)

		# plotCombinedSpike(timeRange=timeRange, pop=pop, colorDict=colorDict, plotTypes=['histogram'], 
		# 	spectDict=None, histDict=histDict, figSize=(10,7), colorMap='jet', minFreq=10, maxFreq=65, 
		# 	vmaxContrast=None, savePath=None, saveFig=True)


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







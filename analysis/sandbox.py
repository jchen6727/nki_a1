from netpyne import sim
import numpy as np
#from simDataAnalysis import *

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
## trying peakF calculations from load.py
from loadSelect import * 

###### THESE ARE "CURRENTLY UNUSED" FUNCTIONS COMING FROM simDataAnalysis.py
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


##### CSD ###### 





###### CONDENSING plotLFP and getLFPData function!!! #######
### LFP ### 
def getLFPData(pop=None, timeRange=None, electrodes=['avg', 'all'], plots=['timeSeries', 'spectrogram', 'PSD'], inputLFP=None, NFFT=256, noverlap=128, nperseg=256, 
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
		**Default:** ``['timeSeries', 'spectrogram']`` <-- added 'PSD' (EYG, 2/10/2022)

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
	electrodes=None
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

	# Power Spectral Density ------------------------------
	if 'PSD' in plots:
		allFreqs = []
		allSignal = []
		data['allFreqs'] = allFreqs
		data['allSignal'] = allSignal

		if electrodes is None: #### THIS IS FOR PSD INFO FOR SUMMED LFP SIGNAL  !!! 
			lfpPlot = lfp
			# Morlet wavelet transform method
			if transformMethod == 'morlet':
				# from ..support.morlet import MorletSpec, index2ms

				Fs = int(1000.0/sim.cfg.recordStep)

				#t_spec = np.linspace(0, index2ms(len(lfpPlot), Fs), len(lfpPlot))
				morletSpec = MorletSpec(lfpPlot, Fs, freqmin=minFreq, freqmax=maxFreq, freqstep=stepFreq)
				freqs = F = morletSpec.f
				spec = morletSpec.TFR
				signal = np.mean(spec, 1)
				ylabel = 'Power'

			# FFT transform method
			elif transformMethod == 'fft':
				Fs = int(1000.0/sim.cfg.recordStep)
				power = mlab.psd(lfpPlot, Fs=Fs, NFFT=NFFT, detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=noverlap, pad_to=None, sides='default', scale_by_freq=None)

				if smooth:
					signal = _smooth1d(10*np.log10(power[0]), smooth)
				else:
					signal = 10*np.log10(power[0])
				freqs = power[1]
				ylabel = 'Power (dB/Hz)'

			allFreqs.append(freqs)
			allSignal.append(signal)

		else:
			for i,elec in enumerate(electrodes):
				if elec == 'avg':
					lfpPlot = np.mean(lfp, axis=1)
				elif isinstance(elec, Number) and (inputLFP is not None or elec <= sim.net.recXElectrode.nsites):
					lfpPlot = lfp[:, elec]

				# Morlet wavelet transform method
				if transformMethod == 'morlet':
					# from ..support.morlet import MorletSpec, index2ms

					Fs = int(1000.0/sim.cfg.recordStep)

					#t_spec = np.linspace(0, index2ms(len(lfpPlot), Fs), len(lfpPlot))
					morletSpec = MorletSpec(lfpPlot, Fs, freqmin=minFreq, freqmax=maxFreq, freqstep=stepFreq)
					freqs = F = morletSpec.f
					spec = morletSpec.TFR
					signal = np.mean(spec, 1)
					ylabel = 'Power'

				# FFT transform method
				elif transformMethod == 'fft':
					Fs = int(1000.0/sim.cfg.recordStep)
					power = mlab.psd(lfpPlot, Fs=Fs, NFFT=NFFT, detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=noverlap, pad_to=None, sides='default', scale_by_freq=None)

					if smooth:
						signal = _smooth1d(10*np.log10(power[0]), smooth)
					else:
						signal = 10*np.log10(power[0])
					freqs = power[1]
					ylabel = 'Power (dB/Hz)'

				allFreqs.append(freqs)
				allSignal.append(signal)


		normPSD=0 ## THIS IS AN ARG I BELIEVE (in plotLFP) -- PERHAPS DO THE SAME HERE...? 
		if normPSD:
			vmax = np.max(allSignal)
			for i, s in enumerate(allSignal):
				allSignal[i] = allSignal[i]/vmax




	outputData = {'LFP': lfp, 'lfpPlot': lfpPlot, 'electrodes': electrodes, 'timeRange': timeRange}
	### Added lfpPlot to this, because that usually has the post-processed electrode-based lfp data

	if 'timeSeries' in plots:
		outputData.update({'t': t})

	if 'spectrogram' in plots:
		outputData.update({'spec': spec, 't': t_spec*1000.0, 'freqs': f[f<=maxFreq]})

	if 'PSD' in plots:
		outputData.update({'allFreqs': allFreqs, 'allSignal': allSignal})


	return outputData










###### DEBUGGING LFP HEATMAP #######

dataFile = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/A1_v34_batch65_v34_batch65_0_0_data.pkl'
timeRange = [1480, 2520]

sim.load(dataFile, instantiate=False)

# total LFP 
totalLFPdata = sim.allSimData['LFP']

# 'IT2' pop data 
pop = 'IT2'
popLFPdata = np.array(sim.allSimData['LFPPops'][pop])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]

## numpy mean vs avg. 
# mean_axis1 = np.mean(popLFPdata, axis=1, dtype='float64')
# avg_axis1 = np.average(popLFPdata, axis=1)
avgPopData = np.mean(popLFPdata, axis=1)


# ## 
elec=1
elecData = popLFPdata[:, elec]
absData = np.absolute(elecData).max()
negData = -elecData

# avgElecData = np.average(popLFPdata[:, elec])
# avgElecData2 = np.mean(popLFPdata[:, elec])
# avgElecData3 = np.average(elecData, axis=0)
# avgElecData4 = np.mean(elecData, axis=0)
# ^^ all equivalent!! 

########################################################################



###### TESTING NEW LFP .pkl FILES ######
### batch 57 ### batch 65 ### batch 67
### 3_2      ### 1_1      ### 0_0
### 3_3      ### 2_2      ### 1_1 
### 3_4      ### 0_0      ### <-- worked in batch65 so no need! 

# testInfo = {
# 'seed_3_2': {'batch57': 'v34_batch57_3_2_data.pkl', 'batch65': 'batch65_old/A1_v34_batch65_v34_batch65_1_1_data.pkl', 'batch67': 'A1_v34_batch67_v34_batch67_0_0_data.pkl'}, 
# 'seed_3_3': {'batch57': 'v34_batch57_3_3_data.pkl', 'batch65': 'batch65_old/A1_v34_batch65_v34_batch65_2_2_data.pkl', 'batch67': 'A1_v34_batch67_v34_batch67_1_1_data.pkl'}, 
# 'seed_3_4': {'batch57': 'v34_batch57_3_4_data.pkl', 'batch65': 'batch65_old/A1_v34_batch65_v34_batch65_0_0_data.pkl', 'batch67': 'A1_v34_batch65_v34_batch65_0_0_data.pkl'}}

# dataFile57 = based + testInfo[seed]['batch57']
# dataFile65 = based + testInfo[seed]['batch65']
# dataFile67 = based + testInfo[seed]['batch67']

## for seed 3_2 --> batch57 == batch65 =/= batch67
## for seed 3_3 --> batch57 == batch65 =/= batch67
## for seed 3_4 --> batch57 =/= batch65 == batch67

####
# remixInfo = {
# 'seed_3_2': {'batch57': 'v34_batch57_3_2_data.pkl', 'batch65': 'batch65_old/A1_v34_batch65_v34_batch65_1_1_data.pkl', 'batch67': 'A1_v34_batch67_v34_batch67_0_0_data.pkl'}, 
# 'seed_3_3': {'batch57': 'v34_batch57_3_3_data.pkl', 'batch65': 'batch65_old/A1_v34_batch65_v34_batch65_2_2_data.pkl', 'batch67': 'A1_v34_batch67_v34_batch67_1_1_data.pkl'}, 
# 'seed_3_4': {'batch57': 'v34_batch57_3_4_data.pkl', 'batch65': 'batch65_old/A1_v34_batch65_v34_batch65_0_0_data.pkl', 'batch67': 'A1_v34_batch65_v34_batch65_0_0_data.pkl'}}

# based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/' 

# seed = 'seed_3_2'

# dataFile57 = based + remixInfo[seed]['batch57']
# dataFile65 = based + remixInfo[seed]['batch65']
# dataFile67 = based + remixInfo[seed]['batch67']

# sim.load(dataFile57, instantiate=False)
# batch57_LFP = sim.allSimData['LFP']
# batch57_timestep = sim.cfg.recordStep

# sim.load(dataFile65, instantiate=False)
# batch65_LFP = sim.allSimData['LFP']
# batch65_timestep = sim.cfg.recordStep

# sim.load(dataFile67, instantiate=False)
# batch67_LFP = sim.allSimData['LFP']
# batch67_timestep = sim.cfg.recordStep

# batch57_LFP == batch65_LFP
# batch65_LFP == batch67_LFP
# batch57_LFP == batch67_LFP


### seed 3_4 --> changing batch65 to 1_1 and 2_2 made everything false
### seed 3_2 --> 


#####
# newFiles = ['A1_v34_batch67_v34_batch67_0_0_data.pkl', 'A1_v34_batch67_v34_batch67_1_1_data.pkl', 'A1_v34_batch65_v34_batch65_0_0_data.pkl']

# based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/' 

# lfpData = {}

# for dFile in newFiles:
# 	sim.load(based + dFile, instantiate=False)
# 	dataName=dFile[19:-9]
# 	lfpData[dataName] = sim.allSimData['LFP']











# for seed in testInfo.keys():
# 	dataFile57 = based + testInfo[seed]['batch57']
# 	dataFile65 = based + testInfo[seed]['batch65']
# 	dataFile67 = based + testInfo[seed]['batch67']

# 	sim.load(dataFile57, instantiate=False)
# 	batch57_LFP = sim.allSimData['LFP']
# 	batch57_timestep = sim.cfg.recordStep

# 	sim.load(dataFile65, instantiate=False)
# 	batch65_LFP = sim.allSimData['LFP']
# 	batch65_timestep = sim.cfg.recordStep

# 	sim.load(dataFile67, instantiate=False)
# 	batch67_LFP = sim.allSimData['LFP']
# 	batch67_timestep = sim.cfg.recordStep

# 	if batch57_timestep == batch65_timestep == batch67_timestep:
# 		print('FOR ' + str(seed) + ': all have matching timeStep')
# 		if batch57_LFP == batch65_LFP == batch67_LFP:
# 			print('FOR ' + str(seed) + ': all have matching LFP!!!')
# 		else:
# 			print('FOR ' + str(seed) + ': all DO NOT have matching LFP!!!')
# 	else:
# 		print('FOR ' + str(seed) + ': batches DO NOT have matching timeSteps!!!!')




######################## DEBUGGING LFP HEAT PLOTS ########################

### OLD INFO ###
# waveletInfo = {'delta': {'dataFile': 'A1_v34_batch65_v34_batch65_0_0_data.pkl', 'timeRange': [1480, 2520]},
# 	'beta': {'dataFile': 'A1_v34_batch65_v34_batch65_1_1_data.pkl', 'timeRange': [456, 572]}, 
# 	'alpha': {'dataFile': 'A1_v34_batch65_v34_batch65_1_1_data.pkl', 'timeRange': [3111, 3325]}, 
# 	'theta': {'dataFile': 'A1_v34_batch65_v34_batch65_2_2_data.pkl', 'timeRange': [2785, 3350]}, 
# 	'test_file': {'dataFile': 'A1_v34_batch65_v34_batch65_0_0_data_NEW.pkl', 'timeRange': [1480, 2520]}}
################

# waveletInfo = {'delta': {'dataFile': 'A1_v34_batch65_v34_batch65_0_0_data.pkl', 'timeRange': [1480, 2520], 'channel': 14},
# 'beta': {'dataFile': 'A1_v34_batch67_v34_batch67_0_0_data.pkl', 'timeRange': [456, 572], 'channel': 14}, 
# 'alpha': {'dataFile': 'A1_v34_batch67_v34_batch67_0_0_data.pkl', 'timeRange': [3111, 3325], 'channel': 9}, 
# 'theta': {'dataFile': 'A1_v34_batch67_v34_batch67_1_1_data.pkl', 'timeRange': [2785, 3350], 'channel': 8}}


# freqBand = 'theta'
# based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/' 

# dataFile = based + waveletInfo[freqBand]['dataFile']
# timeRange = waveletInfo[freqBand]['timeRange']

# print('timeRange = ' + str(timeRange))
# print('dataFile = ' + str(dataFile))

# sim.load(dataFile, instantiate=False)



# ### thal pops
# thalPops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']
# allPops = list(sim.net.allPops.keys())
# pops = [pop for pop in allPops if pop not in thalPops] 			## exclude thal pops 
# ## output --> >>> pops
# # ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'CT5B', 'PT5B', 'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6']

# ### play with lfp data 
# timePoint = 10000			# pick a timepoint for convenience 

# lfpTotal = sim.allSimData['LFP'][timePoint]   # <-- list of 20 (total lfp amplitudes at each electrode for this timepoint)

# popLFPLists = []  # <-- list of 36 lists (1 for each pop); each list is length 20 (lfp amplitudes at each electrode for this timepoint)
# for pop in pops:
# 	lfpSublist = sim.allSimData['LFPPops'][pop][timePoint] 
# 	popLFPLists.append(lfpSublist)
# lfpPopTotal = sum(popLFPLists)


### NOW CHECK --> lfpPopTotal == lfpTotal



########################
#### pop-specific 
# testPop = 'NGF1'

# popLFPdata_TOTAL = sim.allSimData['LFPPops'][testPop]
# print('dims of popLFPdata_TOTAL: ' + str(popLFPdata_TOTAL.shape)) 			 # (230000, 20)


# popLFPdata = np.array(sim.allSimData['LFPPops'][testPop])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]
# print('dims of popLFPdata: ' + str(popLFPdata.shape))
# # dims of new popLFPdata: (20800, 20)

# totalAvg = np.average(popLFPdata)
# print('totalAvg: ' + str(totalAvg))
# totalPeak = np.amax(popLFPdata)
# print('totalPeak: ' + str(totalPeak))


## Checking pop to total comparison
### timepoints should be a for loop I suppose
### lfp totals


#popLFPSum = 0
#for pop in pops:
	#lfpSum = sim.allSimData['LFPPops'][pop][timePoint] #sum(sim.allSimData['LFPPops'][pop][0]) 	# np.sum(sim.allSimData['LFPPops'][pop][0,:])
	#print('lfp amplitude over all electrodes for ' + pop + ' at timepoint ' + str(0) + ': ' + str(lfpSum))
	#popLFPSum+=lfpSum

# print(str(sum(lfpTotal[0])))  #[0])))
# print(str(totalLFPSum))

#lfpTotal[0] == totalLFPSum



### calculate avg and peak 
# for pop in pops:
# 	lfpPopData[pop] = {}

# 	popLFPdata = np.array(sim.allSimData['LFPPops'][pop])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]

# 	lfpPopData[pop]['totalLFP'] = popLFPdata
# 	lfpPopData[pop]['totalAvg'] = np.average(popLFPdata)
# 	lfpPopData[pop]['totalPeak'] = np.amax(popLFPdata)


# 	for elec in evalElecs:
# 		elecKey = 'elec' + str(elec)
# 		lfpPopData[pop][elecKey] = {}
# 		lfpPopData[pop][elecKey]['LFP'] = popLFPdata[:, elec]
# 		lfpPopData[pop][elecKey]['avg'] = np.average(popLFPdata[:, elec])
# 		lfpPopData[pop][elecKey]['peak'] = np.amax(popLFPdata[:, elec])




###############################################################
# pathToFiles = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/'

# ### batch 57 
# batch57Files = ['v34_batch57_3_4_data', 'v34_batch57_3_3_data', 'v34_batch57_3_2_data']

# ### batch 65
# batch65Prefix = 'A1_v34_batch65_'
# batch65Files = ['v34_batch65_0_0_data', 'v34_batch65_1_1_data', 'v34_batch65_2_2_data']
# #batch65Files = ['A1_v34_batch65_v34_batch65_0_0_data.pkl', 'A1_v34_batch65_v34_batch65_0_0_data.pkl', 'A1_v34_batch65_v34_batch65_0_0_data.pkl']



# ### LOAD THE DATA #### 
# batch57_data = {}
# batch57_LFP = {}
# for file in batch57Files:
# 	fullFile = pathToFiles + file + '.pkl'
# 	sim.load(fullFile, instantiate=False)
# 	fileKey = file.split('v34_')[1].split('_data')[0].split('batch57_')[1]
# 	batch57_data[fileKey] = sim.allSimData
# 	batch57_LFP[fileKey] = sim.allSimData['LFP']



# batch65_data = {}
# batch65_LFP = {}
# for file in batch65Files:
# 	fullFile = pathToFiles + batch65Prefix + file + '.pkl'
# 	sim.load(fullFile, instantiate=False)
# 	fileKey = file.split('v34_')[1].split('_data')[0].split('batch65_')[1]
# 	batch65_data[fileKey] = sim.allSimData
# 	batch65_LFP[fileKey] = sim.allSimData['LFP']



#### NOW COMPARE LFP #### 

##### THAL POP DEBUGGING #####
# pops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'CT5B', 'PT5B', 'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']
# thalPops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']

# pops_new = [pop for pop in pops if pop not in thalPops]

# for pop in pops:
# 	if any(elem == pop for elem in thalPops):
# 		print('redundant pop: ' + pop)
		#pops.remove(pop)



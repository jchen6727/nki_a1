from netpyne import sim
import numpy as np
#from simDataAnalysis import *





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



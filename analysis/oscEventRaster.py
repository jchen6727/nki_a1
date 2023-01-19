from netpyne import sim 
from matplotlib import pyplot as plt 
import os 
import numpy as np


basedir = '../data/oscEventRasters/pklFiles/'
# fn = '../data/oscEventRasters/pklFiles/A1_v34_batch65_v34_batch65_0_0_data.pkl'


# allFiles = os.listdir(basedir)

# pklFiles = []
# for file in allFiles:
# 	if '_data.pkl' in file:
# 		pklFiles.append(file)


waveletInfo = {
	'Delta': {'dataFile': 'A1_v34_batch65_v34_batch65_0_0_data.pkl', 'timeRange': [1484.9, 2512.2], 
				'timeBuffer': 308.1, 'dataFile2': 'v34_batch57_3_4_data.pkl', 
				'xtick_locs': [1326.1, 1526.1, 1726.1, 1926.1, 2126.1, 2326.1, 2526.1, 2726.1], 'xtick_labels': [-600, -400, -200, 0, 200, 400, 600, 800]},	# 'timeRange': [1484, 2520] #'channel': 14}, # [1484.9123743729062, 2512.2209353489225]
	'Beta': {'dataFile': 'A1_v34_batch67_v34_batch67_0_0_data.pkl',	'timeRange': [455.8, 571.4], 
				'timeBuffer': 34.6, 'dataFile2': 'v34_batch57_3_2_data.pkl',
				'xtick_locs': [430.3, 455.3, 480.3, 505.3, 530.3, 555.3, 580.3, 605.3], 'xtick_labels': [-75, -50, -25, 0, 25, 50, 75, 100]},	#	'timeRange': [456, 572] 	#'channel': 14}, # [455.80379839663993, 571.4047617460291]
	'Alpha': {'dataFile': 'A1_v34_batch67_v34_batch67_0_0_data.pkl', 'timeRange': [3111, 3324.7], 
				'timeBuffer': 64.1, 'dataFile2': 'v34_batch57_3_2_data.pkl',
				'xtick_locs': [3056.25, 3106.25, 3156.25, 3206.25, 3256.25, 3306.25, 3356.25], 'xtick_labels': [-150, -100, -50, 0, 50, 100, 150]}, # 'timeRange': [3111, 3325]	# NOTE: don't forget 6_11 timeRange	#'channel': 9}, # [3111.0311106222125, 3324.733247664953]
	'Theta': {'dataFile': 'A1_v34_batch67_v34_batch67_1_1_data.pkl', 'timeRange': [2785, 3347.9], 
				'timeBuffer': 168.8, 'dataFile2': 'v34_batch57_3_3_data.pkl',
				'xtick_locs': [2686.95, 2886.95, 3086.95, 3286.95, 3486.95], 'xtick_labels': [-400, -200, 0, 200, 400]}, 	# 'timeRange': [2785, 3350] #'channel': 8}} # [2785.22321038684, 3347.9278996316607]
	'Gamma': {'dataFile': 'v34_batch57_4_4_data.pkl', 'timeRange': [3895.6, 3957.2], 
				'timeBuffer': 18.4, 'dataFile2': 'v34_batch57_4_4_data.pkl',
				'xtick_locs': [3878.3, 3898.3, 3918.3, 3938.3, 3958.3], 'xtick_labels':[-40, -20, 0, 20, 40]}}	# 'timeRange': [3890, 3960] # gamma timeRange: [3895.632463874398, 3957.13297638294]


freqBands = ['Delta']	#list(waveletInfo.keys()) # ['Theta']

batch57 = 1
saveFigs = 0
markerType = 'o'
layerLines = 1

if batch57: 
	for band in freqBands:
		fn = basedir + waveletInfo[band]['dataFile2']
		sim.load(fn, instantiate=False)
		print('Loaded ' + fn)

		orderBy = ['pop']


		timeRange = waveletInfo[band]['timeRange']
		timeBuffer = waveletInfo[band]['timeBuffer']
		timeRangeBuffered = [0,0]
		timeRangeBuffered[0] = timeRange[0] - timeBuffer
		timeRangeBuffered[1] = timeRange[1] + timeBuffer
		print('new timeRange: ' + str(timeRangeBuffered))


		## PLOT RASTER ## 
		sim.analysis.plotRaster(include=['allCells'], timeRange=timeRangeBuffered, labels=False,
			popRates=False, orderInverse=True, lw=0, markerSize=12, marker=markerType,  
			showFig=0, saveFig=0, orderBy=orderBy, figSize=(6.4, 4.8))		# timeRange=timeRange # labels='legend'

		## Set x ticks and x labels 
		plt.xlabel('Time (ms)', fontsize=12)
		x_tick_numbers = waveletInfo[band]['xtick_locs']
		x_tick_labels = waveletInfo[band]['xtick_labels']
		plt.xticks(x_tick_numbers, x_tick_labels) # add in fontsize 

		## Set title of plot 
		plt.title(band + ', Oscillation Event Raster', fontsize=14)


		# vertical demarcation lines to indicate beginning and end of oscillation event #
		rasterData = sim.analysis.prepareRaster(timeRange=timeRangeBuffered, maxSpikes=1e8, orderBy=['pop'],popRates=True)
		spkInds = rasterData['spkInds']
		print('min spkInds: ' + str(min(spkInds)))
		print('max spkInds: ' + str(max(spkInds)))

		plt.vlines(timeRange[0], min(spkInds), max(spkInds), colors='blue', linestyles='dashed')
		plt.vlines(timeRange[1], min(spkInds), max(spkInds), colors='blue', linestyles='dashed')


		### LAYER BOUNDARIES #### 
		if layerLines:
			popLabels = rasterData['popLabels']
			popNumCells = rasterData['popNumCells']

			indPop = []
			for popLabel, popNumCell in zip(popLabels, popNumCells):
				indPop.extend(int(popNumCell) * [popLabel])


			L1 = [0,100] 			# NGF1
			# L2 = [0,0]				#  'IT2'  --> 'NGF2'
			# L3 = [0,0] 				#  'IT3'  ---> 'NGF3'
			L4 = [0,0]				#  'ITP4' --> 'NGF4'
			L5A = [0,0]				#  'IT5A' --> 'NGF5A'
			# L5B = [0,0] 			#  'IT5B'  --> 'NGF5B
			# L6 =  [0,0]				#  'IT6' --> 
			Thal = [12000, 12907]   # TC --> TIM 


			listL1 = []

			# listL2_TOP = []
			# listL2_BOTTOM = []

			# listL3_TOP = []
			# listL3_BOTTOM = []

			listL4_TOP = []
			listL4_BOTTOM = []

			listL5A_TOP = []
			listL5A_BOTTOM = []

			# listL5B_TOP = []
			# listL5B_BOTTOM = []

			# listL6_TOP = []
			# listL6_BOTTOM = []

			listThal_TOP = []
			listThal_BOTTOM = []

			for spk in spkInds:
				if indPop[spk] == 'NGF1':
					listL1.append(spk)

				# elif indPop[spk] == 'IT2':
				# 	listL2_TOP.append(spk)
				# elif indPop[spk] == 'NGF2':
				# 	listL2_BOTTOM.append(spk)

				# elif indPop[spk] == 'IT3':
				# 	listL3_TOP.append(spk)
				# elif indPop[spk] == 'NGF3':
				# 	listL3_BOTTOM.append(spk)

				elif indPop[spk] == 'ITP4':
					listL4_TOP.append(spk)
				elif indPop[spk] == 'NGF4':
					listL4_BOTTOM.append(spk)

				elif indPop[spk] == 'IT5A':
					listL5A_TOP.append(spk)
				# elif indPop[spk] == 'NGF5A':
				# 	listL5A_BOTTOM.append(spk)

				# elif indPop[spk] == 'IT5B':
				# 	listL5B_TOP.append(spk)
				# elif indPop[spk] == 'NGF5B':
				# 	listL5B_BOTTOM.append(spk)

				# elif indPop[spk] == 'IT6':
				# 	listL6_TOP.append(spk)
				# elif indPop[spk] == 'NGF6':
				# 	listL6_BOTTOM.append(spk)

				elif indPop[spk] == 'TC':
					listThal_TOP.append(spk)
				# elif indPop[spk] == 'TIM':
				# 	listThal_BOTTOM.append(spk)

			L1[0] = min(listL1)
			# L1[1] = max(listL1)
			plt.axhline(0, color='black', linestyle=":")


			L4[0] = min(listL4_TOP)
			#L4[1] = max(listL4_BOTTOM)
			plt.axhline(L4[0], color='black', linestyle=":")


			L5A[0] = min(listL5A_TOP)
			#L5A[1] = max(listL5A_BOTTOM)
			plt.axhline(L5A[0], color='black', linestyle=":")

			Thal[0] = min(listThal_TOP)
			#Thal[1] = max(listThal_BOTTOM)
			plt.axhline(Thal[0], color='black', linestyle=":")



		## Setting y-axis labels according to layer boundaries ## 
		plt.ylabel('Neuron ID', fontsize=12) # Neurons (ordered by NCD within each pop)')
		# plt.ylabel('')
		plt.yticks(fontsize=10)


		## Print out plotting confirmed line 
		print(band + ' raster: PLOTTED')


		## SAVING ##
		if saveFigs:
			rasterFilename = band + '_dataFile2_' + waveletInfo[band]['dataFile2'] 
			rasterFile = basedir + rasterFilename + '_RASTER.png'
			plt.savefig(rasterFile, dpi=300)
			print('RASTER SAVED')
		else:
			plt.show()



		# ### USING SIM.PLOTTING.PLOTRASTER ###
		# rasterData = sim.analysis.prepareRaster(timeRange=timeRangeBuffered, maxSpikes=1e8, orderBy=['pop'],popRates=True)

		# plt.rcParams={'figsize':(6.4, 4.8), 'dpi': 600}
		# print(plt.rcParams)


		# sim.plotting.plotRaster(rasterData=rasterData, orderBy=['pop'], popRates=False,
		# 						timeRange=timeRangeBuffered, orderInverse=True, marker=markerType, 
		# 						markerSize=12, ylabel='Neuron ID', legend=False, showFig=1, rcParams=plt.rcParams) # figSize=(6.4, 4.8), 

		# # import matplotlib; print(matplotlib.rcParams)
		# # plt.xlabel('Time (ms)', fontsize=12)
		# # plt.ylabel('Neuron ID', fontsize=12) # Neurons (ordered by NCD within each pop)')

		# # plt.xticks(fontsize=10)
		# # plt.yticks(fontsize=10)

		# # plt.title(band + ', Oscillation Event Raster', fontsize=14)

		# # ### f = plt.figure()






# else:
# 	for band in freqBands:
# 		fn = basedir + waveletInfo[band]['dataFile']
# 		sim.load(fn, instantiate=False)
# 		print('Loaded ' + fn)

# 		orderBy = ['pop']


# 		timeRange = waveletInfo[band]['timeRange'] 			#[0,sim.cfg.duration]#[500,sim.cfg.duration]#[500,10000]

# 		sim.analysis.plotRaster(include=['allCells'], timeRange=timeRange,
# 			popRates=False, orderInverse=True, lw=0, markerSize=12, marker='.',  
# 			showFig=0, saveFig=0, orderBy=orderBy)		# labels='legend' # figSize=(9*0.95, 13*0.9)

# 		plt.ylabel('Neuron ID') # Neurons (ordered by NCD within each pop)'
# 		plt.xlabel('Time (ms)')

# 		plt.title('dataFile !')


# 		print(band + ' raster: PLOTTED')


# 		## SAVING ## 
# 		if saveFigs:
# 			rasterFilename = band + '_dataFile_' + waveletInfo[band]['dataFile']
# 			rasterFile = basedir + rasterFilename + '_RASTER.png'
# 			plt.savefig(rasterFile, dpi=300)
# 			print('RASTER SAVED')
# 		else:
# 			plt.show()



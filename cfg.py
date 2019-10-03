"""
cfg.py 

Simulation configuration for A1 model (using NetPyNE)
This file has sim configs as well as specification for parameterized values in netParams.py 

Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com
"""


from netpyne import specs
import pickle

cfg = specs.SimConfig()

#------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Run parameters
#------------------------------------------------------------------------------
cfg.duration = 0.5*1e3			## Duration of the sim, in ms -- value from M1 cfg.py 
cfg.dt = 0.05                   ## Internal Integration Time Step -- value from M1 cfg.py 
cfg.verbose = 0      	## Show detailed messages


#------------------------------------------------------------------------------
# Recording 
#------------------------------------------------------------------------------
allpops = ['NGF1','IT2','PV2','SOM2','VIP2','NGF2','IT3','SOM3','PV3','VIP3','NGF4','ITP4','ITS4','VIP4','IT5A','PV5A','SOM5A','VIP5A','NGF5A','IT5B','PT5B','PV5B','SOM5B','VIP5B','NGF5B','IT6','CT6','PV6','SOM6','VIP6','NGF6']

cfg.recordTraces = {'V_soma': {'sec':'soma', 'loc':0.5, 'var':'v'}}  ## Dict with traces to record -- taken from M1 cfg.py 
cfg.recordStim = False			## Seen in M1 cfg.py
cfg.recordTime = False  		## SEen in M1 cfg.py 
cfg.recordStep = 0.1            ## Step size (in ms) to save data -- value from M1 cfg.py 

cfg.recordLFP = [[150,250,150], [150,500,150], [150,750,150], [150,1000,150], [150,1250,150], [150,1500,150], [150,1750,150]]

#------------------------------------------------------------------------------
# Saving
#------------------------------------------------------------------------------

# cfg.filename =                	## Set file output name
cfg.savePickle = True         	## Save pkl file
cfg.saveJson = True           	## Save json file
cfg.saveDataInclude = ['simData', 'simConfig', 'netParams'] ## seen in M1 cfg.py (line 58)
cfg.backupCfgFile = None 		## Seen in M1 cfg.py 
cfg.gatherOnlySimData = False	## Seen in M1 cfg.py 
cfg.saveCellSecs = True			## Seen in M1 cfg.py 
cfg.saveCellConns = True		## Seen in M1 cfg.py 

#------------------------------------------------------------------------------
# Analysis and plotting 
#------------------------------------------------------------------------------

cfg.analysis['plotTraces'] = {'include': [(pop,0) for pop in allpops]} 		## Seen in M1 cfg.py (line 68) 
cfg.analysis['plotRaster'] = {'include': allpops, 'saveFig': True, 'showFig': True,'labels': 'overlay', 'popRates': True, 'orderInverse': True, 'timeRange': [0,cfg.duration], 'figSize': (12,12), 'lw': 0.3, 'markerSize':10, 'marker': '.', 'dpi': 300}      	## Plot a raster
cfg.analysis['plotLFP'] = {'include': ['timeSeries', 'PSD', 'spectrogram']}
#cfg.analysis['plot2Dnet'] = True      	## Plot 2D visualization of cell positions & connections 

#------------------------------------------------------------------------------
# Cells
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Synapses
#------------------------------------------------------------------------------
cfg.AMPATau2Factor = 1.0
cfg.synWeightFractionEE = [0.5, 0.5] # E->E AMPA to NMDA ratio
cfg.synWeightFractionEI = [0.5, 0.5] # E->I AMPA to NMDA ratio
cfg.synWeightFractionSOME = [0.9, 0.1] # SOM -> E GABAASlow to GABAB ratio
cfg.synWeightFractionNGF = [0.5, 0.5] # NGF GABAA to GABAB ratio


#------------------------------------------------------------------------------
# Network 
#------------------------------------------------------------------------------
## These values taken from M1 cfg.py (https://github.com/Neurosim-lab/netpyne/blob/development/examples/M1detailed/cfg.py)
cfg.singleCellPops = False
cfg.scale = 1.0     # Is this what should be used? 
cfg.sizeY = 2000.0 #1350.0 in M1_detailed # should this be set to 2000 since that is the full height of the column? 
cfg.sizeX = 400.0 # This may change depending on electrode radius 
cfg.sizeZ = 400.0
cfg.scaleDensity = 0.025 # Should be 1.0 unless need lower cell density for test simulation or visualization


#------------------------------------------------------------------------------
# Connectivity
#------------------------------------------------------------------------------
cfg.synWeightFractionEE = [0.5, 0.5] # E->E AMPA to NMDA ratio
cfg.synWeightFractionEI = [0.5, 0.5] # E->I AMPA to NMDA ratio
cfg.synWeightFractionIE = [0.9, 0.1]  # SOM -> E GABAASlow to GABAB ratio (update this)
cfg.synWeightFractionII = [0.9, 0.1]  # SOM -> E GABAASlow to GABAB ratio (update this)
cfg.synsPerConnWeightFactor = 5  # factor to multiply all weights for given 1 syn contact per conn (instead of ~5 -- avg in cortex)

cfg.addConn = 1
cfg.EEGain = 1.0
cfg.EIGain = 1.0
cfg.IEGain = 1.0
cfg.IIGain = 1.0

## I->E/I layer weights (L2/3+4, L5, L6)
cfg.IEweights = [1.0, 1.0, 1.0]
cfg.IIweights = [1.0, 1.0, 1.0]

#------------------------------------------------------------------------------
# Background inputs
#------------------------------------------------------------------------------
cfg.addBkgConn = 1
cfg.noiseBkg = 1.0  # firing rate random noise
cfg.delayBkg = 5.0  # (ms)
cfg.startBkg = 0  # start at 0 ms
cfg.weightBkg = {'E': 0.5, 'I': 0.5}  # corresponds to unitary connection somatic EPSP (mV)
cfg.rateBkg = {'E': 100, 'I': 100}


#------------------------------------------------------------------------------
# Current inputs 
#------------------------------------------------------------------------------
cfg.addIClamp = 0

#------------------------------------------------------------------------------
# NetStim inputs 
#------------------------------------------------------------------------------

## Attempt to add Background Noise inputs 
## salva: this section was meant to provide single netstim inputs to particular populations; 
## bkg noise inputs can be implemented directly in netParams with just a couple of params in cfg to control strength for eg. E vs I

cfg.addNetStim = 1

## LAYER 1
cfg.NetStim1 = {'pop': 'NGF1', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}

## LAYER 2
cfg.NetStim2 = {'pop': 'IT2',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}








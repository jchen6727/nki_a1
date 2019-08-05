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
cfg.verbose = False           	## Show detailed messages


#------------------------------------------------------------------------------
# Recording 
#------------------------------------------------------------------------------
allpops = ['SOM1','PV1','VIP1','IT2','PV2','SOM2','VIP2','IT3','SOM3','PV3','VIP3','ITP4','ITS4','VIP4','IT5A','PV5A','SOM5A','VIP5A','IT5B','PT5B','PV5B','SOM5B','VIP5B','IT6','CT6','PV6','SOM6','VIP6']

cfg.recordTraces = {'V_soma': {'sec':'soma', 'loc':0.5, 'var':'v'}}  ## Dict with traces to record -- taken from M1 cfg.py 
cfg.recordStim = False			## Seen in M1 cfg.py
cfg.recordTime = False  		## SEen in M1 cfg.py 
cfg.recordStep = 0.1            ## Step size (in ms) to save data -- value from M1 cfg.py 


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

#cfg.analysis['plotTraces'] = {} 		## Seen in M1 cfg.py (line 68) 
cfg.analysis['plotRaster'] = {'include': allpops, 'saveFig': True, 'showFig': True} # from M1 cfg.py --> leaving off these params for now 'labels': 'overlay', 'popRates': True, 'orderInverse': True, 'timeRange': [0,500], 'popColors': popColors, 'figSize': (6,6), 'lw': 0.3, 'markerSize':10, 'marker': '.', 'dpi': 300}      	## Plot a raster
cfg.analysis['plot2Dnet'] = True      	## Plot 2D visualization of cell positions & connections 

#------------------------------------------------------------------------------
# Cells
#------------------------------------------------------------------------------






#------------------------------------------------------------------------------
# Synapses
#------------------------------------------------------------------------------


cfg.AMPATau2Factor = 1.0


#------------------------------------------------------------------------------
# Network 
#------------------------------------------------------------------------------

## These values taken from M1 cfg.py (https://github.com/Neurosim-lab/netpyne/blob/development/examples/M1detailed/cfg.py)
cfg.scale = 1.0     # Is this what should be used? 
cfg.sizeY = 2000.0 #1350.0 in M1_detailed # should this be set to 2000 since that is the full height of the column? 
cfg.sizeX = 400.0 # This may change depending on electrode radius 
cfg.sizeZ = 400.0
cfg.scaleDensity = 0.01 # From M1 --> how to determine appropriate value for this? 


#------------------------------------------------------------------------------
# Current inputs 
#------------------------------------------------------------------------------

cfg.addIClamp = 0

#------------------------------------------------------------------------------
# NetStim inputs 
#------------------------------------------------------------------------------

## Attempt to add Background Noise inputs 
cfg.addNetStim = 1

## LAYER 1
cfg.NetStim01 = {'pop': 'SOM1',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim02 = {'pop': 'PV1', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim03 = {'pop': 'VIP1',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}

## LAYER 2
cfg.NetStim04 = {'pop': 'IT2',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim05 = {'pop': 'SOM2', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim06 = {'pop': 'PV2',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim07 = {'pop': 'VIP2', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0,   'weight': 10.0, 'delay': 0}

## LAYER 3
cfg.NetStim08 = {'pop': 'IT3',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim09 = {'pop': 'SOM3', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim10 = {'pop': 'PV3',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim11 = {'pop': 'VIP3', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0,   'weight': 10.0, 'delay': 0}

## LAYER 4
cfg.NetStim12 = {'pop': 'ITP4', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim13 = {'pop': 'ITS4', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim14 = {'pop': 'SOM4', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim15 = {'pop': 'PV4', 	'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim16 = {'pop': 'VIP4', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5,  'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0,   'weight': 10.0, 'delay': 0}

## LAYER 5A
cfg.NetStim17 = {'pop': 'IT5A',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim18 = {'pop': 'SOM5A', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim19 = {'pop': 'PV5A',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim20 = {'pop': 'VIP5A', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0,   'weight': 10.0, 'delay': 0}

## LAYER 5B
cfg.NetStim21 = {'pop': 'IT5B',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim22 = {'pop': 'PT5B',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim23 = {'pop': 'SOM5B', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim24 = {'pop': 'PV5B',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim25 = {'pop': 'VIP5B', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0,   'weight': 10.0, 'delay': 0}

## LAYER 6
cfg.NetStim26 = {'pop': 'IT6',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim27 = {'pop': 'CT6',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim28 = {'pop': 'SOM6', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim29 = {'pop': 'PV6',  'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0, 	'weight': 10.0, 'delay': 0}
cfg.NetStim30 = {'pop': 'VIP6', 'ynorm': [0,1], 'sec': 'soma', 'loc': 0.5, 'synMech': ['AMPA'], 'synMechWeightFactor': [1.0], 'start': 0, 'interval': 1000.0/60.0, 'noise': 0.0, 'number': 60.0,    'weight': 10.0, 'delay': 0}











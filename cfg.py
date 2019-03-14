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

cfg.recordTraces = {}         	## Dict with traces to record
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

cfg.analysis['plotTraces'] = {} 		## Seen in M1 cfg.py (line 68) 
cfg.analysis['plotRaster'] = True     	## Plot a raster
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
cfg.scaleDensity = 0.01 # From M1 



"""
netParams.py 

High-level specifications for A1 network model using NetPyNE

Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com
"""


from netpyne import specs
import pickle, json

netParams = specs.NetParams()   # object of class NetParams to store the network parameters

#netParams.version = 49 # What is this for? Seen in M1 netParams.py 

try:
	from __main__ import cfg  # import SimConfig object with params from parent module
except:
	from cfg import cfg



#------------------------------------------------------------------------------
#
# NETWORK PARAMETERS
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# General network parameters
#------------------------------------------------------------------------------

netParams.scale = cfg.scale # Scale factor for number of cells
netParams.sizeX = cfg.sizeX # x-dimension (horizontal length) size in um
netParams.sizeY = cfg.sizeY # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = cfg.sizeZ # z-dimension (horizontal depth) size in um
netParams.shape = 'cylinder' # cylindrical (column-like) volume

#------------------------------------------------------------------------------
# General connectivity parameters
#------------------------------------------------------------------------------
## Values below taken from M1 netParams.py (https://github.com/Neurosim-lab/netpyne/blob/development/examples/M1detailed/netParams.py) 
netParams.scaleConnWeight = #1.0 # Connection weight scale factor (default if no model specified)
netParams.scaleConnWeightModels = #{'HH_simple': 1.0, 'HH_reduced': 1.0, 'HH_full': 1.0} #scale conn weight factor for each cell model
netParams.scaleConnWeightNetStims = #1.0 #0.5  # scale conn weight factor for NetStims
netParams.defaultThreshold = #0.0 # spike threshold, 10 mV is NetCon default, lower it for all cells
netParams.defaultDelay = #2.0 # default conn delay (ms)
netParams.propVelocity = #500.0 # propagation velocity (um/ms)
netParams.probLambda = #100.0  # length constant (lambda) for connection probability decay (um)



#------------------------------------------------------------------------------
# Cell parameters
#------------------------------------------------------------------------------

cellModels = [] # List of cell models? -- Seen in M1 netParams.py 

# II: 100-950, IV: 950-1250, V: 1250-1550, VI: 1550-2000 [info from original A1 github repo]
# Layer V has 2 parts --> E5R: 1250-1334 (0.625-0.667), E5B: 1334-1550 (0.667-0.775)
layer = {'2': [0.05,0.475], '4': [0.475,0.625], '5R': [0.625,0.667], '5B': [0.667,0.775],'5': [0.625,0.775], '6': [0.775,1]} # normalized layer boundaries -- seen in M1 netParams.py 

#------------------------------------------------------------------------------




#------------------------------------------------------------------------------
# Population parameters
#------------------------------------------------------------------------------

## These populations are listed in netParams.py from salva's M1 repo (https://github.com/Neurosim-lab/netpyne/blob/development/examples/M1detailed/netParams.py)
### LAYER 2:
netParams.popParams['IT2'] = {'cellType': 'IT', 'cellModel': '', 'ynormRange': layer['2']}
netParams.popParams['SOM2'] = {'cellType': 'SOM', 'cellModel': '', 'ynormRange': layer['2']}
netParams.popParams['PV2'] = {'cellType': 'PV','cellModel': '', 'ynormRange': layer['2']}
### LAYER 4: 
netParams.popParams['IT4'] = {'cellType': 'IT', 'cellModel': '', 'ynormRange': layer['4']}
### LAYER 5:
#### What layer should this be? In M1: 5A, but here, 5R? 
netParams.popParams['IT5A'] = {'cellType': 'IT', 'cellModel': '', 'ynormRange': layer['5R']}
netParams.popParams['SOM5A'] = {'cellType': 'SOM', 'cellModel': '', 'ynormRange': layer['5R']}
netParams.popParams['PV5A'] = {'cellType': 'PV', 'cellModel': '', 'ynormRange': layer['5R']}
#### Same question as above 
netParams.popParams['IT5B'] = {'cellType': 'IT', 'cellModel': '', 'ynormRange': layer['5B']}
netParams.popParams['PT5B'] = {'cellType': 'PT', 'cellModel': '', 'ynormRange': layer['5B']}
netParams.popParams['SOM5B'] = {'cellType': 'SOM', 'cellModel': '', 'ynormRange': layer['5B']}
netParams.popParams['PV5B'] = {'cellType': 'PV', 'cellModel': '', 'ynormRange': layer['5B']}
#### LAYER 6:
netParams.popParams['IT6'] = {'cellType': 'IT', 'cellModel': '', 'ynormRange': layer['6']}
netParams.popParams['CT6'] = {'cellType': 'CT', 'cellModel': '', 'ynormRange': layer['6']}
netParams.popParams['SOM6'] = {'cellType': 'SOM', 'cellModel': '', 'ynormRange': layer['6']}
netParams.popParams['PV6'] = {'cellType': 'PV', 'cellModel': '', 'ynormRange': layer['6']}




################## FROM ORIGINAL A1 GITHUB REPO #########################
### Cell types listed below come from labels.py -- NOT OFFICIAL YET! 2/15/19
### numCells should come from mpisim.py, cpernet, see Google Drive doc 

#### Regular Spiking (RS)
# netParams.popParams['E2'] = {'cellType': 'RS', 'numCells': 329, 'cellModel': ''}
# netParams.popParams['E4'] = {'cellType': 'RS', 'numCells': 140, 'cellModel': ''}
# netParams.popParams['E5R'] = {'cellType': 'RS', 'numCells': 18, 'cellModel': ''}
# netParams.popParams['E6'] = {'cellType': 'RS', 'numCells': 170, 'cellModel': ''}

#### Fast Spiking (FS)
# netParams.popParams['I2'] = {'cellType': 'FS', 'numCells': 54, 'cellModel': ''}
# netParams.popParams['I4'] = {'cellType': 'FS', 'numCells': 23, 'cellModel': ''}
# netParams.popParams['I5'] = {'cellType': 'FS', 'numCells': 6, 'cellModel': ''}
# netParams.popParams['I6'] = {'cellType': 'FS', 'numCells': 28, 'cellModel': ''}

#### Low Threshold Spiking (LTS)
# netParams.popParams['I2L'] = {'cellType': 'LTS', 'numCells': 54, 'cellModel': ''}
# netParams.popParams['I4L'] = {'cellType': 'LTS', 'numCells': 23, 'cellModel': ''}
# netParams.popParams['I5L'] = {'cellType': 'LTS', 'numCells': 6, 'cellModel': ''}
# netParams.popParams['I6L'] = {'cellType': 'LTS', 'numCells': 28, 'cellModel': ''}

#### Burst Firing (Burst)
# netParams.popParams['E5B'] = {'cellType': 'Burst', 'numCells': 26, 'cellModel': ''}

### THALAMUS (core)-- excluded for now per samn (2/17/19)
# netParams.popParams['TC'] = {'cellType': 'thal_core', 'numCells': 0, 'cellModel': ''}
# netParams.popParams['HTC'] = {'cellType': 'thal_core', 'numCells': 0, 'cellModel': ''}
# netParams.popParams['IRE'] = {'cellType': 'thal_core', 'numCells': 0, 'cellModel': ''}
### THALAMUS (matrix) -- excluded for now per samn (2/17/19)
#netParams.popParams['TCM'] = {'cellType': 'thal_matrix', 'numCells': 0, 'cellModel': ''}
#netParams.popParams['IREM'] = {'cellType': 'thal_matrix', 'numCells': 0, 'cellModel': ''}

#########################################################################





#------------------------------------------------------------------------------
# Synaptic mechanism parameters
#------------------------------------------------------------------------------
### wmat in mpisim.py & STYP in labels.py  
netParams.synMechParams['AM2'] = 
netParams.synMechParams['GA'] = 
netParams.synMechParams['GA2'] = 

#------------------------------------------------------------------------------
# Local connectivity parameters
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Subcellular connectivity (synaptic distributions)
#------------------------------------------------------------------------------  


#------------------------------------------------------------------------------
# Description
#------------------------------------------------------------------------------
netParams.description = """ 
- A1 network, ??? layers, ??? cell types 
"""




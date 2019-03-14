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

netParams.scale = cfg.scale # Scale factor for number of cells # NOT DEFINED YET! 3/11/19 # How is this different than scaleDensity? 
netParams.sizeX = cfg.sizeX # x-dimension (horizontal length) size in um
netParams.sizeY = cfg.sizeY # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = cfg.sizeZ # z-dimension (horizontal depth) size in um
netParams.shape = 'cylinder' # cylindrical (column-like) volume

#------------------------------------------------------------------------------
# General connectivity parameters
#------------------------------------------------------------------------------
## Values below taken from M1 netParams.py (https://github.com/Neurosim-lab/netpyne/blob/development/examples/M1detailed/netParams.py) 
netParams.scaleConnWeight = 1.0 # Connection weight scale factor (default if no model specified)
netParams.scaleConnWeightModels = {'HH_simple': 1.0, 'HH_reduced': 1.0, 'HH_full': 1.0} #scale conn weight factor for each cell model
netParams.scaleConnWeightNetStims = 1.0 #0.5  # scale conn weight factor for NetStims
netParams.defaultThreshold = 0.0 # spike threshold, 10 mV is NetCon default, lower it for all cells
netParams.defaultDelay = 2.0 # default conn delay (ms)
netParams.propVelocity = 500.0 # propagation velocity (um/ms)
netParams.probLambda = 100.0  # length constant (lambda) for connection probability decay (um)



#------------------------------------------------------------------------------
# Cell parameters
#------------------------------------------------------------------------------

cellModels = ['HH_simple', 'HH_reduced', 'HH_full'] # List of cell models? -- Seen in M1 netParams.py 

# II: 100-950, IV: 950-1250, V: 1250-1550, VI: 1550-2000 [info from original A1 github repo]
# Layer V has 2 parts --> E5R: 1250-1334 (0.625-0.667), E5B: 1334-1550 (0.667-0.775)
## Layer 5R --> Layer 5A  and  Layer 5B --> Layer 5B (B now stands for lower section of layer 5 vs. Bursting, as it did before)
## Layer 45A --> Layer 4 + Layer 5A 
layer = {'2': [0.05,0.475], '4': [0.475,0.625], '5A': [0.625,0.667], '5B': [0.667,0.775], '6': [0.775,1]} # normalized layer boundaries -- seen in M1 netParams.py 

#------------------------------------------------------------------------------
## Load cell rules previously saved using netpyne format
cellParamLabels = ['IT2_reduced', 'IT4_reduced', 'IT5A_full', 'IT5B_reduced', 'PT5B_reduced', 'IT6_reduced', 'CT6_reduced', 'PV_simple', 'SOM_simple']  # list of cell rules to load from file 
loadCellParams = cellParamLabels
# saveCellParams = False 


for ruleLabel in loadCellParams:
	netParams.loadCellParamsRule(label=ruleLabel, fileName='cells/'+ruleLabel+'_cellParams.pkl') # Load cellParams for each of the above cell subtypes  #PT5B_full was commented out in M1 netParams.py 


#------------------------------------------------------------------------------
# Population parameters
#------------------------------------------------------------------------------

## load densities
with open('cells/cellDensity.pkl', 'r') as fileObj: density = pickle.load(fileObj)['density']
density = {k: [x * cfg.scaleDensity for x in v] for k,v in density.items()} # Scale densities 

## These populations are listed in netParams.py from salva's M1 repo (https://github.com/Neurosim-lab/netpyne/blob/development/examples/M1detailed/netParams.py)
### LAYER 2:
netParams.popParams['IT2'] =    {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['2'],   'density': density[('M1','E')][0]}      # IT2_reduced   	# cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['SOM2'] =   {'cellType': 'SOM', 'cellModel': 'HH_simple',   'ynormRange': layer['2'],   'density': density[('M1','SOM')][0]}    # SOM_simple
netParams.popParams['PV2'] =    {'cellType': 'PV',  'cellModel': 'HH_simple',   'ynormRange': layer['2'],   'density': density[('M1','PV')][0]}     # PV_simple
### LAYER 4: 
netParams.popParams['IT4'] =    {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['4'],   'density': density[('M1','E')][1]}      # IT4_reduced   	# cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['SOM4'] = 	{'cellType': 'SOM', 'cellModel': 'HH_simple',   'ynormRange': layer['4'], 	'density': density[('M1','SOM')][1]}          
netParams.popParams['PV4'] = 	{'cellType': 'PV', 	'cellModel': 'HH_simple',   'ynormRange': layer['4'], 	'density': density[('M1','PV')][1]}          
### LAYER 5A: 
netParams.popParams['IT5A'] =  {'cellType': 'IT',  'cellModel': 'HH_full',     'ynormRange': layer['5A'], 	'density': density[('M1','E')][2]}      # IT5A_full     	# cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['SOM5A'] = {'cellType': 'SOM', 'cellModel': 'HH_simple',   'ynormRange': layer['5A'],	'density': density[('M1','SOM')][2]}          
netParams.popParams['PV5A'] =  {'cellType': 'PV',  'cellModel': 'HH_simple',   'ynormRange': layer['5A'],	'density': density[('M1','PV')][2]}         
### LAYER 5B: 
netParams.popParams['IT5B'] =  {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['5B'], 	'density': 0.5*density[('M1','E')][3]}  # IT5B_reduced  		# cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['PT5B'] =  {'cellType': 'PT',  'cellModel': 'HH_reduced',  'ynormRange': layer['5B'], 	'density': 0.5*density[('M1','E')][3]}  # using PT5B_reduced?	# cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['SOM5B'] = {'cellType': 'SOM', 'cellModel': 'HH_simple',   'ynormRange': layer['5B'],	'density': density[('M1','SOM')][3]}    # SOM_simple
netParams.popParams['PV5B'] =  {'cellType': 'PV',  'cellModel': 'HH_simple',   'ynormRange': layer['5B'],	'density': density[('M1','PV')][3]}     # PV_simple
### LAYER 6:
netParams.popParams['IT6'] =    {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['6'],   'density': 0.5*density[('M1','E')][4]}  # IT6_reduced   	# cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['CT6'] =    {'cellType': 'CT',  'cellModel': 'HH_reduced',  'ynormRange': layer['6'],   'density': 0.5*density[('M1','E')][4]}  # CT6_reduced   	# cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['SOM6'] =   {'cellType': 'SOM', 'cellModel': 'HH_simple',   'ynormRange': layer['6'],   'density': density[('M1','SOM')][4]}    # SOM_simple
netParams.popParams['PV6'] =    {'cellType': 'PV',  'cellModel': 'HH_simple',   'ynormRange': layer['6'],   'density': density[('M1','PV')][4]}     # PV_simple 




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
#netParams.synMechParams['AM2'] = 
#netParams.synMechParams['GA'] = 
#netParams.synMechParams['GA2'] = 

#------------------------------------------------------------------------------
# Local connectivity parameters
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Subcellular connectivity (synaptic distributions)
#------------------------------------------------------------------------------  


#------------------------------------------------------------------------------
# Description
#------------------------------------------------------------------------------




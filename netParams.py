"""
netParams.py 

High-level specifications for A1 network model using NetPyNE

Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com
"""

from netpyne import specs
import pickle, json

netParams = specs.NetParams()   # object of class NetParams to store the network parameters

try:
	from __main__ import cfg  # import SimConfig object with params from parent module
except:
	from cfg import cfg


#------------------------------------------------------------------------------
# VERSION 
#------------------------------------------------------------------------------
netParams.version = 9

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

cellModels = ['HH_simple', 'HH_reduced', 'HH_full'] # List of cell models

# II: 100-950, IV: 950-1250, V: 1250-1550, VI: 1550-2000 
layer = {'1': [0.00, 0.05], '2': [0.05, 0.08], '3': [0.08, 0.475], '4': [0.475,0.625], '5A': [0.625,0.667], '5B': [0.667,0.775], '6': [0.775,1]} # normalized layer boundaries  

#------------------------------------------------------------------------------
## Load cell rules previously saved using netpyne format (**** DOES NOT INCLUDE nonVIP CELLS ****)
cellParamLabels = ['IT2_reduced', 'ITP4_reduced', 'IT5A_full', 'IT5A_reduced', 'IT5B_reduced', 'PT5B_reduced', 'IT6_reduced', 'CT6_reduced', 'PV_simple', 'SOM_simple'] # list of cell rules to load from file 
loadCellParams = cellParamLabels
#saveCellParams = True # This saves the params as a .pkl file


for ruleLabel in loadCellParams:
	netParams.loadCellParamsRule(label=ruleLabel, fileName='cells/'+ruleLabel+'_cellParams.pkl') # Load cellParams for each of the above cell subtypes

## IMPORT VIP 
netParams.importCellParams(label='VIP_simple', conds={'cellType': 'VIP', 'cellModel': 'HH_simple'}, fileName='cells/vipcr_cell.hoc', cellName='VIPCRCell_EDITED', importSynMechs = True)

## IMPORT NGF 
netParams.importCellParams(label='NGF_simple', conds={'cellType': 'NGF', 'cellModel': 'HH_simple'}, fileName='cells/ngf_cell.hoc', cellName='ngfcell', importSynMechs = True)

## IMPORT L4 SPINY STELLATE 
netParams.importCellParams(label='ITS4_simple', conds={'cellType': 'ITS4', 'cellModel': 'HH_simple'}, fileName='cells/ITS4.py', cellName='ITS4_cell')


#------------------------------------------------------------------------------
# Population parameters
#------------------------------------------------------------------------------

## load densities
with open('cells/cellDensity.pkl', 'rb') as fileObj: density = pickle.load(fileObj)['density']
density = {k: [x * cfg.scaleDensity for x in v] for k,v in density.items()} # Scale densities 

### LAYER 1:
netParams.popParams['NGF1'] = {'cellType': 'NGF', 'cellModel': 'HH_simple','ynormRange': layer['1'],   'density': density[('A1','nonVIP')][0]}

### LAYER 2:
netParams.popParams['IT2'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['2'],   'density': density[('A1','E')][1]}      # IT2_reduced   	# cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['SOM2'] =    {'cellType': 'SOM', 'cellModel': 'HH_simple',   'ynormRange': layer['2'],   'density': density[('A1','SOM')][1]}    # SOM_simple
netParams.popParams['PV2'] =     {'cellType': 'PV',  'cellModel': 'HH_simple',   'ynormRange': layer['2'],   'density': density[('A1','PV')][1]}     # PV_simple
netParams.popParams['VIP2'] =    {'cellType': 'VIP', 'cellModel': 'HH_simple',   'ynormRange': layer['2'],   'density': density[('A1','VIP')][1]}
netParams.popParams['NGF2'] = {'cellType': 'NGF', 'cellModel': 'HH_simple','ynormRange': layer['2'],   'density': density[('A1','nonVIP')][1]}

### LAYER 3:
netParams.popParams['IT3'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['3'],   'density': density[('A1','E')][1]} ## CHANGE DENSITY
netParams.popParams['SOM3'] =    {'cellType': 'SOM', 'cellModel': 'HH_simple',   'ynormRange': layer['3'],   'density': density[('A1','SOM')][1]} ## CHANGE DENSITY
netParams.popParams['PV3'] =     {'cellType': 'PV',  'cellModel': 'HH_simple',   'ynormRange': layer['3'],   'density': density[('A1','PV')][1]} ## CHANGE DENSITY
netParams.popParams['VIP3'] =    {'cellType': 'VIP', 'cellModel': 'HH_simple',   'ynormRange': layer['3'],   'density': density[('A1','VIP')][1]} ## CHANGE DENSITY
netParams.popParams['NGF3'] = {'cellType': 'NGF', 'cellModel': 'HH_simple','ynormRange': layer['3'],   'density': density[('A1','nonVIP')][1]}


### LAYER 4: 
netParams.popParams['ITP4'] =	{'cellType': 'IT', 'cellModel': 'HH_reduced',  'ynormRange': layer['4'], 'density': density[('A1','E')][2]}      ## CHANGE DENSITY # ITP4_reduced
netParams.popParams['ITS4'] =	{'cellType': 'ITS4' , 'cellModel': 'HH_simple', 'ynormRange': layer['4'], 'density': density[('A1','E')][2]}      ## CHANGE DENSITY # ITP4_reduced
netParams.popParams['SOM4'] = 	 {'cellType': 'SOM', 'cellModel': 'HH_simple',   'ynormRange': layer['4'], 'density': density[('A1','SOM')][2]}
netParams.popParams['PV4'] = 	 {'cellType': 'PV', 'cellModel': 'HH_simple',   'ynormRange': layer['4'], 'density': density[('A1','PV')][2]}
netParams.popParams['VIP4'] =	{'cellType': 'VIP', 'cellModel': 'HH_simple',   'ynormRange': layer['4'], 'density': density[('A1','VIP')][2]}
netParams.popParams['NGF4'] = {'cellType': 'NGF', 'cellModel': 'HH_simple','ynormRange': layer['4'],   'density': density[('A1','nonVIP')][2]}

### LAYER 5A: 
netParams.popParams['IT5A'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5A'], 	'density': density[('A1','E')][3]}      # IT5A_full or reduced?!  	# cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['SOM5A'] =    {'cellType': 'SOM', 'cellModel': 'HH_simple',    'ynormRange': layer['5A'],	'density': density[('A1','SOM')][3]}          
netParams.popParams['PV5A'] =     {'cellType': 'PV',  'cellModel': 'HH_simple',    'ynormRange': layer['5A'],	'density': density[('A1','PV')][3]}         
netParams.popParams['VIP5A'] =    {'cellType': 'VIP', 'cellModel': 'HH_simple',    'ynormRange': layer['5A'],  'density': density[('A1','VIP')][3]}
netParams.popParams['NGF5A'] = {'cellType': 'NGF', 'cellModel': 'HH_simple', 'ynormRange': layer['5A'],   'density': density[('A1','nonVIP')][3]}

### LAYER 5B: 
netParams.popParams['IT5B'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'], 	'density': 0.5*density[('A1','E')][4]}  # IT5B_reduced  		# cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['PT5B'] =     {'cellType': 'PT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'], 	'density': 0.5*density[('A1','E')][4]}  # using PT5B_reduced?	# cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['SOM5B'] =    {'cellType': 'SOM', 'cellModel': 'HH_simple',    'ynormRange': layer['5B'],	'density': density[('A1','SOM')][4]}    # SOM_simple
netParams.popParams['PV5B'] =     {'cellType': 'PV',  'cellModel': 'HH_simple',    'ynormRange': layer['5B'],	'density': density[('A1','PV')][4]}     # PV_simple
netParams.popParams['VIP5B'] =    {'cellType': 'VIP', 'cellModel': 'HH_simple',    'ynormRange': layer['5B'],   'density': density[('A1','VIP')][4]}
netParams.popParams['NGF5B'] = {'cellType': 'NGF', 'cellModel': 'HH_simple', 'ynormRange': layer['5B'], 'density': density[('A1','nonVIP')][4]}

### LAYER 6:
netParams.popParams['IT6'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['6'],   'density': 0.5*density[('A1','E')][5]}  # IT6_reduced   	# cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['CT6'] =     {'cellType': 'CT',  'cellModel': 'HH_reduced',  'ynormRange': layer['6'],   'density': 0.5*density[('A1','E')][5]}  # CT6_reduced   	# cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['SOM6'] =    {'cellType': 'SOM', 'cellModel': 'HH_simple',   'ynormRange': layer['6'],   'density': density[('A1','SOM')][5]}    # SOM_simple
netParams.popParams['PV6'] =     {'cellType': 'PV',  'cellModel': 'HH_simple',   'ynormRange': layer['6'],   'density': density[('A1','PV')][5]}     # PV_simple 
netParams.popParams['VIP6'] =    {'cellType': 'VIP', 'cellModel': 'HH_simple',   'ynormRange': layer['6'],   'density': density[('A1','VIP')][5]}
netParams.popParams['NGF6'] = {'cellType': 'NGF', 'cellModel': 'HH_simple','ynormRange': layer['6'],   'density': density[('A1','nonVIP')][5]}



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

### From M1 detailed netParams.py 
netParams.synMechParams['NMDA'] = {'mod': 'MyExp2SynNMDABB', 'tau1NMDA': 15, 'tau2NMDA': 150, 'e': 0}
netParams.synMechParams['AMPA'] = {'mod':'MyExp2SynBB', 'tau1': 0.05, 'tau2': 5.3*cfg.AMPATau2Factor, 'e': 0}
netParams.synMechParams['GABAB'] = {'mod':'MyExp2SynBB', 'tau1': 3.5, 'tau2': 260.9, 'e': -93} 
netParams.synMechParams['GABAA'] = {'mod':'MyExp2SynBB', 'tau1': 0.07, 'tau2': 18.2, 'e': -80}
netParams.synMechParams['GABAASlow'] = {'mod': 'MyExp2SynBB','tau1': 2, 'tau2': 100, 'e': -80}


#------------------------------------------------------------------------------
# Current inputs (IClamp)
#------------------------------------------------------------------------------
# if cfg.addIClamp:
#  	for key in [k for k in dir(cfg) if k.startswith('IClamp')]:
# 		params = getattr(cfg, key, None)
# 		[pop,sec,loc,start,dur,amp] = [params[s] for s in ['pop','sec','loc','start','dur','amp']]
		
#         		# add stim source
# 		netParams.stimSourceParams[key] = {'type': 'IClamp', 'delay': start, 'dur': dur, 'amp': amp}
		
# 		# connect stim source to target
# 		netParams.stimTargetParams[key+'_'+pop] =  {
# 			'source': key, 
# 			'conds': {'pop': pop},
# 			'sec': sec, 
# 			'loc': loc}
#------------------------------------------------------------------------------
# NetStim inputs
#------------------------------------------------------------------------------
if cfg.addNetStim:
	for key in [k for k in dir(cfg) if k.startswith('NetStim')]:
		params = getattr(cfg, key, None)
		[pop, ynorm, sec, loc, synMech, synMechWeightFactor, start, interval, noise, number, weight, delay] = \
		[params[s] for s in ['pop', 'ynorm', 'sec', 'loc', 'synMech', 'synMechWeightFactor', 'start', 'interval', 'noise', 'number', 'weight', 'delay']] 

		# add stim source
		netParams.stimSourceParams[key] = {'type': 'NetStim', 'start': start, 'interval': interval, 'noise': noise, 'number': number}

		# connect stim source to target 
		netParams.stimTargetParams[key+'_'+pop] =  {
			'source': key, 
			'conds': {'pop': pop, 'ynorm': ynorm},
			'sec': sec, 
			'loc': loc,
			'synMech': synMech,
			'weight': weight,
			'synMechWeightFactor': synMechWeightFactor,
			'delay': delay}

#------------------------------------------------------------------------------
# Local connectivity parameters
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Subcellular connectivity (synaptic distributions)
#------------------------------------------------------------------------------  


#------------------------------------------------------------------------------
# Description
#------------------------------------------------------------------------------





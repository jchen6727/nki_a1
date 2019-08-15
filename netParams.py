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
netParams.version = 7

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
cellParamLabels = ['IT2_reduced', 'IT4_reduced', 'IT5A_full', 'IT5A_reduced', 'IT5B_reduced', 'PT5B_reduced', 'IT6_reduced', 'CT6_reduced', 'PV_simple', 'SOM_simple'] # list of cell rules to load from file 
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
netParams.popParams['IT2'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['2'],   'density': density[('A1','E')][1]}     # cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['SOM2'] =    {'cellType': 'SOM', 'cellModel': 'HH_simple',   'ynormRange': layer['2'],   'density': density[('A1','SOM')][1]}   
netParams.popParams['PV2'] =     {'cellType': 'PV',  'cellModel': 'HH_simple',   'ynormRange': layer['2'],   'density': density[('A1','PV')][1]}    
netParams.popParams['VIP2'] =    {'cellType': 'VIP', 'cellModel': 'HH_simple',   'ynormRange': layer['2'],   'density': density[('A1','VIP')][1]}
netParams.popParams['NGF2'] =    {'cellType': 'NGF', 'cellModel': 'HH_simple',   'ynormRange': layer['2'],   'density': density[('A1','nonVIP')][1]}

### LAYER 3:
netParams.popParams['IT3'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['3'],   'density': density[('A1','E')][1]} ## CHANGE DENSITY
netParams.popParams['SOM3'] =    {'cellType': 'SOM', 'cellModel': 'HH_simple',   'ynormRange': layer['3'],   'density': density[('A1','SOM')][1]} ## CHANGE DENSITY
netParams.popParams['PV3'] =     {'cellType': 'PV',  'cellModel': 'HH_simple',   'ynormRange': layer['3'],   'density': density[('A1','PV')][1]} ## CHANGE DENSITY
netParams.popParams['VIP3'] =    {'cellType': 'VIP', 'cellModel': 'HH_simple',   'ynormRange': layer['3'],   'density': density[('A1','VIP')][1]} ## CHANGE DENSITY
netParams.popParams['NGF3'] =    {'cellType': 'NGF', 'cellModel': 'HH_simple',   'ynormRange': layer['3'],   'density': density[('A1','nonVIP')][1]}


### LAYER 4: 
netParams.popParams['ITP4'] =	 {'cellType': 'IT', 'cellModel': 'HH_reduced',  'ynormRange': layer['4'],   'density': density[('A1','E')][2]}      ## CHANGE DENSITY #
netParams.popParams['ITS4'] =	 {'cellType': 'ITS4' , 'cellModel': 'HH_simple', 'ynormRange': layer['4'],  'density': density[('A1','E')][2]}      ## CHANGE DENSITY 
netParams.popParams['SOM4'] = 	 {'cellType': 'SOM', 'cellModel': 'HH_simple',   'ynormRange': layer['4'],  'density': density[('A1','SOM')][2]}
netParams.popParams['PV4'] = 	 {'cellType': 'PV', 'cellModel': 'HH_simple',   'ynormRange': layer['4'],   'density': density[('A1','PV')][2]}
netParams.popParams['VIP4'] =	 {'cellType': 'VIP', 'cellModel': 'HH_simple',   'ynormRange': layer['4'],  'density': density[('A1','VIP')][2]}
netParams.popParams['NGF4'] =    {'cellType': 'NGF', 'cellModel': 'HH_simple',   'ynormRange': layer['4'],  'density': density[('A1','nonVIP')][2]}

### LAYER 5A: 
netParams.popParams['IT5A'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5A'], 	'density': density[('A1','E')][3]}      
netParams.popParams['SOM5A'] =    {'cellType': 'SOM', 'cellModel': 'HH_simple',    'ynormRange': layer['5A'],	'density': density[('A1','SOM')][3]}          
netParams.popParams['PV5A'] =     {'cellType': 'PV',  'cellModel': 'HH_simple',    'ynormRange': layer['5A'],	'density': density[('A1','PV')][3]}         
netParams.popParams['VIP5A'] =    {'cellType': 'VIP', 'cellModel': 'HH_simple',    'ynormRange': layer['5A'],   'density': density[('A1','VIP')][3]}
netParams.popParams['NGF5A'] =    {'cellType': 'NGF', 'cellModel': 'HH_simple',    'ynormRange': layer['5A'],   'density': density[('A1','nonVIP')][3]}

### LAYER 5B: 
netParams.popParams['IT5B'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'], 	'density': 0.5*density[('A1','E')][4]}  
netParams.popParams['PT5B'] =     {'cellType': 'PT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'], 	'density': 0.5*density[('A1','E')][4]}  
netParams.popParams['SOM5B'] =    {'cellType': 'SOM', 'cellModel': 'HH_simple',    'ynormRange': layer['5B'],   'density': density[('A1', 'SOM')][4]}
netParams.popParams['PV5B'] =     {'cellType': 'PV',  'cellModel': 'HH_simple',    'ynormRange': layer['5B'],	'density': density[('A1','PV')][4]}     
netParams.popParams['VIP5B'] =    {'cellType': 'VIP', 'cellModel': 'HH_simple',    'ynormRange': layer['5B'],   'density': density[('A1','VIP')][4]}
netParams.popParams['NGF5B'] =    {'cellType': 'NGF', 'cellModel': 'HH_simple',    'ynormRange': layer['5B'],   'density': density[('A1','nonVIP')][4]}

### LAYER 6:
netParams.popParams['IT6'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['6'],   'density': 0.5*density[('A1','E')][5]}  
netParams.popParams['CT6'] =     {'cellType': 'CT',  'cellModel': 'HH_reduced',  'ynormRange': layer['6'],   'density': 0.5*density[('A1','E')][5]} 
netParams.popParams['SOM6'] =    {'cellType': 'SOM', 'cellModel': 'HH_simple',   'ynormRange': layer['6'],   'density': density[('A1','SOM')][5]}   
netParams.popParams['PV6'] =     {'cellType': 'PV',  'cellModel': 'HH_simple',   'ynormRange': layer['6'],   'density': density[('A1','PV')][5]}     
netParams.popParams['VIP6'] =    {'cellType': 'VIP', 'cellModel': 'HH_simple',   'ynormRange': layer['6'],   'density': density[('A1','VIP')][5]}
netParams.popParams['NGF6'] =    {'cellType': 'NGF', 'cellModel': 'HH_simple',   'ynormRange': layer['6'],   'density': density[('A1','nonVIP')][5]}

if cfg.singleCellPops:
    for pop in netParams.popParams.values(): pop['numCells'] = 1

## List of E and I pops to use later on
Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6']  # all layers

Ipops = ['NGF1',                            # L1
        'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
        'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
        'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
        'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
        'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
        'PV6', 'SOM6', 'VIP6', 'NGF6']      # L6 


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

ESynMech = ['AMPA', 'NMDA']
SOMESynMech = ['GABAASlow','GABAB']
SOMISynMech = ['GABAASlow']
PVSynMech = ['GABAA']


#------------------------------------------------------------------------------
# Background inputs
#------------------------------------------------------------------------------ 


#------------------------------------------------------------------------------
# Local connectivity parameters
#------------------------------------------------------------------------------

## load data from conn pre-processing file
with open('conn/conn.pkl', 'rb') as fileObj: connData = pickle.load(fileObj)
pmat = connData['pmat']
wmat = connData['wmat']

#------------------------------------------------------------------------------
## E -> E

for pre in Epops:
    for post in Epops:
        netParams.connParams['EE_pre_post'] = { 
            'preConds': {'pop': pre}, 
            'postConds': {'pop': post},
            'synMech': ESynMech,
            'probability': pmat[pre][post],
            'weight': pmat[pre][post] * cfg.EEGain * cfg.synsPerConnWeightFactor, 
            'synMechWeightFactor': cfg.synWeightFractionEE,
            'delay': 'defaultDelay+dist_2D/propVelocity',
            'synsPerConn': 1,
            'sec': 'spiny'}  # 'perisomatic' should be a secList with spiny dends in each cellParams

#------------------------------------------------------------------------------
## E -> I

for pre in Epops:
    for post in Ipops:
        netParams.connParams['EI_pre_post'] = { 
            'preConds': {'pop': pre}, 
            'postConds': {'pop': post},
            'synMech': ESynMech,
            'probability': pmat[pre][post],
            'weight': pmat[pre][post] * cfg.EEGain * cfg.synsPerConnWeightFactor, 
            'synMechWeightFactor': cfg.synWeightFractionEI,
            'delay': 'defaultDelay+dist_2D/propVelocity',
            'synsPerConn': 1,
            'sec': 'perisomatic'}  # 'perisomatic' should be a secList with spiny dends in each cellParams

#------------------------------------------------------------------------------
## I -> E

for pre in Epops:
    if 'SOM' in pre:
        synMech = SOMESynMech
    else:
        synMech = PVSynMech

    for post in Ipops:
        netParams.connParams['IE_pre_post'] = { 
            'preConds': {'pop': pre}, 
            'postConds': {'pop': post},
            'synMech': synMech,
            'probability': pmat[pre][post],
            'weight': pmat[pre][post] * cfg.IEGain * cfg.synsPerConnWeightFactor, 
            'synMechWeightFactor': cfg.synWeightFractionIE,
            'delay': 'defaultDelay+dist_2D/propVelocity',
            'synsPerConn': 1,
            'sec': 'spiny'}

#------------------------------------------------------------------------------
## I -> I

for pre in Ipops:
    if 'SOM' in pre:
        synMech = SOMESynMech
    else:
        synMech = PVSynMech

    for post in Ipops:
        netParams.connParams['IE_pre_post'] = { 
            'preConds': {'pop': pre}, 
            'postConds': {'pop': post},
            'synMech': synMech,
            'probability': '%f * exp(-dist_3D/probLambda)' % (pmat[pre][post]),
            'weight': pmat[pre][post] * cfg.IIGain * cfg.synsPerConnWeightFactor, 
            'synMechWeightFactor': cfg.synWeightFractionII,
            'delay': 'defaultDelay+dist_2D/propVelocity',
            'synsPerConn': 1,
            'sec': 'perisomatic'}


#------------------------------------------------------------------------------
# Subcellular connectivity (synaptic distributions)
#------------------------------------------------------------------------------  
# add bkg sources for E and I cells
netParams.stimSourceParams['bkgE'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['E'], 'noise': cfg.noiseBkg, 'number': 1e9}
netParams.stimSourceParams['bkgI'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['I'], 'noise': cfg.noiseBkg, 'number': 1e9}

# connect stim sources to target cells
netParams.stimTargetParams['bkgE->E'] =  {
    'source': 'bkgE', 
    'conds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
    'sec': 'soma', 
    'loc': 0.5,
    'synMech': ESynMech,
    'weight': cfg.weightBkg['E'],
    'synMechWeightFactor': cfg.synWeightFractionEE,
    'delay': cfg.delayBkg}

netParams.stimTargetParams['bkgE->E'] =  {
    'source': 'bkgE', 
    'conds': {'cellType': ['PV', 'SOM', 'PIV', 'NGF']},
    'sec': 'soma', 
    'loc': 0.5,
    'synMech': ESynMech,
    'weight': cfg.weightBkg['I'],
    'synMechWeightFactor': cfg.synWeightFractionEI,
    'delay': cfg.delayBkg}


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
# NetStim inputs (to simulate short external stimuli; not bkg)
#------------------------------------------------------------------------------
# if cfg.addNetStim:
# 	for key in [k for k in dir(cfg) if k.startswith('NetStim')]:
# 		params = getattr(cfg, key, None)
# 		[pop, ynorm, sec, loc, synMech, synMechWeightFactor, start, interval, noise, number, weight, delay] = \
# 		[params[s] for s in ['pop', 'ynorm', 'sec', 'loc', 'synMech', 'synMechWeightFactor', 'start', 'interval', 'noise', 'number', 'weight', 'delay']] 

# 		# add stim source
# 		netParams.stimSourceParams[key] = {'type': 'NetStim', 'start': start, 'interval': interval, 'noise': noise, 'number': number}

# 		# connect stim source to target 
# 		netParams.stimTargetParams[key+'_'+pop] =  {
# 			'source': key, 
# 			'conds': {'pop': pop, 'ynorm': ynorm},
# 			'sec': sec, 
# 			'loc': loc,
# 			'synMech': synMech,
# 			'weight': weight,
# 			'synMechWeightFactor': synMechWeightFactor,
# 			'delay': delay}

#------------------------------------------------------------------------------
# Description
#------------------------------------------------------------------------------

netParams.description = """
v7 - Added template for connectivity

"""
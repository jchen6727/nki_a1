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
netParams.version = 15

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
netParams.scaleConnWeightModels = {'HH_reduced': 1.0, 'HH_reduced': 1.0, 'HH_full': 1.0} #scale conn weight factor for each cell model
netParams.scaleConnWeightNetStims = 1.0 #0.5  # scale conn weight factor for NetStims
netParams.defaultThreshold = 0.0 # spike threshold, 10 mV is NetCon default, lower it for all cells
netParams.defaultDelay = 2.0 # default conn delay (ms)
netParams.propVelocity = 500.0 # propagation velocity (um/ms)
netParams.probLambda = 100.0  # length constant (lambda) for connection probability decay (um)


#------------------------------------------------------------------------------
# Cell parameters
#------------------------------------------------------------------------------

Etypes = ['IT', 'ITS4', 'PT', 'CT']
Itypes = ['PV', 'SOM', 'VIP', 'NGF']
cellModels = ['HH_reduced', 'HH_full'] # List of cell models

# II: 100-950, IV: 950-1250, V: 1250-1550, VI: 1550-2000 
layer = {'1': [0.00, 0.05], '2': [0.05, 0.08], '3': [0.08, 0.475], '4': [0.475, 0.625], '5A': [0.625, 0.667], '5B': [0.667, 0.775], '6': [0.775, 1], 'thal': [1.2, 1.4]}  # normalized layer boundaries  

# add layer border correction ??
#netParams.correctBorder = {'threshold': [cfg.correctBorderThreshold, cfg.correctBorderThreshold, cfg.correctBorderThreshold], 
#                        'yborders': [layer['2'][0], layer['5A'][0], layer['6'][0], layer['6'][1]]}  # correct conn border effect


#------------------------------------------------------------------------------
## Load cell rules previously saved using netpyne format (DOES NOT INCLUDE VIP, NGF and spiny stellate)
## include conditions ('conds') for each cellRule
cellParamLabels = { 'IT2_A1':  {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['2']},
                    'IT3_A1':  {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['3']},
                    'ITP4_A1': {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['4']},
                    'IT5A_A1': {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['5A']},
                    'CT5A_A1': {'cellModel': 'HH_reduced', 'cellType': 'CT', 'ynorm': layer['5A']},
                    'IT5B_A1': {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['5B']},
                    'PT5B_A1': {'cellModel': 'HH_reduced', 'cellType': 'PT', 'ynorm': layer['5B']},
                    'CT5B_A1': {'cellModel': 'HH_reduced', 'cellType': 'CT', 'ynorm': layer['5B']},
                    'IT6_A1':  {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['6']},
                    'CT6_A1':  {'cellModel': 'HH_reduced', 'cellType': 'CT', 'ynorm': layer['6']},
                    'PV_reduced':  {'cellModel': 'HH_reduced', 'cellType': 'PV', 'ynorm': [layer['2'][0],layer['6'][1]]},
                    'SOM_reduced': {'cellModel': 'HH_reduced', 'cellType': 'SOM', 'ynorm': [layer['2'][0], layer['6'][1]]}}

# Load cell rules from .pkl file 
loadCellParams = cellParamLabels

for ruleLabel in loadCellParams:
    netParams.loadCellParamsRule(label=ruleLabel, fileName='cells/' + ruleLabel + '_cellParams.pkl')  # Load cellParams for each of the above cell subtype
    netParams.cellParams[ruleLabel]['conds'] = cellParamLabels[ruleLabel]

    # set section lists
    secLists = {}
    if ruleLabel in ['IT2_A1', 'IT3_A1', 'ITP4_A1', 'IT5A_A1', 'CT5A_A1', 'IT5B_A1', 'PT5B_A1', 'CT5B_A1', 'IT6_A1', 'CT6_A1']:
        secLists['all'] = ['soma', 'Adend1', 'Adend2', 'Adend3', 'Bdend']
        secLists['proximal'] = ['soma', 'Bdend', 'Adend1']
        secLists['dend_all'] = ['Adend1', 'Adend2', 'Adend3', 'Bdend']
        secLists['apic'] = ['Adend1', 'Adend2', 'Adend3']
        secLists['apic_trunk'] = ['Adend1', 'Adend2']
        secLists['apic_lowertrunk'] = ['Adend1']
        secLists['apic_uppertrunk'] = ['Adend2']
        secLists['apic_tuft'] = ['Adend3']

    elif ruleLabel in ['ITS4']:
        secLists['all'] = secLists['proximal'] = ['soma', 'dend', 'dend1']
        secLists['dend_all'] = secLists['apic'] = secLists['apic_trunk'] = secLists['apic_lowertrunk'] = \
            secLists['apic_uppertrunk'] = secLists['apic_tuft'] = ['dend', 'dend1']

    elif ruleLabel in ['PV_reduced', 'SOM_reduced', 'NGF_reduced']:
        secLists['all'] = secLists['proximal'] = ['soma', 'dend']
        secLists['dend_all'] = ['dend']

    elif ruleLabel in ['VIP_reduced']:
        secLists['all'] = ['soma', 'rad1', 'rad2', 'ori1', 'ori2']
        secLists['proximal'] = ['soma', 'rad1', 'ori1']
        secLists['dend_all'] = ['rad1', 'rad2', 'ori1', 'ori2']

    # store secLists in netParams
    netParams.cellParams[ruleLabel]['secLists'] = dict(secLists)


## Import VIP cell rule from hoc file 
netParams.importCellParams(label='VIP_reduced', conds={'cellType': 'VIP', 'cellModel': 'HH_reduced'}, fileName='cells/vipcr_cell.hoc', cellName='VIPCRCell_EDITED', importSynMechs=True)
netParams.cellParams['VIP_reduced']['conds'] = {'cellModel': 'HH_reduced', 'cellType': 'VIP', 'ynorm': [layer['2'][0], layer['6'][1]]}

## Import NGF cell rule from hoc file
netParams.importCellParams(label='NGF_reduced', conds={'cellType': 'NGF', 'cellModel': 'HH_reduced'}, fileName='cells/ngf_cell.hoc', cellName='ngfcell', importSynMechs=True)
netParams.cellParams['NGF_reduced']['conds'] = {'cellModel': 'HH_reduced', 'cellType': 'NGF', 'ynorm': [layer['1'][0], layer['6'][1]]}

## Import L4 Spiny Stellate cell rule from .py file
netParams.importCellParams(label='ITS4_reduced', conds={'cellType': 'ITS4', 'cellModel': 'HH_reduced'}, fileName='cells/ITS4.py', cellName='ITS4_cell')
netParams.cellParams['ITS4_reduced']['conds'] = {'cellModel': 'HH_reduced', 'cellType': 'ITS4', 'ynorm': layer['4']}

## THALAMIC CELL MODELS

# Import RE (reticular) cell rule from .py file 
netParams.importCellParams(label='RE_reduced', conds={'cellType': 'RE', 'cellModel': 'HH_reduced'}, fileName='cells/sRE.py', cellName='sRE', importSynMechs=True)
netParams.cellParams['RE_reduced']['conds'] = {'cellModel': 'HH_reduced', 'cellType': 'RE', 'ynorm': layer['thal']}

# Import TC cell rule from .py file 
netParams.importCellParams(label='TC_reduced', conds={'cellType': 'TC', 'cellModel': 'HH_reduced'}, fileName='cells/sTC.py', cellName='sTC', importSynMechs=True)
netParams.cellParams['TC_reduced']['conds'] = {'cellModel': 'HH_reduced', 'cellType': 'TC', 'ynorm': layer['thal']}

# Import HTC cell rule from .py file 
netParams.importCellParams(label='HTC_reduced', conds={'cellType': 'HTC', 'cellModel': 'HH_reduced'}, fileName='cells/sHTC.py', cellName='sHTC', importSynMechs=True)
netParams.cellParams['HTC_reduced']['conds'] = {'cellModel': 'HH_reduced', 'cellType': 'HTC', 'ynorm': layer['thal']}


## Set weightNorm for each cell type
for ruleLabel in netParams.cellParams.keys():
    netParams.addCellParamsWeightNorm(ruleLabel, 'cells/' + ruleLabel + '_weightNorm.pkl', threshold=cfg.weightNormThreshold)  # add weightNorm


## Set 3D geometry for each cell type
for label in netParams.cellParams:
    if label in ['PV_reduced', 'SOM_reduced']:
        offset, prevL = 0, 0
        somaL = netParams.cellParams[label]['secs']['soma']['geom']['L']
        for secName, sec in netParams.cellParams[label]['secs'].items():
            sec['geom']['pt3d'] = []
            if secName in ['soma', 'dend']:  # set 3d geom of soma and Adends
                sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
                prevL = float(prevL + sec['geom']['L'])
                sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
            if secName in ['axon']:  # set 3d geom of axon
                sec['geom']['pt3d'].append([offset+0, 0, 0, sec['geom']['diam']])
                sec['geom']['pt3d'].append([offset + 0, -sec['geom']['L'], 0, sec['geom']['diam']])

    elif label in ['NGF_reduced']:
        offset, prevL = 0, 0
        somaL = netParams.cellParams[label]['secs']['soma']['geom']['L']
        for secName, sec in netParams.cellParams[label]['secs'].items():
            sec['geom']['pt3d'] = []
            if secName in ['soma', 'dend']:  # set 3d geom of soma and Adends
                sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
                prevL = float(prevL + sec['geom']['L'])
                sec['geom']['pt3d'].append([offset + 0, prevL, 0, sec['geom']['diam']])
                
    elif label in ['ITS4_reduced']:
        offset, prevL = 0, 0
        somaL = netParams.cellParams[label]['secs']['soma']['geom']['L']
        for secName, sec in netParams.cellParams[label]['secs'].items():
            sec['geom']['pt3d'] = []
            if secName in ['soma']:  # set 3d geom of soma 
                sec['geom']['pt3d'].append([offset+0, prevL, 0, 25])
                prevL = float(prevL + sec['geom']['L'])
                sec['geom']['pt3d'].append([offset + 0, prevL, 0, 25])
            if secName in ['dend']:  # set 3d geom of apic dendds
                sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
                prevL = float(prevL + sec['geom']['L'])
                sec['geom']['pt3d'].append([offset + 0, prevL, 0, sec['geom']['diam']])
            if secName in ['dend1']:  # set 3d geom of basal dend
                sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
                sec['geom']['pt3d'].append([offset+0.707*sec['geom']['L'], -(somaL+0.707*sec['geom']['L']), 0, sec['geom']['diam']])   
    elif label in ['RE_reduced', 'TC_reduced', 'HTC_reduced', 'VIP_reduced']:
        pass

    else: # E cells
        # set 3D pt geom
        offset, prevL = 0, 0
        somaL = netParams.cellParams[label]['secs']['soma']['geom']['L']
        for secName, sec in netParams.cellParams[label]['secs'].items():
            sec['geom']['pt3d'] = []
            if secName in ['soma', 'Adend1', 'Adend2', 'Adend3']:  # set 3d geom of soma and Adends
                sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
                prevL = float(prevL + sec['geom']['L'])
                sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
            if secName in ['Bdend']:  # set 3d geom of Bdend
                sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
                sec['geom']['pt3d'].append([offset+0.707*sec['geom']['L'], -(somaL+0.707*sec['geom']['L']), 0, sec['geom']['diam']])        
            if secName in ['axon']:  # set 3d geom of axon
                sec['geom']['pt3d'].append([offset+0, 0, 0, sec['geom']['diam']])
                sec['geom']['pt3d'].append([offset+0, -sec['geom']['L'], 0, sec['geom']['diam']])   

''' Temporary fixes for SfN19 poster

# # invert TC and HTC weightNorm -- for some reason are negative! (temporary fix!)
# netParams.cellParams['TC_reduced']['secs']['soma']['weightNorm'][0] *= -1
# netParams.cellParams['HTC_reduced']['secs']['soma']['weightNorm'][0] *= -1

# # increase some weightNorms
# netParams.cellParams['PV_reduced']['secs']['soma']['weightNorm'][0] *= 1.5
# netParams.cellParams['NGF_reduced']['secs']['soma']['weightNorm'][0] *= 3
# for i in range(len(netParams.cellParams['ITP4_A1']['secs']['soma']['weightNorm'])):
#     netParams.cellParams['ITP4_A1']['secs']['soma']['weightNorm'][i] *= 3.0
# for i in range(len(netParams.cellParams['ITS4_reduced']['secs']['soma']['weightNorm'])):
#     netParams.cellParams['ITS4_reduced']['secs']['soma']['weightNorm'][i] *= 3
'''

#------------------------------------------------------------------------------
# Population parameters
#------------------------------------------------------------------------------

## load densities
with open('cells/cellDensity.pkl', 'rb') as fileObj: density = pickle.load(fileObj)['density']
density = {k: [x * cfg.scaleDensity for x in v] for k,v in density.items()} # Scale densities 

### LAYER 1:
netParams.popParams['NGF1'] = {'cellType': 'NGF', 'cellModel': 'HH_reduced','ynormRange': layer['1'],   'density': density[('A1','nonVIP')][0]}

### LAYER 2:
netParams.popParams['IT2'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['2'],   'density': density[('A1','E')][1]}     # cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['SOM2'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','SOM')][1]}   
netParams.popParams['PV2'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','PV')][1]}    
netParams.popParams['VIP2'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','VIP')][1]}
netParams.popParams['NGF2'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','nonVIP')][1]}

### LAYER 3:
netParams.popParams['IT3'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['3'],   'density': density[('A1','E')][1]} ## CHANGE DENSITY
netParams.popParams['SOM3'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','SOM')][1]} ## CHANGE DENSITY
netParams.popParams['PV3'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','PV')][1]} ## CHANGE DENSITY
netParams.popParams['VIP3'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','VIP')][1]} ## CHANGE DENSITY
netParams.popParams['NGF3'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['3'],   'density': density[('A1','nonVIP')][1]}


### LAYER 4: 
netParams.popParams['ITP4'] =	 {'cellType': 'IT', 'cellModel': 'HH_reduced',  'ynormRange': layer['4'],   'density': 0.5*density[('A1','E')][2]}      ## CHANGE DENSITY #
netParams.popParams['ITS4'] =	 {'cellType': 'ITS4', 'cellModel': 'HH_reduced', 'ynormRange': layer['4'],  'density': 0.5*density[('A1','E')][2]}      ## CHANGE DENSITY 
netParams.popParams['SOM4'] = 	 {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','SOM')][2]}
netParams.popParams['PV4'] = 	 {'cellType': 'PV', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],   'density': density[('A1','PV')][2]}
netParams.popParams['VIP4'] =	 {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','VIP')][2]}
netParams.popParams['NGF4'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','nonVIP')][2]}

### LAYER 5A: 
netParams.popParams['IT5A'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5A'], 	'density': 0.5*density[('A1','E')][3]}      
netParams.popParams['CT5A'] =     {'cellType': 'CT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5A'],   'density': 0.5*density[('A1','E')][3]}  # density is [5] because we are using same numbers for L5A and L6 for CT cells? 
netParams.popParams['SOM5A'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],	'density': density[('A1','SOM')][3]}          
netParams.popParams['PV5A'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],	'density': density[('A1','PV')][3]}         
netParams.popParams['VIP5A'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],   'density': density[('A1','VIP')][3]}
netParams.popParams['NGF5A'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],   'density': density[('A1','nonVIP')][3]}

### LAYER 5B: 
netParams.popParams['IT5B'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'], 	'density': (1/3)*density[('A1','E')][4]}  
netParams.popParams['CT5B'] =     {'cellType': 'CT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'],   'density': (1/3)*density[('A1','E')][4]}  # density is [5] because we are using same numbers for L5B and L6 for CT cells? 
netParams.popParams['PT5B'] =     {'cellType': 'PT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'], 	'density': (1/3)*density[('A1','E')][4]}  
netParams.popParams['SOM5B'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],   'density': density[('A1', 'SOM')][4]}
netParams.popParams['PV5B'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],	'density': density[('A1','PV')][4]}     
netParams.popParams['VIP5B'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],   'density': density[('A1','VIP')][4]}
netParams.popParams['NGF5B'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',    'ynormRange': layer['5B'],   'density': density[('A1','nonVIP')][4]}

### LAYER 6:
netParams.popParams['IT6'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['6'],   'density': 0.5*density[('A1','E')][5]}  
netParams.popParams['CT6'] =     {'cellType': 'CT',  'cellModel': 'HH_reduced',  'ynormRange': layer['6'],   'density': 0.5*density[('A1','E')][5]} 
netParams.popParams['SOM6'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','SOM')][5]}   
netParams.popParams['PV6'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','PV')][5]}     
netParams.popParams['VIP6'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','VIP')][5]}
netParams.popParams['NGF6'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['6'],   'density': density[('A1','nonVIP')][5]}


### THALAMIC POPULATIONS (from prev model)
thalDensity = density[('A1','PV')][2] * 1.25  # temporary estimate (from prev model)

netParams.popParams['TC'] =     {'cellType': 'TC',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 0.75*thalDensity}  
netParams.popParams['TCM'] =    {'cellType': 'TC',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': thalDensity} 
netParams.popParams['HTC'] =    {'cellType': 'HTC', 'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 0.25*thalDensity}   
netParams.popParams['IRE'] =    {'cellType': 'RE',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': thalDensity}     
netParams.popParams['IREM'] =   {'cellType': 'RE',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': thalDensity}


if cfg.singleCellPops:
    for pop in netParams.popParams.values(): pop['numCells'] = 1

## List of E and I pops to use later on
Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'CT5B' , 'PT5B', 'IT6', 'CT6']  # all layers

Ipops = ['NGF1',                            # L1
        'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
        'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
        'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
        'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
        'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
        'PV6', 'SOM6', 'VIP6', 'NGF6']      # L6 




#------------------------------------------------------------------------------
# Synaptic mechanism parameters
#------------------------------------------------------------------------------

### From M1 detailed netParams.py 
netParams.synMechParams['NMDA'] = {'mod': 'MyExp2SynNMDABB', 'tau1NMDA': 15, 'tau2NMDA': 150, 'e': 0}
netParams.synMechParams['AMPA'] = {'mod':'MyExp2SynBB', 'tau1': 0.05, 'tau2': 5.3*cfg.AMPATau2Factor, 'e': 0}
netParams.synMechParams['GABAB'] = {'mod':'MyExp2SynBB', 'tau1': 3.5, 'tau2': 260.9, 'e': -93} 
netParams.synMechParams['GABAA'] = {'mod':'MyExp2SynBB', 'tau1': 0.07, 'tau2': 18.2, 'e': -80}
netParams.synMechParams['GABAA_VIP'] = {'mod':'MyExp2SynBB', 'tau1': 0.3, 'tau2': 6.4, 'e': -80}  # Pi et al 2013
netParams.synMechParams['GABAASlow'] = {'mod': 'MyExp2SynBB','tau1': 2, 'tau2': 100, 'e': -80}
netParams.synMechParams['GABAASlowSlow'] = {'mod': 'MyExp2SynBB', 'tau1': 200, 'tau2': 400, 'e': -80}

ESynMech = ['AMPA', 'NMDA']
SOMESynMech = ['GABAASlow','GABAB']
SOMISynMech = ['GABAASlow']
PVSynMech = ['GABAA']
VIPSynMech = ['GABAA_VIP']
NGFSynMech = ['GABAA', 'GABAB']


#------------------------------------------------------------------------------
# Background inputs
#------------------------------------------------------------------------------ 


#------------------------------------------------------------------------------
# Local connectivity parameters
#------------------------------------------------------------------------------

## load data from conn pre-processing file
with open('conn/conn.pkl', 'rb') as fileObj: connData = pickle.load(fileObj)
pmat = connData['pmat']
lmat = connData['lmat']
wmat = connData['wmat']
bins = connData['bins']
connDataSource = connData['connDataSource']

#------------------------------------------------------------------------------
## E -> E
if cfg.addConn:
    for pre in Epops:
        for post in Epops:
            if connDataSource['E->E/I'] in ['Allen_V1', 'Allen_custom']:
                prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])
            else:
                prob = pmat[pre][post]
            netParams.connParams['EE_'+pre+'_'+post] = { 
                'preConds': {'pop': pre}, 
                'postConds': {'pop': post},
                'synMech': ESynMech,
                'probability': prob,
                'weight': wmat[pre][post] * cfg.EEGain, 
                'synMechWeightFactor': cfg.synWeightFractionEE,
                'delay': 'defaultDelay+dist_3D/propVelocity',
                'synsPerConn': 1,
                'sec': 'spiny'}  # 'spiny' should be a secList with spiny dends in each cellParams


#------------------------------------------------------------------------------
## E -> I
if cfg.addConn:
    for pre in Epops:
        for post in Ipops:
            if connDataSource['E->E/I'] in ['Allen_V1', 'Allen_custom']:
                prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])
            else:
                prob = pmat[pre][post]
            netParams.connParams['EI_'+pre+'_'+post] = { 
                'preConds': {'pop': pre}, 
                'postConds': {'pop': post},
                'synMech': ESynMech,
                'probability': prob,
                'weight': wmat[pre][post] * cfg.EIGain, 
                'synMechWeightFactor': cfg.synWeightFractionEI,
                'delay': 'defaultDelay+dist_3D/propVelocity',
                'synsPerConn': 1,
                'sec': 'perisomatic'}  # 'perisomatic' should be a secList with perisomatic dends in each cellParams


#------------------------------------------------------------------------------
## I -> E
if cfg.addConn and cfg.IEGain > 0.0:

    if connDataSource['I->E/I'] == 'custom_A1':
        binsLabel = 'inh'
        preTypes = Itypes
        synMechs =  [PVSynMech, SOMESynMech, VIPSynMech, NGFSynMech]  
        weightFactors = [[1.0], cfg.synWeightFractionSOME, [1.0], cfg.synWeightFractionNGF] 
        secs = ['perisom', 'apicdend', 'apicdend', 'apicdend']
        postTypes = Etypes
        for ipreType, (preType, synMech, weightFactor, sec) in enumerate(zip(preTypes, synMechs, weightFactors, secs)):
            for ipostType, postType in enumerate(postTypes):
                for ipreBin, preBin in enumerate(bins[binsLabel]):
                    for ipostBin, postBin in enumerate(bins[binsLabel]):
                        ruleLabel = preType+'_'+postType+'_'+str(ipreBin)+'_'+str(ipostBin)
                        netParams.connParams[ruleLabel] = {
                            'preConds': {'cellType': preType, 'ynorm': list(preBin)},
                            'postConds': {'cellType': postType, 'ynorm': list(postBin)},
                            'synMech': synMech,
                            'probability': '%f * exp(-dist_2D/probLambda)' % (pmat[preType]['E'][ipreBin,ipostBin]),
                            'weight': cfg.IEweights[ipostBin] * cfg.IEGain,
                            'synMechWeightFactor': weightFactor,
                            'delay': 'defaultDelay+dist_3D/propVelocity',
                            'sec': sec} # simple I cells used right now only have soma
    #  BBP_S1 or Allen_V1
    else:

        ESynMech = ['AMPA', 'NMDA']
        SOMESynMech = ['GABAASlow','GABAB']
        SOMISynMech = ['GABAASlow']
        PVSynMech = ['GABAA']
        VIPSynMech = ['GABAA_VIP']
        NGFSynMech = ['GABAA', 'GABAB']

        for pre in Ipops:
            for post in Epops:
                if connDataSource['I->E/I'] in ['Allen_V1', 'Allen_custom']:
                    prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])
                else:
                    prob = pmat[pre][post]
                
                if 'SOM' in pre:
                    synMech = SOMESynMech
                elif 'PV' in pre:
                    synMech = PVSynMech
                elif 'VIP' in pre:
                    synMech = VIPSynMech
                elif 'NGF' in pre:
                    synMech = NGFSynMech

                netParams.connParams['IE_'+pre+'_'+post] = { 
                    'preConds': {'pop': pre}, 
                    'postConds': {'pop': post},
                    'synMech': synMech,
                    'probability': prob,
                    'weight': wmat[pre][post] * cfg.EIGain, 
                    'synMechWeightFactor': cfg.synWeightFractionEI,
                    'delay': 'defaultDelay+dist_3D/propVelocity',
                    'synsPerConn': 1,
                    'sec': 'perisomatic'}  # 'perisomatic' should be a secList with perisomatic dends in each cellParams

#------------------------------------------------------------------------------
## I -> I
if cfg.addConn and cfg.IIGain > 0.0:

    if connDataSource['I->E/I'] == 'custom_A1':
        binsLabel = 'inh'
        preTypes = Itypes
        synMechs = [PVSynMech, SOMISynMech, SOMISynMech, SOMISynMech] # Update VIP and NGF syns! 
        sec = 'perisom'
        postTypes = Itypes
        for ipre, (preType, synMech) in enumerate(zip(preTypes, synMechs)):
            for ipost, postType in enumerate(postTypes):
                for iBin, bin in enumerate(bins[binsLabel]):
                    ruleLabel = preType+'_'+postType+'_'+str(iBin)
                    netParams.connParams[ruleLabel] = {
                        'preConds': {'cellType': preType, 'ynorm': bin},
                        'postConds': {'cellType': postType, 'ynorm': bin},
                        'synMech': synMech,
                        'probability': '%f * exp(-dist_2D/probLambda)' % (pmat[preType][postType]),
                        'weight': cfg.IIweights[iBin] * cfg.IIGain,
                        'delay': 'defaultDelay+dist_3D/propVelocity',
                        'sec': sec} # simple I cells used right now only have soma

    #  BBP_S1 or Allen_V1
    else: 
        for pre in Ipops:
            for post in Ipops:
                if connDataSource['I->E/I'] in ['Allen_V1', 'Allen_custom']:
                    prob = '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post])
                else:
                    prob = pmat[pre][post]

                if 'SOM' in pre:
                    synMech = SOMISynMech
                elif 'PV' in pre:
                    synMech = PVSynMech
                elif 'VIP' in pre:
                    synMech = VIPSynMech
                elif 'NGF' in pre:
                    synMech = NGFSynMech

                netParams.connParams['II_'+pre+'_'+post] = { 
                    'preConds': {'pop': pre}, 
                    'postConds': {'pop': post},
                    'synMech': synMech,
                    'probability': prob,
                    'weight': wmat[pre][post] * cfg.IIGain, 
                    'synMechWeightFactor': cfg.synWeightFractionII,
                    'delay': 'defaultDelay+dist_3D/propVelocity',
                    'synsPerConn': 1,
                    'sec': 'perisomatic'}  # 'perisomatic' should be a secList with perisomatic dends in each cellParams


#------------------------------------------------------------------------------
# Thalamic connectivity parameters
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
## Intrathalamic 

TEpops = ['TC', 'TCM', 'HTC']
TIpops = ['IRE', 'IREM']

if cfg.addIntraThalamicConn:
    for pre in TEpops+TIpops:
        for post in TEpops+TIpops:
            if post in pmat[pre]:
                # for syns use ESynMech, SOMESynMech and SOMISynMech 
                if pre in TEpops:     # E->E
                    syn = ESynMech
                    synWeightFactor = cfg.synWeightFractionEE
                elif post in TEpops:  # I->E
                    syn = SOMESynMech
                    synWeightFactor = cfg.synWeightFractionIE
                else:                  # I->I
                    syn = SOMISynMech
                    synWeightFactor = [1.0]
                    
                netParams.connParams['ITh_'+pre+'_'+post] = { 
                    'preConds': {'pop': pre}, 
                    'postConds': {'pop': post},
                    'synMech': syn,
                    'probability': pmat[pre][post],
                    'weight': wmat[pre][post] * cfg.intraThalamicGain, 
                    'synMechWeightFactor': synWeightFactor,
                    'delay': 'defaultDelay+dist_3D/propVelocity',
                    'synsPerConn': 1,
                    'sec': 'soma'}  


#------------------------------------------------------------------------------
## Corticothalamic 
if cfg.addCorticoThalamicConn:
    for pre in Epops:
        for post in TEpops+TIpops:
            if post in pmat[pre]:
                netParams.connParams['CxTh_'+pre+'_'+post] = { 
                    'preConds': {'pop': pre}, 
                    'postConds': {'pop': post},
                    'synMech': ESynMech,
                    'probability': pmat[pre][post],
                    'weight': wmat[pre][post] * cfg.corticoThalamicGain, 
                    'synMechWeightFactor': cfg.synWeightFractionEE,
                    'delay': 'defaultDelay+dist_3D/propVelocity',
                    'synsPerConn': 1,
                    'sec': 'soma'}  

#------------------------------------------------------------------------------
## Thalamocortical 
if cfg.addThalamoCorticalConn:
    for pre in TEpops+TIpops:
        for post in Epops+Ipops:
            if post in pmat[pre]:
                # for syns use ESynMech, SOMESynMech and SOMISynMech 
                if pre in TEpops:     # E->E/I
                    syn = ESynMech
                    synWeightFactor = cfg.synWeightFractionEE
                elif post in Epops:  # I->E
                    syn = SOMESynMech
                    synWeightFactor = cfg.synWeightFractionIE
                else:                  # I->I
                    syn = SOMISynMech
                    synWeightFactor = [1.0]

                netParams.connParams['ThCx_'+pre+'_'+post] = { 
                    'preConds': {'pop': pre}, 
                    'postConds': {'pop': post},
                    'synMech': syn,
                    'probability': '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post]),
                    'weight': wmat[pre][post] * cfg.thalamoCorticalGain, 
                    'synMechWeightFactor': synWeightFactor,
                    'delay': 'defaultDelay+dist_3D/propVelocity',
                    'synsPerConn': 1,
                    'sec': 'soma'}  



#------------------------------------------------------------------------------
# Subcellular connectivity (synaptic distributions)
#------------------------------------------------------------------------------  

# Set target sections (somatodendritic distribution of synapses)
# From Billeh 2019 (Allen V1) (fig 4F) and Tremblay 2016 (fig 3)



if cfg.addSubConn:
    #------------------------------------------------------------------------------
    # E -> E2/3,4: soma,dendrites <200um
    netParams.subConnParams['E->E2,3,4'] = {
        'preConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']}, 
        'postConds': {'pops': ['IT2', 'IT3', 'ITP4', 'ITS4']},
        'sec': 'proximal',
        'groupSynMechs': ESynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # E -> E5,6: soma,dendrites (all)
    netParams.subConnParams['E->E5,6'] = {
        'preConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']}, 
        'postConds': {'pops': ['IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6']},
        'sec': 'all',
        'groupSynMechs': ESynMech, 
        'density': 'uniform'}
        
    #------------------------------------------------------------------------------
    # E -> I: soma, dendrite (all)
    netParams.subConnParams['E->I'] = {
        'preConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']}, 
        'postConds': {'cellType': ['PV','SOM','NGF', 'VIP']},
        'sec': 'all',
        'groupSynMechs': ESynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # NGF1 -> E: apic_tuft
    netParams.subConnParams['NGF1->E5,6'] = {
        'preConds': {'pops': ['NGF1']}, 
        'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
        'sec': 'apic_tuft',
        'groupSynMechs': NGFSynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # NGF2,3,4 -> E2,3,4: apic_trunk
    netParams.subConnParams['NGF2,3,4->E2,3,4'] = {
        'preConds': {'pops': ['NGF2', 'NGF3', 'NGF4']}, 
        'postConds': {'pops': ['IT2', 'IT3', 'ITP4', 'ITS4']},
        'sec': 'apic_trunk',
        'groupSynMechs': NGFSynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # NGF2,3,4 -> E5,6: apic_uppertrunk
    netParams.subConnParams['NGF2,3,4->E5,6'] = {
        'preConds': {'pops': ['NGF2', 'NGF3', 'NGF4']}, 
        'postConds': {'pops': ['IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6']},
        'sec': 'apic_uppertrunk',
        'groupSynMechs': NGFSynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    # NGF5,6 -> E5,6: apic_lowerrunk
    netParams.subConnParams['NGF5,6->E5,6'] = {
        'preConds': {'pops': ['NGF5A', 'NGF5B', 'NGF6']}, 
        'postConds': {'pops': ['IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6']},
        'sec': 'apic_lowertrunk',
        'groupSynMechs': NGFSynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    #  SOM -> E: all_dend (not close to soma)
    netParams.subConnParams['SOM->E'] = {
        'preConds': {'cellType': ['SOM']}, 
        'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
        'sec': 'dend_all',
        'groupSynMechs': SOMESynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    #  PV -> E: proximal
    netParams.subConnParams['PV->E'] = {
        'preConds': {'cellType': ['PV']}, 
        'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
        'sec': 'proximal',
        'groupSynMechs': PVSynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    #  TC -> E: proximal
    netParams.subConnParams['TC->E'] = {
        'preConds': {'cellType': ['TC', 'HTC']}, 
        'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
        'sec': 'proximal',
        'groupSynMechs': ESynMech, 
        'density': 'uniform'} 

    #------------------------------------------------------------------------------
    #  TCM -> E: apical
    netParams.subConnParams['TC->E'] = {
        'preConds': {'cellType': ['TCM']}, 
        'postConds': {'cellType': ['IT', 'ITS4', 'PT', 'CT']},
        'sec': 'apic',
        'groupSynMechs': ESynMech, 
        'density': 'uniform'}
        

#------------------------------------------------------------------------------
# Bakcground inputs 
#------------------------------------------------------------------------------  
if cfg.addBkgConn:
    # add bkg sources for E and I cells
    netParams.stimSourceParams['bkgE'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['E'], 'noise': cfg.noiseBkg['A1'], 'number': 1e9}
    netParams.stimSourceParams['bkgI'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['I'], 'noise': cfg.noiseBkg['A1'], 'number': 1e9}
    netParams.stimSourceParams['bkgThalE'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['ThalE'], 'noise': cfg.noiseBkg['thalamus'], 'number': 1e9}
    netParams.stimSourceParams['bkgThalI'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['ThalI'], 'noise': cfg.noiseBkg['thalamus'], 'number': 1e9}

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

    # connect stim sources to target cells
    netParams.stimTargetParams['bkgE->ITS4'] =  {
        'source': 'bkgE', 
        'conds': {'cellType': ['ITS4']},
        'sec': 'soma', 
        'loc': 0.5,
        'synMech': ESynMech,
        'weight': cfg.weightBkg['E']*1.0,
        'synMechWeightFactor': cfg.synWeightFractionEE,
        'delay': cfg.delayBkg}

    netParams.stimTargetParams['bkgI->I'] =  {
        'source': 'bkgI', 
        'conds': {'cellType': ['PV', 'SOM', 'VIP', 'NGF']},
        'sec': 'soma', 
        'loc': 0.5,
        'synMech': ESynMech,
        'weight': cfg.weightBkg['I'],
        'synMechWeightFactor': cfg.synWeightFractionEI,
        'delay': cfg.delayBkg}

    netParams.stimTargetParams['bkgThalE->ThalE'] =  {
        'source': 'bkgThalE', 
        'conds': {'cellType': ['TC', 'HTC']},
        'sec': 'soma', 
        'loc': 0.5,
        'synMech': ESynMech,
        'weight': cfg.weightBkg['ThalE'],
        'synMechWeightFactor': cfg.synWeightFractionEE,
        'delay': cfg.delayBkg}

    netParams.stimTargetParams['bkgThalI->ThalI'] =  {
        'source': 'bkgThalI', 
        'conds': {'cellType': ['RE']},
        'sec': 'soma', 
        'loc': 0.5,
        'synMech': ESynMech,
        'weight': cfg.weightBkg['ThalI'],
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
# Description
#------------------------------------------------------------------------------

netParams.description = """
v7 - Added template for connectivity
v8 - Added cell types
v9 - Added local connectivity
v10 - Added thalamic populations from prev model
v11 - Added thalamic conn from prev model
v12 - Added CT cells to L5B
v13 - Added CT cells to L5A
v14 - Fixed L5A & L5B E cell densities + added CT5A & CT5B to 'Epops'
v15 - Added cortical and thalamic conn to CT5A and CT5B 
"""
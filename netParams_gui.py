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
    from cfg_cell import cfg


#------------------------------------------------------------------------------
# VERSION 
#------------------------------------------------------------------------------
netParams.version = 22

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
cellParamLabels = { 'IT2_reduced':  {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['2']},
                    'IT3_reduced':  {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['3']},
                    'ITP4_reduced': {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['4']},
                    'IT5A_reduced': {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['5A']},
                    'CT5A_reduced': {'cellModel': 'HH_reduced', 'cellType': 'CT', 'ynorm': layer['5A']},
                    'IT5B_reduced': {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['5B']},
                    'PT5B_reduced': {'cellModel': 'HH_reduced', 'cellType': 'PT', 'ynorm': layer['5B']},
                    'CT5B_reduced': {'cellModel': 'HH_reduced', 'cellType': 'CT', 'ynorm': layer['5B']},
                    'IT6_reduced':  {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['6']},
                    'CT6_reduced':  {'cellModel': 'HH_reduced', 'cellType': 'CT', 'ynorm': layer['6']},
                    'PV_reduced':  {'cellModel': 'HH_reduced', 'cellType': 'PV', 'ynorm': [layer['2'][0],layer['6'][1]]},
                    'SOM_reduced': {'cellModel': 'HH_reduced', 'cellType': 'SOM', 'ynorm': [layer['2'][0], layer['6'][1]]}}

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

# Import Thalamic Interneuron cell from .py file 
netParams.importCellParams(label='TI_reduced', conds={'cellType': 'TI', 'cellModel': 'HH_reduced'}, fileName='cells/sTI.py', cellName='sTI_cell', importSynMechs=True)
netParams.cellParams['TI_reduced']['conds'] = {'cellModel': 'HH_reduced', 'cellType': 'TI', 'ynorm': layer['thal']}

# Load cell rules from .pkl / .json file 
cellParamLabels = ['IT2_reduced', 'IT3_reduced', 'ITP4_reduced', 'ITS4_reduced',
                    'IT5A_reduced', 'CT5A_reduced', 'IT5B_reduced',
                    'PT5B_reduced', 'CT5B_reduced', 'IT6_reduced', 'CT6_reduced',
                    'PV_reduced', 'SOM_reduced', 'VIP_reduced', 'NGF_reduced',
                    'RE_reduced', 'TC_reduced', 'HTC_reduced', 'TI_reduced']

for ruleLabel in cellParamLabels:
    netParams.loadCellParamsRule(label=ruleLabel, fileName='cells/' + ruleLabel + '_cellParams.json')  # Load cellParams for each of the above cell subtype
    #netParams.cellParams[ruleLabel]['conds'] = cellParamLabels[ruleLabel]

## Options to add to cellParams
addSecLists = False
add3DGeom = False

## Set weightNorm for each cell type and add section lists (used in connectivity)
for ruleLabel in netParams.cellParams.keys():
    try:
        netParams.addCellParamsWeightNorm(ruleLabel, 'cells/' + ruleLabel + '_weightNorm.pkl', threshold=cfg.weightNormThreshold)  # add weightNorm
        print('   Loaded weightNorm pkl file for %s...' % (ruleLabel))
    except:
        print('   No weightNorm pkl file for %s...' % (ruleLabel))

    # remove
    if cfg.removeWeightNorm:
        for sec in netParams.cellParams[ruleLabel]['secs']:
            if 'weightNorm' in netParams.cellParams[ruleLabel]['secs'][sec]:    
                del netParams.cellParams[ruleLabel]['secs'][sec]['weightNorm']

    if addSecLists:
        secLists = {}
        if ruleLabel in ['IT2_reduced', 'IT3_reduced', 'ITP4_reduced', 'IT5A_reduced', 'CT5A_reduced', 'IT5B_reduced', 'PT5B_reduced', 'CT5B_reduced', 'IT6_reduced', 'CT6_reduced']:
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

        elif ruleLabel in ['PV_reduced', 'SOM_reduced', 'NGF_reduced', 'TI_reduced']:
            secLists['all'] = secLists['proximal'] = ['soma', 'dend']
            secLists['dend_all'] = ['dend']

        elif ruleLabel in ['VIP_reduced']:
            secLists['all'] = ['soma', 'rad1', 'rad2', 'ori1', 'ori2']
            secLists['proximal'] = ['soma', 'rad1', 'ori1']
            secLists['dend_all'] = ['rad1', 'rad2', 'ori1', 'ori2']

        # store secLists in netParams
        netParams.cellParams[ruleLabel]['secLists'] = dict(secLists)


if add3DGeom:
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

        elif label in ['NGF_reduced', 'TI_reduced']:
            offset, prevL = 0, 0
            somaL = netParams.cellParams[label]['secs']['soma']['geom']['L']
            for secName, sec in netParams.cellParams[label]['secs'].items():
                sec['geom']['pt3d'] = []
                if secName in ['soma', 'dend']:  # set 3d geom of soma and Adends
                    sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
                    prevL = float(prevL + sec['geom']['L'])
                    sec['geom']['pt3d'].append([offset + 0, prevL, 0, sec['geom']['diam']])

        elif label in ['VIP_reduced']:
            offset, prevL = 0, 0
            somaL = netParams.cellParams[label]['secs']['soma']['geom']['L']
            for secName, sec in netParams.cellParams[label]['secs'].items():
                sec['geom']['pt3d'] = []
                if secName in ['soma']:  # set 3d geom of soma 
                    sec['geom']['pt3d'].append([offset+0, prevL, 0, 25])
                    prevL = float(prevL + sec['geom']['L'])
                    sec['geom']['pt3d'].append([offset + 0, prevL, 0, 25])
                if secName in ['rad1']:  # set 3d geom of rad1 (radiatum)
                    sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
                    sec['geom']['pt3d'].append([offset+0.5*sec['geom']['L'], +(somaL+0.866*sec['geom']['L']), 0, sec['geom']['diam']])   
                if secName in ['rad2']:  # set 3d geom of rad2 (radiatum)
                    sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
                    sec['geom']['pt3d'].append([offset-0.5*sec['geom']['L'], +(somaL+0.866*sec['geom']['L']), 0, sec['geom']['diam']])   
                if secName in ['ori1']:  # set 3d geom of ori1 (oriens)
                    sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
                    sec['geom']['pt3d'].append([offset+0.707*sec['geom']['L'], -(somaL+0.707*sec['geom']['L']), 0, sec['geom']['diam']])   
                if secName in ['ori2']:  # set 3d geom of ori2 (oriens)
                    sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
                    sec['geom']['pt3d'].append([offset-0.707*sec['geom']['L'], -(somaL+0.707*sec['geom']['L']), 0, sec['geom']['diam']])   


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
                    sec['geom']['pt3d'].append([offset + 0, -sec['geom']['L'], 0, sec['geom']['diam']])
                    

# save cellParams rules to .pkl file
saveCellParams = True
if saveCellParams:
    for ruleLabel in netParams.cellParams.keys():
        netParams.saveCellParamsRule(label=ruleLabel, fileName='cells/' + ruleLabel + '_cellParams.json')


#------------------------------------------------------------------------------
# Population parameters
#------------------------------------------------------------------------------

## load densities
with open('cells/cellDensity.pkl', 'rb') as fileObj: density = pickle.load(fileObj)['density']
density = {k: [x * cfg.scaleDensity for x in v] for k,v in density.items()} # Scale densities 

### LAYER 1:
netParams.popParams['NGF1'] = {'cellType': 'NGF', 'cellModel': 'HH_reduced', 'ynormRange': layer['1'],   'density': density[('A1','nonVIP')][0]}

### LAYER 2:
netParams.popParams['IT2'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': layer['2'],   'density': density[('A1','E')][1]}     # cfg.cellmod for 'cellModel' in M1 netParams.py 
netParams.popParams['SOM2'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','SOM')][1]}   
netParams.popParams['PV2'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','PV')][1]}    
netParams.popParams['VIP2'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','VIP')][1]}
netParams.popParams['NGF2'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['2'],   'density': density[('A1','nonVIP')][1]}

### LAYER 3:
netParams.popParams['IT3'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',  'ynormRange': [0.3, 0.3],   'density': density[('A1','E')][1]} ## CHANGE DENSITY
netParams.popParams['SOM3'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': [0.3, 0.3],   'density': density[('A1','SOM')][1]} ## CHANGE DENSITY
netParams.popParams['PV3'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',   'ynormRange': [0.3, 0.3],   'density': density[('A1','PV')][1]} ## CHANGE DENSITY
netParams.popParams['VIP3'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': [0.3, 0.3],   'density': density[('A1','VIP')][1]} ## CHANGE DENSITY
netParams.popParams['NGF3'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': [0.3, 0.3],   'density': density[('A1','nonVIP')][1]}


### LAYER 4: 
netParams.popParams['ITP4'] =	 {'cellType': 'IT', 'cellModel': 'HH_reduced',  'ynormRange': layer['4'],   'density': 0.5*density[('A1','E')][2]}      ## CHANGE DENSITY #
netParams.popParams['ITS4'] =	 {'cellType': 'ITS4', 'cellModel': 'HH_reduced', 'ynormRange': layer['4'],  'density': 0.5*density[('A1','E')][2]}      ## CHANGE DENSITY 
netParams.popParams['SOM4'] = 	 {'cellType': 'SOM', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','SOM')][2]}
netParams.popParams['PV4'] = 	 {'cellType': 'PV', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],   'density': density[('A1','PV')][2]}
netParams.popParams['VIP4'] =	 {'cellType': 'VIP', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','VIP')][2]}
netParams.popParams['NGF4'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',   'ynormRange': layer['4'],  'density': density[('A1','nonVIP')][2]}

### LAYER 5A: 
netParams.popParams['IT5A'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5A'], 	'density': density[('A1','E')][3]}      
netParams.popParams['SOM5A'] =    {'cellType': 'SOM', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],	'density': density[('A1','SOM')][3]}          
netParams.popParams['PV5A'] =     {'cellType': 'PV',  'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],	'density': density[('A1','PV')][3]}         
netParams.popParams['VIP5A'] =    {'cellType': 'VIP', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],   'density': density[('A1','VIP')][3]}
netParams.popParams['NGF5A'] =    {'cellType': 'NGF', 'cellModel': 'HH_reduced',    'ynormRange': layer['5A'],   'density': density[('A1','nonVIP')][3]}

### LAYER 5B: 
netParams.popParams['IT5B'] =     {'cellType': 'IT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'], 	'density': 0.5*density[('A1','E')][4]}  
netParams.popParams['PT5B'] =     {'cellType': 'PT',  'cellModel': 'HH_reduced',   'ynormRange': layer['5B'], 	'density': 0.5*density[('A1','E')][4]}  
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
thalDensity = density[('A1','PV')][2]  # temporary estimate (from prev model)

netParams.popParams['TC'] =     {'cellType': 'TC',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 0.75*thalDensity}  
netParams.popParams['TCM'] =    {'cellType': 'TC',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': thalDensity} 
netParams.popParams['HTC'] =    {'cellType': 'HTC', 'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 0.25*thalDensity}   
netParams.popParams['IRE'] =    {'cellType': 'RE',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': thalDensity}     
netParams.popParams['IREM'] =   {'cellType': 'RE',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': thalDensity}
netParams.popParams['TI'] =     {'cellType': 'TI',  'cellModel': 'HH_reduced',  'ynormRange': layer['thal'],   'density': 2*0.33 * thalDensity} ## Winer & Larue 1996; 

cfg.singleCellPops = 1 # since for GUI
if cfg.singleCellPops:
    for pop in netParams.popParams.values(): pop['numCells'] = 1

# set x,z spacing
step = 150

for i,pop in enumerate(netParams.popParams):
    netParams.popParams[pop]['znormRange'] = [0.5, 0.5]
    netParams.popParams[pop]['xRange'] = [i*step, i*step]


#------------------------------------------------------------------------------
# Description
#------------------------------------------------------------------------------

netParams.description = """
v7 - Added template for connectivity
v8 - Added cell types
v9 - Added local connectivity
v10 - Added thalamic populations from prev model
"""
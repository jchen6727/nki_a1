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
netParams.version = 11

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
                    'IT5B_A1': {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['5B']},
                    'PT5B_A1': {'cellModel': 'HH_reduced', 'cellType': 'PT', 'ynorm': layer['5B']},
                    'IT6_A1':  {'cellModel': 'HH_reduced', 'cellType': 'IT', 'ynorm': layer['6']},
                    'CT6_A1':  {'cellModel': 'HH_reduced', 'cellType': 'CT', 'ynorm': layer['6']},
                    'PV_reduced':  {'cellModel': 'HH_reduced', 'cellType': 'PV', 'ynorm': [layer['2'][0],layer['6'][1]]},
                    'SOM_reduced': {'cellModel': 'HH_reduced', 'cellType': 'SOM', 'ynorm': [layer['2'][0], layer['6'][1]]}}
                    

# temporary weightNorm value (temporary fix!)
weightNorm = 1.0

# Load cell rules from .pkl file 
loadCellParams = cellParamLabels

for ruleLabel in loadCellParams:
    netParams.loadCellParamsRule(label=ruleLabel, fileName='cells/' + ruleLabel + '_cellParams.pkl')  # Load cellParams for each of the above cell subtype
    netParams.cellParams[ruleLabel]['conds'] = cellParamLabels[ruleLabel]
    
    # set weightNorm (temporary fix!)
    for sec in netParams.cellParams[ruleLabel]['secs']:
        netParams.cellParams[ruleLabel]['secs'][sec]['weightNorm'] = weightNorm
    

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


## Set weightNorm for VIP, NGS ITS4, RE, TC, HTC (temporary fix!)
for ruleLabel in ['VIP_reduced', 'NGF_reduced', 'ITS4_reduced', 'RE_reduced', 'TC_reduced', 'HTC_reduced']:
    for sec in netParams.cellParams[ruleLabel]['secs']:
        netParams.cellParams[ruleLabel]['secs'][sec]['weightNorm'] = weightNorm

#------------------------------------------------------------------------------
# Population parameters
#------------------------------------------------------------------------------

## load densities
with open('cells/cellDensity.pkl', 'rb') as fileObj: density = pickle.load(fileObj)['density']
density = {k: [x * cfg.scaleDensity for x in v] for k,v in density.items()} # Scale densities 

### LAYER 1:

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

if cfg.singleCellPops:
    for popName,pop in netParams.popParams.items():
        if cfg.NetStim1['pop'] == popName:
            pop['numCells'] = 1
        else:
            pop['numCells'] = 0

## List of E and I pops to use later on
Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6']  # all layers

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

#------------------------------------------------------------------------------
## E -> E
if cfg.addConn:
    for pre in Epops:
        for post in Epops:
            netParams.connParams['EE_'+pre+'_'+post] = { 
                'preConds': {'pop': pre}, 
                'postConds': {'pop': post},
                'synMech': ESynMech,
                'probability': '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post]),
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
            netParams.connParams['EI_'+pre+'_'+post] = { 
                'preConds': {'pop': pre}, 
                'postConds': {'pop': post},
                'synMech': ESynMech,
                'probability': '%f * exp(-dist_2D/%f)' % (pmat[pre][post], lmat[pre][post]),
                'weight': wmat[pre][post] * cfg.EIGain, 
                'synMechWeightFactor': cfg.synWeightFractionEI,
                'delay': 'defaultDelay+dist_3D/propVelocity',
                'synsPerConn': 1,
                'sec': 'perisomatic'}  # 'perisomatic' should be a secList with perisomatic dends in each cellParams


#------------------------------------------------------------------------------
## I -> E
if cfg.addConn and cfg.IEGain > 0.0:

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


#------------------------------------------------------------------------------
## I -> I
if cfg.addConn and cfg.IIGain > 0.0:

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
if cfg.addCoreThalamoCorticalConn:
    for pre in TEpops+TIpops:
        for post in Epops+Ipops:
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

                netParams.connParams['ThCx_'+pre+'_'+post] = { 
                    'preConds': {'pop': pre}, 
                    'postConds': {'pop': post},
                    'synMech': syn,
                    'probability': pmat[pre][post],
                    'weight': wmat[pre][post] * cfg.corticoThalamicGain, 
                    'synMechWeightFactor': synWeightFactor,
                    'delay': 'defaultDelay+dist_3D/propVelocity',
                    'synsPerConn': 1,
                    'sec': 'soma'}  



#------------------------------------------------------------------------------
# Subcellular connectivity (synaptic distributions)
#------------------------------------------------------------------------------  



#------------------------------------------------------------------------------
# Bakcgroudn inputs 
#------------------------------------------------------------------------------  
if cfg.addBkgConn:
    # add bkg sources for E and I cells
    netParams.stimSourceParams['bkgE'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['E'], 'noise': cfg.noiseBkg, 'number': 1e9}
    netParams.stimSourceParams['bkgI'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['I'], 'noise': cfg.noiseBkg, 'number': 1e9}
    netParams.stimSourceParams['bkgThalE'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['ThalE'], 'noise': cfg.noiseBkg, 'number': 1e9}
    netParams.stimSourceParams['bkgThalI'] = {'type': 'NetStim', 'start': cfg.startBkg, 'rate': cfg.rateBkg['ThalI'], 'noise': cfg.noiseBkg, 'number': 1e9}

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

        cfg.analysis['plotTraces']['include'] = [(pop, 0)]

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
"""
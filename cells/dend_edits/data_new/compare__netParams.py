## CHANGE 3 THINGS 
from netpyne import specs, sim

try:
	from __main__ import cfg  # import SimConfig object with params from parent module
except:
	from cfg import cfg  # if no simConfig in parent module, import directly from tut8_cfg module


netParams = specs.NetParams() 
netParams.popParams['TestPop'] = {'cellType': 'ITS', 'numCells': 1, 'cellModel': 'HH_reduced'} #cellType

## FOR PY FILES: 
netParams.importCellParams(label='ITS4_reduced', conds={'cellType':'ITS', 'cellModel':'HH_reduced'}, fileName = 'ITS4.py', cellName = 'ITS4_cell')

## FOR PKL FILES: 
#netParams.loadCellParamsRule(label='CT6_reduced', fileName='CT6_reduced_new_cellParams.pkl') #cellName
#del netParams.cellParams['CT6_reduced']['conds'] # To make sure that cell is instantiated 


## V INIT FOR PKL
#netParams.cellParams['CT6_reduced']['globals']['v_init'] = -80 
#netParams.cellParams['CT6_reduced']['secs']['soma']['vinit'] = -80 
#netParams.cellParams['CT6_reduced']['secs']['Adend1']['vinit'] = -85.3
#netParams.cellParams['CT6_reduced']['secs']['Adend2']['vinit'] = -85.3
#netParams.cellParams['CT6_reduced']['secs']['Adend3']['vinit'] = -85.3
#netParams.cellParams['CT6_reduced']['secs']['Bdend']['vinit'] = -81.2
#netParams.cellParams['CT6_reduced']['secs']['axon']['vinit'] = -81.2

## V INIT FOR PY FILE: 
#netParams.cellParams['ITS4_reduced']['secs']['soma']['vinit'] = 
#netParams.cellParams['ITS4_reduced']['secs']['dend']['vinit'] = 

# ## PASSIVE ##
netParams.cellParams['ITS4_reduced']['secs']['soma']['mechs']['pas']['g'] = 3.79e-03 # ORIG: 3.3333333333333335e-05

#netParams.cellParams['ITS4_reduced']['secs']['soma']['mechs']['pas']['e'] =  # ORIG: -70.0
#netParams.cellParams['ITS4_reduced']['secs']['soma']['geom']['cm'] = # ORIG: 0.75

# ## ACTIVE ##
# SOMA
netParams.cellParams['ITS4_reduced']['secs']['soma']['mechs']['naz']['gmax'] = 72000 # ORIG: 30000 
netParams.cellParams['ITS4_reduced']['secs']['soma']['mechs']['kv']['gbar'] = 1700 # ORIG: 1500 


# DENDRITE
#netParams.cellParams['ITS4_reduced']['secs']['dend']['mechs']['naz']['gmax'] = 1 # ORIG: 15
#netParams.cellParams['ITS4_reduced']['secs']['soma']['mechs']['km']['gbar'] = 50 # ORIG: 0.1 


## Add in Stimulation Source (IClamp) 
netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 250, 'dur': 1000, 'amp': cfg.amp} 
netParams.stimTargetParams['Input->TestPop'] = {'source': 'Input', 'sec':'soma', 'loc': 0.5, 'conds': {'pop':'TestPop'}}
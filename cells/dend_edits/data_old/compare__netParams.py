### CHANGE 3 THINGS
from netpyne import specs, sim

try:
	from __main__ import cfg  # import SimConfig object with params from parent module
except:
	from cfg import cfg  # if no simConfig in parent module, import directly from tut8_cfg module


netParams = specs.NetParams() 
netParams.popParams['TestPop'] = {'cellType': 'ITS', 'numCells': 1, 'cellModel': 'HH_reduced'} # (1) #cellType

## FOR PKL FILES: 
# cellRule = netParams.loadCellParamsRule(label='CT6_reduced', fileName='CT6_reduced_cellParams.pkl') # (2) & (3) #cellName
# del netParams.cellParams['CT6_reduced']['conds']

## FOR PY FILES: 
netParams.importCellParams(label='ITS4_reduced', conds={'cellType':'ITS', 'cellModel':'HH_reduced'}, fileName = 'ITS4_ORIG.py', cellName = 'ITS4_cell')




## Add in Stimulation Source (IClamp) 
netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 250, 'dur': 1000, 'amp': cfg.amp} 
netParams.stimTargetParams['Input->TestPop'] = {'source': 'Input', 'sec':'soma', 'loc': 0.5, 'conds': {'pop':'TestPop'}}

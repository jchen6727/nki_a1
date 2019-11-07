from netpyne import specs, sim

try:
	from __main__ import cfg  # import SimConfig object with params from parent module
except:
	from cfg import cfg  # if no simConfig in parent module, import directly from tut8_cfg module


netParams = specs.NetParams() 
netParams.popParams['NGF_pop'] = {'cellType': 'ngf', 'numCells': 1, 'cellModel': 'NGF'}


cellRule = netParams.importCellParams(label='NGF_Rule', conds={'cellType': 'ngf', 'cellModel': 'NGF'}, fileName='ngf_cell.hoc', cellName='ngfcell', importSynMechs = True)

## SOMA
cellRule['secs']['soma']['vinit'] = -66.9 #-65.6
cellRule['secs']['soma']['mechs']['pas']['g'] =  5e-7 #5e-7
cellRule['secs']['soma']['mechs']['pas']['e'] = -85
cellRule['secs']['soma']['mechs']['hd']['gbar'] = 1e-05
cellRule['secs']['soma']['geom']['cm'] = 1.5 #0.5

## DEND 
cellRule['secs']['dend']['vinit'] = -66.9 #-65.6
cellRule['secs']['dend']['mechs']['pas']['g'] = 3e-4 #2e-4
#cellRule['secs']['dend']['mechs']['pas']['g'] = 5e-6
#cellRule['secs']['dend']['mechs']['pas']['e'] = -85
#cellRule['secs']['dend']['mechs']['hd']['gbar'] = 1e-05
#cellRule['secs']['dend']['geom']['cm'] = 0.5

## Add in Stimulation Source (IClamp) 
netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 250, 'dur': 1000, 'amp': cfg.amp} 
netParams.stimTargetParams['Input->NGF_pop'] = {'source': 'Input', 'sec':'soma', 'loc': 0.5, 'conds': {'pop':'NGF_pop'}}

	# THIS SCRIPT CONTAINS THE FIXED NETWORK PARAMETERS FOR EACH RUN 
from netpyne import specs, sim
try:
	from __main__ import cfg  # import SimConfig object with params from parent module
except:
	from NGF_cfg import cfg  # if no simConfig in parent module, import directly 

# Network parameters
netParams = specs.NetParams()  # object of class NetParams to store the network parameters

# Population Parameters
netParams.popParams['NGF_pop'] = {'cellType': 'NGF', 'numCells': 1, 'cellModel': 'NGF_hh'}

## Import template file
cellRule = netParams.importCellParams(label = 'NGF_rule', conds = {'cellType': 'NGF', 'cellModel': 'NGF_hh'}, fileName='ngf_cell.hoc', cellName = 'ngfcell')
# # Insert candidates from passive optimization scheme 
cellRule['secs']['soma']['vinit'] = -74 #-65 # may need to use this in conjunction with other values to set RMP 


## Add in Stimulation Source (IClamp) 
netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 100, 'dur': 1000, 'amp': cfg.amp} 
netParams.stimTargetParams['Input->NGF'] = {'source': 'Input', 'sec':'soma', 'loc': 0.5, 'conds': {'pop':'NGF_pop'}}


##########
# CHANGE THESE!! parameters to be modified 
cellRule['secs']['soma']['mechs']['hh2']['gnabar']=0.0023484731943662143
cellRule['secs']['soma']['mechs']['hh2']['gkbar']=0.0002
cellRule['secs']['soma']['mechs']['im']['taumax']=574.4074029847205
cellRule['secs']['soma']['mechs']['im']['gkbar']=7.5e-05
cellRule['secs']['soma']['mechs']['pas']['g']=1.170084859132038e-05
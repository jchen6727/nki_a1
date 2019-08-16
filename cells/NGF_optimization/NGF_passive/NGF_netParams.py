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
#cellRule['secs']['soma']['vinit'] = -74


## Add in Stimulation Source (IClamp) 
netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 100, 'dur': 1000, 'amp': cfg.amp} 
netParams.stimTargetParams['Input->NGF_pop'] = {'source': 'Input', 'sec':'soma', 'loc': 0.5, 'conds': {'pop':'NGF_pop'}}


##########
# CHANGE THESE!! parameters to be modified 
cellRule['NGF_Rule']['secs']['soma']['mechs']['ch_Navngf']['gmax'] = 3.8826035386091324
cellRule['NGF_Rule']['secs']['soma']['mechs']['ch_Navngf']['ena'] = 79.90870174183883
cellRule['NGF_Rule']['secs']['soma']['geom']['cm'] =  1.1742365971831072
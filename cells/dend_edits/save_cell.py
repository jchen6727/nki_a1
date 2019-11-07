import pickle
from netpyne import specs

def pkl_tuned(cellName):
	netParams = specs.NetParams()
	#netParams.loadCellParamsRule(label=cellName, fileName=cellName+'_tuned_cellParams.pkl')
	netParams.loadCellParamsRule(label=cellName, fileName=cellName+'_new_cellParams.pkl')
	del netParams.cellParams[cellName]['conds']

	## V INIT
	netParams.cellParams[cellName]['globals']['v_init'] = -86
	netParams.cellParams[cellName]['secs']['soma']['vinit'] = -86
	netParams.cellParams[cellName]['secs']['Adend1']['vinit'] = -86
	netParams.cellParams[cellName]['secs']['Adend2']['vinit'] = -86
	netParams.cellParams[cellName]['secs']['Adend3']['vinit'] = -86
	netParams.cellParams[cellName]['secs']['Bdend']['vinit'] = -86
	netParams.cellParams[cellName]['secs']['axon']['vinit'] = -86

	## PASSIVE ##
	netParams.cellParams[cellName]['secs']['soma']['mechs']['pas']['g'] = 1.3e-04 # ORIG: 9.442294539558377e-05
	#netParams.cellParams[cellName]['secs']['soma']['mechs']['pas']['e'] = -77 # ORIG: -87.1335623948
	
	## ACTIVE ##
	netParams.cellParams[cellName]['secs']['soma']['mechs']['nax']['gbar'] = 0.116	# ORIG: 0.0768253702194
	netParams.cellParams[cellName]['secs']['soma']['mechs']['kdr']['gbar'] = 0.0073 	# ORIG: 0.00833766634808
	#netParams.cellParams[cellName]['secs']['soma']['mechs']['kdr']['vhalfn'] = 14 # ORIG: 11.6427471384

	# ynorm
	#netParams.cellParams[cellName]['conds']['ynorm'] = [0.475, 0.625]

	netParams.saveCellParamsRule(label = cellName, fileName=cellName+'_tuned_cellParams.pkl')
	#netParams.saveCellParamsRule(label = cellName, fileName=cellName+'_renorm_cellParams.pkl')
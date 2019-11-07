import pickle
from netpyne import specs, sim
import matplotlib.pyplot as plt

def dend_info(cellName):
	netParams = specs.NetParams()
	netParams.loadCellParamsRule(label=cellName, fileName=cellName+'_cellParams.pkl')
	sections = list(netParams.cellParams[cellName]['secs'].keys())
	#print(sections)
	apical_dends = []
	for sec in sections:
		if 'Adend' in sec:
			apical_dends.append(sec)
			print(sec)

	num_apical_dends = len(apical_dends)

	dend_lengths = []
	pt3d_info = []
	for dend in apical_dends:
		L = netParams.cellParams[cellName]['secs'][dend]['geom']['L']
		dend_lengths.append(L)
		pt3d_start = netParams.cellParams[cellName]['secs'][dend]['geom']['pt3d'][0][1]
		pt3d_info.append(dend + ' start: ' + str(pt3d_start))
		pt3d_end = netParams.cellParams[cellName]['secs'][dend]['geom']['pt3d'][1][1]
		pt3d_info.append(dend + ' end: ' + str(pt3d_end))

	total_dend_length = sum(dend_lengths)
	
	return num_apical_dends, total_dend_length, pt3d_info




def change_geom(cellName, newLength):
	netParams = specs.NetParams()
	netParams.loadCellParamsRule(label=cellName, fileName=cellName+'_cellParams.pkl')

	## This will produce a list w/ the names of the apical dendrite
	sections = list(netParams.cellParams[cellName]['secs'].keys())
	apical_dends = []
	for sec in sections:
		if 'Adend' in sec:
			apical_dends.append(sec)
			#print(sec)
	num_apical_dends = len(apical_dends)

	soma_length = netParams.cellParams[cellName]['secs']['soma']['geom']['pt3d'][1][1]
	dend_length = newLength-soma_length
	if dend_length < 0:
		print(' --- WARNING WARNING WARNING: DENDRITE LENGTH IS LESS THAN 0 ---')

	## This will produce 2 dictionaries, storing orig lengths and pt3d info for each apical dend
	orig_lengths = {}
	orig_pt3d = {}
	for dend in apical_dends:
		dend_L = netParams.cellParams[cellName]['secs'][dend]['geom']['L']
		dend_pt3d = netParams.cellParams[cellName]['secs'][dend]['geom']['pt3d']
		orig_lengths[dend] = dend_L
		orig_pt3d[dend] = dend_pt3d


	## This will put the new lengths for each dend into the netParams dict
	for n in range(len(apical_dends)):
		# CHANGE THE LENGTH FOR ALL DENDS 
		netParams.cellParams[cellName]['secs'][apical_dends[n]]['geom']['L'] = dend_length/num_apical_dends
		#netParams.cellParams[cellName]['secs'][apical_dends[n]]['geom']['pt3d'][0][0]
		if n == 0:
			netParams.cellParams[cellName]['secs'][apical_dends[n]]['geom']['pt3d'][1][1] = \
			netParams.cellParams[cellName]['secs'][apical_dends[n]]['geom']['pt3d'][0][1] + dend_length/num_apical_dends
		else:
			netParams.cellParams[cellName]['secs'][apical_dends[n]]['geom']['pt3d'][0][1] = \
			netParams.cellParams[cellName]['secs'][apical_dends[n-1]]['geom']['pt3d'][1][1]
			netParams.cellParams[cellName]['secs'][apical_dends[n]]['geom']['pt3d'][1][1] = \
			netParams.cellParams[cellName]['secs'][apical_dends[n]]['geom']['pt3d'][0][1] + dend_length/num_apical_dends



	netParams.saveCellParamsRule(label = cellName, fileName=cellName+'_new_cellParams.pkl')




def test_func(cellName,cellType,redo):
	netParams = specs.NetParams()
	if redo == 0:
		netParams.loadCellParamsRule(label=cellName, fileName=cellName+'_cellParams.pkl')
	if redo == 1:
		netParams.loadCellParamsRule(label=cellName, fileName=cellName+'_new_cellParams.pkl')
	
	netParams.popParams['TestPop'] = {'cellType': cellType, 'numCells': 1, 'cellModel': 'HH_reduced'}

	netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 250, 'dur': 1000, 'amp': 2} 
	netParams.stimTargetParams['Input-->TestPop'] = {'source': 'Input', 'sec':'soma', 'loc': 0.5, 'conds': {'pop':'TestPop'}}
	
	simConfig = specs.SimConfig()					# object of class SimConfig to store simulation configuration
	simConfig.duration = 1.5*1e3 #1*1e3 						# Duration of the simulation, in ms
	simConfig.dt = 0.02								# Internal integration timestep to use
	simConfig.verbose = 1							# Show detailed messages 
	simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
	simConfig.recordStep = 0.02 			
	simConfig.filename = 'model_output'  			# Set file output name
	simConfig.analysis['plotTraces'] = {'include': [0]} # Plot recorded traces for this list of cells


	## Create network and run simulation
	sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)
	#plt.show()



#### MAIN CODE ####
# L1 = [0,100]
# L2 = [100,160]
# L3 = [160,950]
# L4 = [950,1250]
# L5A = [1250,1334]
# L5B = [1334,1550]
# L6 = [1550,2000]
















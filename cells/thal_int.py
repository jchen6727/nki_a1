from netpyne import specs, sim
from neuron import h
#import matplotlib.pyplot as plt


netParams = specs.NetParams()

## CREATE POP 
netParams.popParams['sTI'] = {'cellModel': 'HH', 'cellType': 'TI', 'numCells': 1} 

#########################
###### CREATE CELL ######  
#########################
cellRule = {'conds': {'cellModel': 'HH', 'cellType': 'TI'}, 'secs':{}, 'globals': {}} # cell rule dict

#### SOMA #### 
cellRule['secs']['soma'] = {'geom': {}, 'mechs': {}, 'ions': {}}
cellRule['secs']['soma']['geom'] = {'diam': 10, 'L': 16, 'Ra': 120, 'cm': 1} #Ra: 100 


#### PROXIMAL DENDRITES ####
prox_dends = ['dend_%d' % i for i in range(1)]

for prox_dend in prox_dends:
	cellRule['secs'][prox_dend] = {'geom':{}, 'mechs':{}} 
	cellRule['secs'][prox_dend]['geom'] = {'diam': 3.25, 'L': 240, 'Ra': 120, 'cm': 1} #'Ra': 100

cellRule['secs']['dend_0']['topol'] = {'parentSec': 'soma', 'parentX': 0, 'childX': 0}
# cellRule['secs']['dend_1']['topol'] = {'parentSec': 'soma', 'parentX': 1, 'childX': 0}






#############################################
############# INSERT MECHANISMS #############
#############################################


######################
#### LEAK CURRENT ####
######################
## SOMA ## 
cellRule['secs']['soma']['mechs']['Pass'] ={'g': 13e-06, 'erev':-74}
## trial values: 
# cellRule['secs']['soma']['mechs']['Pass']['g'] =   		# ORIG in STI.hoc: 18e-6 ; 'g' from jun.pdf: 8e-03
# cellRule['secs']['soma']['mechs']['Pass']['erev'] = 		# ORIG in sTI.hoc: -72.5 

## PROXIMAL DENDRITES ## 
for prox_dend in prox_dends:
	cellRule['secs'][prox_dend]['mechs']['Pass'] = {'g': 13e-06, 'erev': -74}


#####################
#### FAST SODIUM ####
#####################
cellRule['secs']['soma']['mechs']['naf2'] = {'gmax': 0.1, 'mvhalf': -40, 'mvalence': 5, 'hvhalf': -43, 'hvalence': -6}
## mvhalf in orig sTI.hoc is set to -45, but in thesis, mvhalf = -40 
## trial line: 
# cellRule['secs']['soma']['mechs']['naf2']['gmax'] = 		# ORIG?? 

#####################################
#### POTASSIUM DELAYED RECTIFIER ####
#####################################
cellRule['secs']['soma']['mechs']['kdr2orig'] = {'gmax': 0.1, 'mvhalf': -31, 'mvalence': 3.8} # ORIG in thesis: 0.05 # ORIG in sTI.hoc 0.07 
## USING ORIGINAL 
# 'mexp': 4 ## SHOULD BE 4, BUT LOOKS LIKE SET TO 1 IF LOOK AT THESIS... 
# 'mbaserate': 0.0008,
# 'mgamma': 0.5,
# 'mbasetau': 10
cellRule['secs']['soma']['ions']['k'] = {'e': -95, 'i': 54.5, 'o': 2.5} # set Ek = -95 


####################
#### IH current ####
####################
cellRule['secs']['soma']['mechs']['iar'] = {'ghbar': 0.7e-04, 'shift': -0.0} # ORIG ghbar: 0.13 mS/cm2; correct re: jun.pdf


######################
#### ICAN current ###
######################
cellRule['secs']['soma']['mechs']['ican'] = {'gbar': 0.0001, 'ratc': 0.8, 'ratC': 0.1} #gbar: 0.0003 # ratc: proportion coming from low-thresh pool (IT) # ratC: proportion coming from high-thresh pool (IL)
cellRule['globals']['beta_ican'] = 0.003 		# correct re: jun.pdf & thesis (units: ms^-1)
cellRule['globals']['cac_ican'] = 1.1e-04		# alpha = beta / (cac^x), alpha = 1.4e05 ms^-1 microM^-8
cellRule['globals']['x_ican'] = 8				# correct re: jun.pdf, if x_ican == "n" (yes)
## trial line: 
# cellRule['secs']['soma']['mechs']['ican']['gbar'] = 		# ORIG in THESIS: 1mS/cm2 # ORIG in sTI.hoc: 2e-05  #1e-7?? 
# cellRule['globals']['cac_ican'] = 

######################
#### IAHP current ####
######################
cellRule['secs']['soma']['mechs']['iahp'] = {'gkbar': 0.45, 'ratc': 0.2, 'ratC': 1} #gkbar: 0.18
cellRule['globals']['beta_iahp'] = 0.02 		# correct re: jun.pdf
cellRule['globals']['cac_iahp'] = 8e-04			
## trial line: 
# cellRule['secs']['soma']['mechs']['iahp']['gkbar'] = 		# ORIG in sTI.hoc: 0.4 



####################
#### IT current ####
####################
cellRule['secs']['soma']['mechs']['it2'] = {'gcabar': 0.4e-04, 'shift1': 7}
cellRule['globals']['shift2_it2'] = 0 		# correct re: jun.pdf & thesis 
cellRule['globals']['mx_it2'] = 3.0			# correct re: jun.pdf & thesis
cellRule['globals']['hx_it2'] = 1.5			# correct re: jun.pdf & thesis
cellRule['globals']['sm_it2'] = 4.8 		# correct re: jun.pdf & thesis
cellRule['globals']['sh_it2'] = 4.6			# correct re: jun.pdf & thesis
## trial line: 
# cellRule['secs']['soma']['mechs']['it2']['gcabar'] = 		# ORIG in sTI.hoc: 4.0e-04


#############################################################
#### CALCIUM PUMP FOR "ca" ION POOL - associated with IT ####
#############################################################
cellRule['secs']['soma']['mechs']['cad'] = {'taur': 150, 'taur2': 80, 'cainf': 1e-8, 'cainf2': 5.3e-5, 'kt': 0e-6, 'kt2': 0e-7, 'k': 7.5e-3}
## TRIAL VALUES: 
#cellRule['secs']['soma']['mechs']['cad']['k'] = 7.5e-3 #50					# ORIG: 7.5e-3

####################
#### IL current ####
####################
cellRule['secs']['soma']['mechs']['ical'] = {'pcabar': 0.00009}
cellRule['globals']['sh1_ical'] = -10
cellRule['globals']['sh2_ical'] = 0
## trial line: 
# cellRule['secs']['soma']['mechs']['ical']['pcabar'] = 			# ORIG: 9.0e-04

##############################################################
#### CALCIUM PUMP FOR "Ca" ION POOL -- associated with IL ####
##############################################################
cellRule['secs']['soma']['mechs']['Cad'] = {'taur': 150, 'taur2': 80, 'Cainf': 1e-8, 'Cainf2': 5.2e-5, 'kt': 0e-6, 'kt2': 0e-7, 'k': 5e-3}
## TRIAL VALUES:
#cellRule['secs']['soma']['mechs']['Cad']['k'] = 7.5e-3 #50					# ORIG: 7.5e-3





#############################################
########### CHANGE ION PARAMETERS ###########
#############################################

### K2 ION VALUES FOR IAHP CURRENT ### /significance of this is debatable/ ###
cellRule['secs']['soma']['ions']['k2'] = {'e': -95, 'i': 54.5, 'o': 2.5} # -77 by default, but set to -95 
cellRule['globals']['k2i0_k2_ion'] = 54.5 			# NEURON: h.k2i0_k2_ion = 54.5
cellRule['globals']['k2o0_k2_ion'] = 2.5 			# NEURON: h.k2o0_k2_ion = 2.5 


### Ca2+ ION VALUES FOR Cad.mod & IL CURRENT ### 
cellRule['globals']['Cai0_Ca_ion'] = 5e-5			# NEURON: h.Cai0_Ca_ion = 5e-5
cellRule['globals']['Cao0_Ca_ion'] = 2				# NEURON: h.Cao0_Ca_ion = 2


########################################
## PUT CELLRULE INTO NETPARAMS STRUCT ##
########################################
netParams.cellParams['sTI'] = cellRule



###############################
######### STIM OBJECT #########
###############################
netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 400, 'dur': 1000, 'amp': 0.06}
netParams.stimTargetParams['Input->sTI'] = {'source': 'Input', 'sec':'soma', 'loc': 0.5, 'conds': {'pop':'sTI'}}




#########################
######## RUN cfg ########
#########################
cfg = specs.SimConfig()					# object of class SimConfig to store simulation configuration
cfg.duration = 1.8e3 						# Duration of the simulation, in ms
cfg.dt = 0.025 #0.01 								# Internal integration timestep to use
cfg.verbose = 1							# Show detailed messages 
cfg.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}, \
					#'ina_soma': {'sec':'soma', 'loc':0.5, 'var':'ina'}, \
					#'ik_soma': {'sec':'soma', 'loc':0.5, 'var':'ik'}}
					#'IH':{'sec':'soma','loc':0.5,'var':'iother'},\
					'IT':{'sec':'soma','loc':0.5,'var':'ica'},\
					'Cai':{'sec':'soma','loc':0.5,'var':'cai'},\
					'IL':{'sec':'soma','loc':0.5,'var':'iCa'},\
					'ICAN':{'sec':'soma','loc':0.5,'var':'iother2'},\
					'cai':{'sec':'soma','loc':0.5,'var':'Cai'},\
					'IAHP':{'sec':'soma','loc':0.5,'var':'ik2'}} ### SWITCHED cai and Cai labels to be consistent with jun
cfg.recordStep = 0.025 			
cfg.filename = 'model_output'  			# Set file output name
cfg.saveJson = True
cfg.analysis['plotTraces'] = {'include': [0], 'saveFig': True, 'overlay': False, 'axis':'on'} #, 'oneFigPer':'trace'} # Plot recorded traces for this list of cells
cfg.hParams = {'celsius': 36} #'v_init': -75}		# celsius from batch_.hoc was 37?




###############################
### CREATE & RUN SIMULATION ###
###############################
sim.createSimulateAnalyze(netParams = netParams, simConfig = cfg)










#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:00:01 2019

@author: ricky
"""

from netpyne import specs, sim

# NETWORK PARAMETERS
netParams = specs.NetParams()

## POPULATION PARAMETERS
### Cell types listed below come from labels.py -- NOT OFFICIAL YET! 2/15/19
### numCells should come from mpisim.py, cpernet, see Google Drive doc 
#### Regular Spiking (RS)
netParams.popParams['E2'] = {'cellType': 'RS', 'numCells': 329, 'cellModel': ''}
netParams.popParams['E4'] = {'cellType': 'RS', 'numCells': 140, 'cellModel': ''}
netParams.popParams['E5R'] = {'cellType': 'RS', 'numCells': 18, 'cellModel': ''}
netParams.popParams['E6'] = {'cellType': 'RS', 'numCells': 170, 'cellModel': ''}
#### Fast Spiking (FS)
netParams.popParams['I2'] = {'cellType': 'FS', 'numCells': 54, 'cellModel': ''}
netParams.popParams['I4'] = {'cellType': 'FS', 'numCells': 23, 'cellModel': ''}
netParams.popParams['I5'] = {'cellType': 'FS', 'numCells': 6, 'cellModel': ''}
netParams.popParams['I6'] = {'cellType': 'FS', 'numCells': 28, 'cellModel': ''}
#### Low Threshold Spiking (LTS)
netParams.popParams['I2L'] = {'cellType': 'LTS', 'numCells': 54, 'cellModel': ''}
netParams.popParams['I4L'] = {'cellType': 'LTS', 'numCells': 23, 'cellModel': ''}
netParams.popParams['I5L'] = {'cellType': 'LTS', 'numCells': 6, 'cellModel': ''}
netParams.popParams['I6L'] = {'cellType': 'LTS', 'numCells': 28, 'cellModel': ''}
#### Burst Firing (Burst)
netParams.popParams['E5B'] = {'cellType': 'Burst', 'numCells': 26, 'cellModel': ''}
### THALAMUS (core)-- excluded for now per samn (2/17/19)
# netParams.popParams['TC'] = {'cellType': 'thal_core', 'numCells': 0, 'cellModel': ''}
# netParams.popParams['HTC'] = {'cellType': 'thal_core', 'numCells': 0, 'cellModel': ''}
# netParams.popParams['IRE'] = {'cellType': 'thal_core', 'numCells': 0, 'cellModel': ''}
### THALAMUS (matrix) -- excluded for now per samn (2/17/19)
#netParams.popParams['TCM'] = {'cellType': 'thal_matrix', 'numCells': 0, 'cellModel': ''}
#netParams.popParams['IREM'] = {'cellType': 'thal_matrix', 'numCells': 0, 'cellModel': ''}



## CELL PROPERTY RULES
### These labels are all placeholders -- fill in when have more info about cell types 
cellRule = {'conds': {'cellType': 'RS'}, 'secs': {}}
cellRule = {'conds': {'cellType': 'FS'}, 'secs': {}}
cellRule = {'conds': {'cellType': 'LTS'}, 'secs': {}}
cellRule = {'conds': {'cellType': 'Burst'}, 'secs': {}}
#cellRule = {'conds': {'cellType': 'thal_core'}, 'secs': {}}
#cellRule = {'conds': {'cellType': 'thal_matrix'}, 'secs': {}}

## SYNAPTIC MECHANISM PARAMETERS
### wmat in mpisim.py & STYP in labels.py  
netParams.synMechParams['AM2'] = 
netParams.synMechParams['GA'] = 
netParams.synMechParams['GA2'] = 

## STIMULATION PARAMETERS
### bkg and bkg-->? are just placeholders 2/15/19
netParams.stimSourceParams['bkg']
netParams.stimSourceParams['bkg --> ?']

## CELL CONNECTIVITY RULES
### probability from pmat in mpisim.py & weights / synapse types from wmat in mpisim.py & STYP in labels.py  
#### Intracortical
##### Presynaptic I2L
netParams.connParams['I2L -> E2'] = {
        'preConds': {'pop':'I2L'}, 
        'postConds': {'pop':'E2'},
        'probability': 0.35,
        'weight': 0.83, # * 37.5 # 150 # *35 # new test increase in weight
        'delay':,
        'synMech': 'GA2'} 
netParams.connParams['I2L -> E5B'] = {
        'preConds': {'pop':'I2L'}, 
        'postConds': {'pop':'E5B'},
        'probability': 0.5,
        'weight': 0.83,
        'delay':,
        'synMech': 'GA2'} 
netParams.connParams['I2L -> E5R'] = {
        'preConds': {'pop':'I2L'}, 
        'postConds': {'pop':'E5R'},
        'probability': 0.35,
        'weight': 0.83,
        'delay':,
        'synMech': 'GA2'} 
netParams.connParams['I2L -> E6'] = {
        'preConds': {'pop':'I2L'}, 
        'postConds': {'pop':'E6'},
        'probability': 0.25,
        'weight': 0.83,
        'delay':,
        'synMech': 'GA2'} 
netParams.connParams['I2L -> I2L'] = {
        'preConds': {'pop':'I2L'}, 
        'postConds': {'pop':'I2L'},
        'probability': 0.09,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA2'} 
netParams.connParams['I2L -> I2'] = {
        'preConds': {'pop':'I2L'}, 
        'postConds': {'pop':'I2'},
        'probability': 0.53,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA2'} 
netParams.connParams['I2L -> I5'] = {
        'preConds': {'pop':'I2L'}, 
        'postConds': {'pop':'I5'},
        'probability': 0.53,
        'weight': 0.83,
        'delay':,
        'synMech': 'GA2'} 
netParams.connParams['I2L -> I6'] = {
        'preConds': {'pop':'I2L'}, 
        'postConds': {'pop':'I6'},
        'probability': 0.53,
        'weight': 0.83,
        'delay':,
        'synMech': 'GA2'}
 
##### Presynaptic I2
netParams.connParams['I2 -> E2'] = {
        'preConds': {'pop':'I2'}, 
        'postConds': {'pop':'E2'},
        'probability': 0.44,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA'} 
netParams.connParams['I2 -> I2L'] = {
        'preConds': {'pop':'I2'}, 
        'postConds': {'pop':'I2L'},
        'probability': 0.34,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA'} 
netParams.connParams['I2 -> I2'] = {
        'preConds': {'pop':'I2'}, 
        'postConds': {'pop':'I2'},
        'probability': 0.62,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA'}

##### Presynaptic I4L
netParams.connParams['I4L -> E4'] = {
        'preConds': {'pop':'I4L'}, 
        'postConds': {'pop':'E4'},
        'probability': 0.35,
        'weight': 0.83,
        'delay':,
        'synMech': 'GA2'}  
netParams.connParams['I4L -> I4L'] = {
        'preConds': {'pop':'I4L'}, 
        'postConds': {'pop':'I4L'},
        'probability': 0.09,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA2'}  
netParams.connParams['I4L -> I4'] = {
        'preConds': {'pop':'I4L'}, 
        'postConds': {'pop':'0.53'},
        'probability': 0.53,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA2'}  

##### Presynaptic I4
netParams.connParams['I4 -> E4'] = {
        'preConds': {'pop':'I4'}, 
        'postConds': {'pop':'E4'},
        'probability': 0.44,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA'}  
netParams.connParams['I4 -> I4L'] = {
        'preConds': {'pop':'I4'}, 
        'postConds': {'pop':'I4L'},
        'probability': 0.34,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA'}  
netParams.connParams['I4 -> I4'] = {
        'preConds': {'pop':'I4'}, 
        'postConds': {'pop':'I4'},
        'probability': 0.62,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA'}  

##### Presynaptic I5L
netParams.connParams['I5L -> E2'] = {
        'preConds': {'pop':'I5L'}, 
        'postConds': {'pop':'E2'},
        'probability': 0.35,
        'weight': 0.83,
        'delay':,
        'synMech': 'GA2'}  
netParams.connParams['I5L -> E5B'] = {
        'preConds': {'pop':'I5L'}, 
        'postConds': {'pop':'E5B'},
        'probability': 0.35,
        'weight': 0.83,
        'delay':,
        'synMech': 'GA2'}  
netParams.connParams['I5L -> E5R'] = {
        'preConds': {'pop':'I5L'}, 
        'postConds': {'pop':'E5R'},
        'probability': 0.35,
        'weight': 0.83,
        'delay':,
        'synMech': 'GA2'}  
netParams.connParams['I5L -> E6'] = {
        'preConds': {'pop':'I5L'}, 
        'postConds': {'pop':'E6'},
        'probability': 0.25,
        'weight': 0.83,
        'delay':,
        'synMech': 'GA2'}  
netParams.connParams['I5L -> I2'] = {
        'preConds': {'pop':'I5L'}, 
        'postConds': {'pop':'I2'},
        'probability': 0.53,
        'weight': 0.83,
        'delay':,
        'synMech': 'GA2'}  
netParams.connParams['I5L -> I5L'] = {
        'preConds': {'pop':'I5L'}, 
        'postConds': {'pop':'I5L'},
        'probability': 0.09,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA2'}  
netParams.connParams['I5L -> I5'] = {
        'preConds': {'pop':'I5L'}, 
        'postConds': {'pop':'I5'},
        'probability': 0.53,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA2'}  
netParams.connParams['I5L -> I6'] = {
        'preConds': {'pop':'I5L'}, 
        'postConds': {'pop':'I6'},
        'probability': 0.53,
        'weight': 0.83,
        'delay':,
        'synMech': 'GA2'}  


##### Presynaptic I5
netParams.connParams['I5 -> E5B'] = {
        'preConds': {'pop':'I5'}, 
        'postConds': {'pop':'E5B'},
        'probability': 0.44,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA'}  
netParams.connParams['I5 -> E5R'] = {
        'preConds': {'pop':'I5'}, 
        'postConds': {'pop':'E5R'},
        'probability': 0.44,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA'} 
netParams.connParams['I5 -> I5L'] = {
        'preConds': {'pop':'I5'}, 
        'postConds': {'pop':'I5L'},
        'probability': 0.34,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA'} 
netParams.connParams['I5 -> I5'] = {
        'preConds': {'pop':'I5'}, 
        'postConds': {'pop':'I5'},
        'probability': 0.62,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA'} 


##### Presynaptic I6L
netParams.connParams['I6L -> E2'] = {
        'preConds': {'pop':'I6L'}, 
        'postConds': {'pop':'E2'},
        'probability': 0.35,
        'weight': 0.83,
        'delay':,
        'synMech': 'GA2'} 
netParams.connParams['I6L -> E5B'] = {
        'preConds': {'pop':'I6L'}, 
        'postConds': {'pop':'E5B'},
        'probability': 0.25,
        'weight': 0.83,
        'delay':,
        'synMech': 'GA2'} 
netParams.connParams['I6L -> E5R'] = {
        'preConds': {'pop':'I6L'}, 
        'postConds': {'pop':'E5R'},
        'probability': 0.25,
        'weight': 0.83,
        'delay':,
        'synMech': } 
netParams.connParams['I6L -> E6'] = {
        'preConds': {'pop':'I6L'}, 
        'postConds': {'pop':'E6'},
        'probability': 0.35,
        'weight': 0.83,
        'delay':,
        'synMech': } 
netParams.connParams['I6L -> I2'] = {
        'preConds': {'pop':'I6L'}, 
        'postConds': {'pop':'I2'},
        'probability': 0.53,
        'weight': 0.83,
        'delay':,
        'synMech': } 
netParams.connParams['I6L -> I5'] = {
        'preConds': {'pop':'I6L'}, 
        'postConds': {'pop':'I5'},
        'probability': 0.53,
        'weight': 0.83,
        'delay':,
        'synMech': } 
netParams.connParams['I6L -> I6L'] = {
        'preConds': {'pop':'I6L'}, 
        'postConds': {'pop':'I6L'},
        'probability': 0.09,
        'weight': 1.5,
        'delay':,
        'synMech': } 
netParams.connParams['I6L -> I6'] = {
        'preConds': {'pop':'I6L'}, 
        'postConds': {'pop':'I6'},
        'probability': 0.53,
        'weight': 1.5,
        'delay':,
        'synMech': } 

##### Presynaptic I6
netParams.connParams['I6 -> E6'] = {
        'preConds': {'pop':'I6'}, 
        'postConds': {'pop':'E6'},
        'probability': 0.44,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA'} 
netParams.connParams['I6 -> I6L'] = {
        'preConds': {'pop':'I6'}, 
        'postConds': {'pop':'I6L'},
        'probability': 0.34,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA'} 
netParams.connParams['I6 -> I6'] = {
        'preConds': {'pop':'I6'}, 
        'postConds': {'pop':'I6'},
        'probability': 0.62,
        'weight': 1.5,
        'delay':,
        'synMech': 'GA'} 


##### Presynaptic E2
netParams.connParams['E2 -> E2'] = {
        'preConds': {'pop':'E2'}, 
        'postConds': {'pop':'E2'},
        'probability': 0.2, #"weak by wiring matrix in Weiler et al., 2008"
        'weight': 0.66,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E2 -> E4'] = {
        'preConds': {'pop':'E2'}, 
        'postConds': {'pop':'E4'},
        'probability': 0.024,
        'weight': 0.36,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E2 -> E5B'] = {
        'preConds': {'pop':'E2'}, 
        'postConds': {'pop':'E5B'},
        'probability': 0.8, #"strong by wiring matrix in Weiler et al., 2008"
        'weight': 0.26,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E2 -> E5R'] = {
        'preConds': {'pop':'E2'}, 
        'postConds': {'pop':'E5R'},
        'probability': 0.8, #"strong by wiring matrix in Weiler et al., 2008"
        'weight': 0.93,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E2 -> I5L'] = {
        'preConds': {'pop':'E2'}, 
        'postConds': {'pop':'I5L'},
        'probability': 0.51, #"L2/3 E -> L5 LTS justified by Apicella et al., 2012
        'weight': 0.36,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E2 -> E6'] = {
        'preConds': {'pop':'E2'}, 
        'postConds': {'pop':'E6'},
        'probability': 0, #"none by wiring matrix in Weiler et al., 2008"
        'weight': 0,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E2 -> I2L'] = {
        'preConds': {'pop':'E2'}, 
        'postConds': {'pop':'I2L'},
        'probability': 0.51,
        'weight': 0.23,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E2 -> I2'] = {
        'preConds': {'pop':'E2'}, 
        'postConds': {'pop':'I2'},
        'probability': 0.43,
        'weight': 0.23,
        'delay':,
        'synMech': 'AM2'} 


##### Presynaptic E4
netParams.connParams['E4 -> E2'] = {
        'preConds': {'pop':'E4'}, 
        'postConds': {'pop':'E2'},
        'probability': 0.145,
        'weight': 0.58,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E4 -> E4'] = {
        'preConds': {'pop':'E4'}, 
        'postConds': {'pop':'E4'},
        'probability': 0.243,
        'weight': 0.95,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E4 -> E5B'] = {
        'preConds': {'pop':'E4'}, 
        'postConds': {'pop':'E5B'},
        'probability': 0.122,
        'weight': 1.01,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E4 -> E5R'] = {
        'preConds': {'pop':'E4'}, 
        'postConds': {'pop':'E5R'},
        'probability': 0.116,
        'weight': 0.54,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E4 -> E6'] = {
        'preConds': {'pop':'E4'}, 
        'postConds': {'pop':'E6'},
        'probability': 0.032,
        'weight': 2.27,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E4 -> I4L'] = {
        'preConds': {'pop':'E4'}, 
        'postConds': {'pop':'I4L'},
        'probability': 0.51,
        'weight': 0.23,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E4 -> I4'] = {
        'preConds': {'pop':'E4'}, 
        'postConds': {'pop':'I4'},
        'probability': 0.43,
        'weight': 0.23,
        'delay':,
        'synMech': 'AM2'} 

##### Presynaptic E5B
netParams.connParams['E5B -> E2'] = {
        'preConds': {'pop':'E5B'}, 
        'postConds': {'pop':'E2'},
        'probability': 0, #"none by wiring matrix in Weiler et al., 2008"
        'weight': 0.26,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E5B -> E4'] = {
        'preConds': {'pop':'E5B'}, 
        'postConds': {'pop':'E4'},
        'probability': 0.007,
        'weight': 0.17,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E5B -> E5B'] = {
        'preConds': {'pop':'E5B'}, 
        'postConds': {'pop':'E5B'},
        'probability': 0.04*4, #"set using Kiritani et al., 2012 Fig 6D, Table 1, value x 5"
        'weight': 0.66,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E5B -> E5R'] = {
        'preConds': {'pop':'E5B'}, 
        'postConds': {'pop':'E5R'},
        'probability': 0, #"set using Kiritani et al., 2012 Fig 6D, Table 1, value x 5"
        'weight': 0, # pulled from Fig. 6D, Table 1 of (Kiritani et al., 2012)
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E5B -> E6'] = {
        'preConds': {'pop':'E5B'}, 
        'postConds': {'pop':'E6'},
        'probability': 0, #"none by suggestion of Ben and Gordon over phone"
        'weight': 0.66,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E5B -> I5L'] = {
        'preConds': {'pop':'E5B'}, 
        'postConds': {'pop':'I5L'},
        'probability': 0, #"ruled out by Apicella et al., 2012, Fig. 7"
        'weight': 0, # ruled out by (Apicella et al., 2012) Fig. 7
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E5B -> I5'] = {
        'preConds': {'pop':'E5B'}, 
        'postConds': {'pop':'I5'},
        'probability': 0.43,
        'weight': 0.23, # (Apicella et al., 2012) Fig. 7F (weight = 1/2 x weight for E5R->I5)
        'delay':,
        'synMech': 'AM2'} 



##### Presynaptic E5R
netParams.connParams['E5R -> E2'] = {
        'preConds': {'pop':'E5R'}, 
        'postConds': {'pop':'E2'},
        'probability': 0.2, #"weak by wiring matrix in Weiler et al., 2008
        'weight': 0.66,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E5R -> E5B'] = {
        'preConds': {'pop':'E5R'}, 
        'postConds': {'pop':'E5B'},
        'probability': 0.21 * 4, #"need to set using Kiritani et al., 2012, Fig. 6D, Table 1, value x 5"
        'weight': 0.66,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E5R -> E5R'] = {
        'preConds': {'pop':'E5R'}, 
        'postConds': {'pop':'E5R'}, 
        'probability': 0.11*4, #"need to set using Kiritani et al., 2012, Fig. 6D, Table 1, value x 5"
        'weight': 0.66,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E5R -> E6'] = {
        'preConds': {'pop':'E5R'}, 
        'postConds': {'pop':'E6'},
        'probability': 0.2, #"weak by wiring matrix in Weiler et al., 2008
        'weight': 0.66,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E5R -> I5L'] = {
        'preConds': {'pop':'E5R'}, 
        'postConds': {'pop':'I5L'},
        'probability': 0, #"ruled out by Apicella et al., 2012, Fig. 7"
        'weight': 0, # ruled out by (Apicella et al., 2012) Fig. 7
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E5R -> I5'] = {
        'preConds': {'pop':'E5R'}, 
        'postConds': {'pop':'I5'},
        'probability': 0.43,
        'weight': 0.46, # (Apicella et al., 2012) Fig. 7E (weight = 2 x weight for E5B->I5)
        'delay':,
        'synMech': 'AM2'} 


##### Presynaptic E6
netParams.connParams['E6 -> E2'] = {
        'preConds': {'pop':'E6'}, 
        'postConds': {'pop':'E2'},
        'probability': 0, #"none by wiring matrix in Weiler et al., 2008
        'weight': 0,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E6 -> E5B'] = {
        'preConds': {'pop':'E6'}, 
        'postConds': {'pop':'E5B'},
        'probability': 0.2, #"weak by wiring matrix in Weiler et al., 2008
        'weight': 0.66,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E6 -> E5R'] = {
        'preConds': {'pop':'E6'}, 
        'postConds': {'pop':'E5R'},
        'probability': 0.2, #"weak by wiring matrix in Weiler et al., 2008
        'weight': 0.66,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E6 -> E6'] = {
        'preConds': {'pop':'E6'}, 
        'postConds': {'pop':'E6'},
        'probability': 0.2, #"weak by wiring matrix in Weiler et al., 2008
        'weight': 0.66,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E6 -> I6L'] = {
        'preConds': {'pop':'E6'}, 
        'postConds': {'pop':'I6L'},
        'probability': 0.51,
        'weight': 0.23,
        'delay':,
        'synMech': 'AM2'} 
netParams.connParams['E6 -> I6'] = {
        'preConds': {'pop':'E6'}, 
        'postConds': {'pop':'I6'},
        'probability': 0.43,
        'weight': 0.23,
        'delay':,
        'synMech': 'AM2'} 

##### THALAMIC CONNECTIVITY COMMENTED OUT FOR NOW (2/17/19)
#### Intrathalamic
# netParams.connParams['TC -> TC'] = {
#         'preConds': {'pop':'TC'}, 
#         'postConds': {'pop':'TC'},
#         'probability': 0.1,
#         'weight': 0.1,
#         'delay': ,
#         'synMech': 'AM2'} # insert a synMech from synMechParams above 
# netParams.connParams['HTC -> HTC'] = {
#         'preConds': {'pop':'HTC'}, 
#         'postConds': {'pop':'HTC'},
#         'probability': 0.1,
#         'weight': 0.1,
#         'delay': ,
#         'synMech': 'AM2'} 
# netParams.connParams['TC -> HTC'] = {
#         'preConds': {'pop':'TC'}, 
#         'postConds': {'pop':'HTC'},
#         'probability': 0.1,
#         'weight': ,
#         'delay': ,
#         'synMech': } 
# netParams.connParams['HTC -> TC'] = {
#         'preConds': {'pop':'HTC'}, 
#         'postConds': {'pop':'TC'},
#         'probability': 0.1,
#         'weight': ,
#         'delay': ,
#         'synMech': } 
# netParams.connParams['TCM -> TCM'] = {
#         'preConds': {'pop':'TCM'}, 
#         'postConds': {'pop':'TCM'},
#         'probability': 0.1,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['IRE -> IRE'] = {
#         'preConds': {'pop':'IRE'}, 
#         'postConds': {'pop':'IRE'},
#         'probability': 0.1,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['IREM -> IREM'] =  {
#         'preConds': {'pop':'IREM'}, 
#         'postConds': {'pop':'IREM'},
#         'probability': 0.1,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['IRE -> IREM'] = {
#         'preConds': {'pop':'IRE'}, 
#         'postConds': {'pop':'IREM'},
#         'probability': 0.1,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['IREM -> IRE'] = {
#         'preConds': {'pop':'IREM'}, 
#         'postConds': {'pop':'IRE'},
#         'probability': 0.1,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TC -> IREM'] = {
#         'preConds': {'pop':'TC'}, 
#         'postConds': {'pop':'IREM'},
#         'probability': 0.2,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['HTC -> IREM'] = {
#         'preConds': {'pop':'HTC'}, 
#         'postConds': {'pop':'IREM'},
#         'probability': 0.2,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['IREM -> TC'] = {
#         'preConds': {'pop':'IREM'}, 
#         'postConds': {'pop':'TC'},
#         'probability': 0.1,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['IREM -> HTC'] = {
#         'preConds': {'pop':'IREM'}, 
#         'postConds': {'pop':'HTC'},
#         'probability': 0.1,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TCM -> IRE'] = {
#         'preConds': {'pop':'TCM'}, 
#         'postConds': {'pop':'IRE'},
#         'probability': 0.2,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['IRE -> TCM'] = {
#         'preConds': {'pop':'IRE'}, 
#         'postConds': {'pop':'TCM'},
#         'probability': 0.1,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TC -> IRE'] = {
#         'preConds': {'pop':'TC'}, 
#         'postConds': {'pop':'IRE'},
#         'probability': 0.4,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['HTC -> IRE'] = {
#         'preConds': {'pop':'HTC'}, 
#         'postConds': {'pop':'IRE'},
#         'probability': 0.4,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['IRE -> TC'] = {
#         'preConds': {'pop':'IRE'}, 
#         'postConds': {'pop':'TC'},
#         'probability': 0.3,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['IRE -> HTC'] = {
#         'preConds': {'pop':'IRE'}, 
#         'postConds': {'pop':'HTC'},
#         'probability': 0.3,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TCM -> IREM'] = {
#         'preConds': {'pop':'TCM'}, 
#         'postConds': {'pop':'IREM'},
#         'probability': 0.4,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['IREM -> TCM'] = {
#         'preConds': {'pop':'IREM'}, 
#         'postConds': {'pop':'TCM'},
#         'probability': 0.3,
#         'weight': ,
#         'delay':,
#         'synMech': } 




#### Core Thalamocortical
# netParams.connParams['TC-> E4'] = {
#         'preConds': {'pop':'TC'}, 
#         'postConds': {'pop':'E4'},
#         'probability': 0.25,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['HTC -> E4'] = {
#         'preConds': {'pop':'HTC'}, 
#         'postConds': {'pop':'E4'},
#         'probability': 0.25,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TC -> E5B'] = {
#         'preConds': {'pop':'TC'}, 
#         'postConds': {'pop':'E5B'},
#         'probability': , #0.1*thalfctr 
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['HTC -> E5B'] = {
#         'preConds': {'pop':'HTC'}, 
#         'postConds': {'pop':'E5B'},
#         'probability': , #0.1*thalfctr
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TC -> E5R'] = {
#         'preConds': {'pop':'TC'}, 
#         'postConds': {'pop':'E5R'},
#         'probability': , #0.1*thalfctr
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['HTC -> E5R'] = {
#         'preConds': {'pop':'HTC'}, 
#         'postConds': {'pop':'E5R'},
#         'probability': , #0.1*thalfctr
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TC -> E6'] = {
#         'preConds': {'pop':'TC'}, 
#         'postConds': {'pop':'E6'},
#         'probability': , #0.15*thalfctr
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['HTC -> E6'] = {
#         'preConds': {'pop':'HTC'}, 
#         'postConds': {'pop':'E6'},
#         'probability': , #0.15*thalfctr
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TC -> I4'] = {
#         'preConds': {'pop':'TC'}, 
#         'postConds': {'pop':'I4'},
#         'probability': 0.25, 
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['HTC -> I4'] = {
#         'preConds': {'pop':'HTC'}, 
#         'postConds': {'pop':'I4'},
#         'probability': 0.25,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TC -> I5'] = {
#         'preConds': {'pop':'TC'}, 
#         'postConds': {'pop':'I5'},
#         'probability': , #0.1*thalfctr
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['HTC -> I5'] = {
#         'preConds': {'pop':'HTC'}, 
#         'postConds': {'pop':'I5'},
#         'probability': , #0.1*thalfctr
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TC -> I6'] = {
#         'preConds': {'pop':'TC'}, 
#         'postConds': {'pop':'I6'},
#         'probability': , #0.15*thalfctr
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['HTC -> I6'] = {
#         'preConds': {'pop':'HTC'}, 
#         'postConds': {'pop':'I6'},
#         'probability': , #0.15*thalfctr 
#         'weight': ,
#         'delay':,
#         'synMech': } 


# #### Matrix Thalamocortical
# netParams.connParams['TCM -> E2'] = {
#         'preConds': {'pop':'TCM'}, 
#         'postConds': {'pop':'E2'},
#         'probability': 0.25,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TCM -> E5B'] = {
#         'preConds': {'pop':'TCM'}, 
#         'postConds': {'pop':'E5B'},
#         'probability': , #0.15*thalfctr
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TCM -> E5R'] = {
#         'preConds': {'pop':'TCM'}, 
#         'postConds': {'pop':'E5R'},
#         'probability': , #0.15*thalfctr
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TCM -> E6'] = {
#         'preConds': {'pop':'TCM'}, 
#         'postConds': {'pop':'E6'},
#         'probability': , #0.05*thalfctr
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TCM -> I2'] = {
#         'preConds': {'pop':'TCM'}, 
#         'postConds': {'pop':'I2'},
#         'probability': 0.25,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TCM -> I5'] = {
#         'preConds': {'pop':'TCM'}, 
#         'postConds': {'pop':'I5'},
#         'probability': , #0.15*thalfctr
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TCM -> I6'] = {
#         'preConds': {'pop':'TCM'}, 
#         'postConds': {'pop':'I6'},
#         'probability': , #0,05*thalfctr
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['TCM -> I2L'] = {
#         'preConds': {'pop':'TCM'}, 
#         'postConds': {'pop':'I2L'},
#         'probability': 0.25, # new - test (from mpisim.py)
#         'weight': ,
#         'delay':,
#         'synMech': } 



# #### Corticothalamic
# netParams.connParams['E6 -> TC'] = {
#         'preConds': {'pop':'E6'}, 
#         'postConds': {'pop':'TC'},
#         'probability': 0.1,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['E6 -> HTC'] = {
#         'preConds': {'pop':'E6'}, 
#         'postConds': {'pop':'HTC'},
#         'probability': 0.1,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['E6 -> IRE'] = {
#         'preConds': {'pop':'E6'}, 
#         'postConds': {'pop':'IRE'},
#         'probability': 0.1,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['E5R -> TCM'] = {
#         'preConds': {'pop':'E5R'}, 
#         'postConds': {'pop':'TCM'},
#         'probability': 0.1,
#         'weight': ,
#         'delay':,
#         'synMech': } 
# netParams.connParams['E5B -> TCM'] = {
#         'preConds': {'pop':'E5B'}, 
#         'postConds': {'pop':'TCM'},
#         'probability': 0.1,
#         'weight': ,
#         'delay':,
#         'synMech': } 




# SIMULATION PARAMETERS
simConfig = specs.SimConfig()    # object of class SimConfig to store simulation configuration

simConfig.duration =                ## Duration of the sim, in ms
simConfig.dt =                      ## Internal Integration Time Step 
simConfig.verbose = False           ## Show detailed messages
simConfig.recordTraces = {}         ## Dict with traces to record
simConfig.recordStep =              ## Step size (in ms) to save data 
simConfig.filename =                ## Set file output name
simConfig.savePickle = True         ## Save pkl file
simConfig.saveJson = True           ## Save json file

simConfig.analysis['plotRaster'] = True     ## Plot a raster
simConfig.analysis['plot2Dnet'] = True      ## Plot 2D visualization of cell positions & connections 

# CREATE NETWORK AND RUN SIM
sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)


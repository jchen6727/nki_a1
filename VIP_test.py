#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:23:50 2019

@author: ricky
"""

# Imports
from netpyne import specs, sim

# Network Parameters
netParams = specs.NetParams()
simConfig = specs.SimConfig()

# Population Parameters
netParams.popParams['VIP_orig'] = {'cellType': 'VIP', 'numCells': 1, 'cellModel': 'HH_CA1'} # what determines cellModel? 
netParams.popParams['VIP_edited'] = {'cellType': 'VIP', 'numCells': 1, 'cellModel': 'HH_CA1'} # what determines cellModel? 

# Define lambda_f?
#lambda_f = 100

# Import .hoc
netParams.importCellParams(label='VIP_orig_rule', conds={'cellType': 'VIP', 'cellModel': 'HH_CA1'}, fileName='cells/vipcr_cell_orig.hoc', cellName='VIPCRCell', importSynMechs=True)
netParams.importCellParams(label='VIP_edited_rule', conds={'cellType': 'VIP', 'cellModel': 'HH_CA1'}, fileName='cells/vipcr_cell.hoc', cellName='VIPCRCell_EDITED', importSynMechs=True)


# Add a stimulation
netParams.stimSourceParams['Input1'] = {'type': 'IClamp', 'del': 50, 'dur': 500, 'amp': 0.4}
netParams.stimSourceParams['Input2'] = {'type': 'IClamp', 'del': 50, 'dur': 500, 'amp': 0.4}

netParams.stimTargetParams['Input1--> VIP_orig'] = {'source': 'Input1', 'sec': 'soma', 'loc': 0.5, 'conds': {'pop': 'VIP_orig'}}#, 'cellList': range(1)}}
netParams.stimTargetParams['Input2--> VIP_edited'] = {'source': 'Input2', 'sec': 'soma', 'loc': 0.5, 'conds': {'pop': 'VIP_edited'}}#, 'cellList': range(1)}}

# Sim options
allpops = ['VIP_orig', 'VIP_edited']
simConfig.duration = 1*1e3
simConfig.dt = 0.05
simConfig.recordTraces = {'V_soma':{'sec':'soma', 'loc': 0.5, 'var': 'v'}}
simConfig.recordStep = 0.1

#simConfig.analysis['plotRaster'] = 1
simConfig.analysis['plotTraces'] = {'include': ['allCells', 'eachPop'], 'overlay': True} #'oneFigPer': 'trace'}

sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

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
netParams.popParams['VIP_pop'] = {'cellType': 'VIP', 'numCells': 1, 'cellModel': 'HH_CA1'} # what determines cellModel? 

# Define lambda_f?
#lambda_f = 100

# Import .hoc
netParams.importCellParams(label='VIP_rule', conds={'cellType': 'VIP', 'cellModel': 'HH_CA1'}, fileName='cells/vipcr_cell.hoc', cellName='VIPCRCell', importSynMechs=True)


# Add a stimulation
netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 50, 'dur': 200, 'amp': 0.4}

netParams.stimTargetParams['Input--> VIP'] = {'source': 'Input', 'sec': 'soma', 'loc': 0.5, 'conds': {'pop': 'VIP_pop'}}#, 'cellList': range(1)}}

# Sim options
simConfig.duration = 1*1e3
simConfig.dt = 0.05
simConfig.recordTraces = {'V_soma':{'sec':'soma', 'loc': 0.5, 'var': 'v'}}
simConfig.recordStep = 0.1

simConfig.analysis['plotRaster'] = 1
simConfig.analysis['plotTraces'] = {'include': [0]}

sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

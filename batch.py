"""
batch.py 

Batch simulation for M1 model using NetPyNE

Contributors: salvadordura@gmail.com, samuel.neymotin@nki.rfmh.org
"""
from netpyne.batch import Batch
from netpyne import specs
import numpy as np
import pickle
from signals import (lfp2csd, raw2avgerp)
from scipy.stats import pearsonr

search_params = {
    # these params control IC -> Thal [LB, UB]
    'ICThalweightECore': [0.75, 1.25],
    'ICThalweightICore': [0.1875, 0.3125],
    'ICThalprobECore': [0.1425, 0.2375],
    'ICThalprobICore': [0.09, 0.15],
    'ICThalMatrixCoreFactor': [0.075, 0.125],
    # these params added from Christoph Metzner branch
    'thalL4PV': [0.1875, 0.3125],
    'thalL4SOM': [0.1875, 0.3125],
    'thalL4E': [1.5, 2.5]
}

initial_params = {
    'duration': 5000,
    'printPopAvgRates': [300, 4000],
    'scaleDensity': 1.0,
    'recordStep': 0.05,
    # SET SEEDS FOR CONN AND STIM
    'seeds': {'conn': 0, 'stim': 0, 'loc': 0},
}

# experimental data from nonhuman primates
with open('NHP_20kHz.pkl', 'rb') as fptr:
    nhp_data = pickle.load(fptr)

nhp_erp = nhp_data['ttavg']
nhp_csd = nhp_data['avgCSD']
fs = nhp_data['sampr']

def fitness(simData, **kwargs):
    sim_lfp = np.array(simData['LFP'])
    sim_csd = lfp2csd(sim_lfp, fs)
    sim_channels = [4, 10, 15]
    nhp_channels = [11-1, 15-1, 20-1] # s2, g, i1 channels for primary CSD sinks are at indices 10, 14, 19
    event_times = np.arange(3000, 4000, 300)
    response_duration = 150
    sim_erp = raw2avgerp(sim_csd, fs, event_times, response_duration)
    loss = -pearsonr( sim_erp.avg[sim_channels[1]], nhp_erp[nhp_channels[1]] )[0]
    return loss

search_config = {
    'fitnessFunc': fitness,  # fitness expression (should read simData)
    'fitnessFuncArgs': {}, # can be passed to **kwargs: see fitness
    'maxFitness': 1.0, # pearson regression
    'maxiters': 100,  # Maximum number of iterations (1 iteration = 1 function evaluation)
    'maxtime': None,  # Maximum time allowed, in seconds
    'maxiter_wait': 500,
    'time_sleep': 120,
}

b = Batch(params=search_params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initial_params)
b.method = 'optuna'

b.batchLabel = 'optunaERP'
b.saveFolder = 'data/'+b.batchLabel
b.run()

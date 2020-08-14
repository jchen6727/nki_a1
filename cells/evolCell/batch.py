import numpy as np
from netpyne import specs
from netpyne.batch import Batch

''' Example of evolutionary algorithm optimization of a cell using NetPyNE

To run use: mpiexec -np [num_cores] nrniv -mpi batch.py
'''

# ---------------------------------------------------------------------------------------------- #
def evolCellITS4():
    # parameters space to explore
    params = specs.ODict()

    params[('tune', 'soma', 'Ra')] = [100.*0.5, 100*1.5] 
    params[('tune', 'soma', 'cm')] = [0.75*0.5, 0.75*1.5]
    params[('tune', 'soma', 'kv', 'gbar')] = [1700.0*0.5, 1700.0*1.5]
    params[('tune', 'soma', 'naz', 'gmax')] = [72000.0*0.5, 72000.0*1.5]
    params[('tune', 'soma', 'pas', 'e')] = [-70*1.5, -70.0*0.5]
    params[('tune', 'soma', 'pas', 'g')] = [3.3333333333333335e-05*0.5, 3.3333333333333335e-05*1.5]

    params[('tune', 'dend', 'Ra')] = [0.02974858749381221*0.5, 0.02974858749381221*1.5] 
    params[('tune', 'dend', 'cm')] = [0.75*0.5, 0.75*1.5]
    params[('tune', 'dend', 'Nca', 'gmax')] = [0.3*0.5, 0.3*1.5]
    params[('tune', 'dend', 'kca', 'gbar')] = [3.0 * 0.5, 3.0 * 1.5]
    params[('tune', 'dend', 'km', 'gbar')] = [0.1*0.5, 0.1*1.5]
    params[('tune', 'dend', 'naz', 'gmax')] = [15.0*0.5, 15.0*1.5]
    params[('tune', 'dend', 'pas', 'e')] = [-70*1.5, -70.0*0.5]
    params[('tune', 'dend', 'pas', 'g')] = [3.3333333333333335e-05*0.5, 3.3333333333333335e-05*1.5]

    params[('tune', 'dend1', 'Ra')] = [0.015915494309189534*0.5, 0.015915494309189534*1.5] 
    params[('tune', 'dend1', 'cm')] = [0.75*0.5, 0.75*1.5]
    params[('tune', 'dend1', 'Nca', 'gmax')] = [0.3*0.5, 0.3*1.5]
    params[('tune', 'dend1', 'kca', 'gbar')] = [3.0*0.5, 3.0*1.5]
    params[('tune', 'dend1', 'km', 'gbar')] = [0.1*0.5, 0.1*1.5]
    params[('tune', 'dend1', 'naz', 'gmax')] = [15.0*0.5, 15.0*1.5]
    params[('tune', 'dend1', 'pas', 'e')] = [-70*1.5, -70.0*0.5]
    params[('tune', 'dend1', 'pas', 'g')] = [3.3333333333333335e-05*0.5, 3.3333333333333335e-05*1.5]


    # current injection params
    amps = list(np.arange(0.0, 0.65, 0.05))  # amplitudes
    times = list(np.arange(1000, 2000 * len(amps), 2000))  # start times
    dur = 500  # ms
    targetRates = [0., 0., 19., 29., 37., 45., 51., 57., 63., 68., 73., 77., 81.]
 
    # initial cfg set up
    initCfg = {} # specs.ODict()
    initCfg['duration'] = 2000 * len(amps)
    initCfg[('hParams', 'celsius')] = 37

    initCfg['savePickle'] = True
    initCfg['saveJson'] = False
    initCfg['saveDataInclude'] = ['simConfig', 'netParams', 'net', 'simData']

    initCfg[('IClamp1', 'pop')] = 'ITS4'
    initCfg[('IClamp1', 'amp')] = amps
    initCfg[('IClamp1', 'start')] = times
    initCfg[('IClamp1', 'dur')] = 1000

    initCfg[('analysis', 'plotfI', 'amps')] = amps
    initCfg[('analysis', 'plotfI', 'times')] = times
    initCfg[('analysis', 'plotfI', 'dur')] = dur
    initCfg[('analysis', 'plotfI', 'targetRates')] = targetRates
    
    for k, v in params.items():
        initCfg[k] = v[0]  # initialize params in cfg so they can be modified    

    # fitness function
    fitnessFuncArgs = {}
    fitnessFuncArgs['targetRates'] = targetRates
    
    def fitnessFunc(simData, **kwargs):
        targetRates = kwargs['targetRates']
            
        diffRates = [abs(x-t) for x,t in zip(simData['fI'], targetRates)]
        fitness = np.mean(diffRates)
        
        print(' Candidate rates: ', simData['fI'])
        print(' Target rates:    ', targetRates)
        print(' Difference:      ', diffRates)

        return fitness
        

    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, initCfg=initCfg)
    
    # Set output folder, grid method (all param combinations), and run configuration
    b.batchLabel = 'ITS4_evol'
    b.saveFolder = 'data/'+b.batchLabel
    b.method = 'evol'
    b.runCfg = {
        'type': 'mpi_bulletin', #'hpc_slurm', 
        'script': 'init.py',
        # # options required only for hpc
        # 'mpiCommand': 'mpirun',  
        # 'nodes': 1,
        # 'coresPerNode': 2,
        # 'allocation': 'default',
        # 'email': 'salvadordura@gmail.com',
        # 'reservation': None,
        # 'folder': '/home/salvadord/evol'
        # #'custom': 'export LD_LIBRARY_PATH="$HOME/.openmpi/lib"' # only for conda users
    }
    b.evolCfg = {
        'evolAlgorithm': 'custom',
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'pop_size': 40,
        'num_elites': 1, # keep this number of parents for next generation if they are fitter than children
        'mutation_rate': 0.4,
        'crossover': 0.5,
        'maximize': False, # maximize fitness function?
        'max_generations': 20,
        'time_sleep': 5, # wait this time before checking again if sim is completed (for each generation)
        'maxiter_wait': 20, # max number of times to check if sim is completed (for each generation)
        'defaultFitness': 1000 # set fitness value in case simulation time is over
    }
    # Run batch simulations
    b.run()


# ---------------------------------------------------------------------------------------------- #
def evolCellNGF():
    # parameters space to explore
    params = specs.ODict()

    scalingRange = [0.5, 1.5]
    
    params[('tune', 'soma', 'Ra')] = scalingRange
    params[('tune', 'soma', 'cm')] = scalingRange
    params[('tune', 'soma', 'ch_CavL', 'gmax')] = scalingRange
    params[('tune', 'soma', 'ch_CavN', 'gmax')] = scalingRange
    params[('tune', 'soma', 'ch_KCaS', 'gmax')] = scalingRange
    params[('tune', 'soma', 'ch_Kdrfastngf', 'gmax')] = scalingRange
    params[('tune', 'soma', 'ch_KvAngf', 'gmax')] = scalingRange
    params[('tune', 'soma', 'ch_KvAngf', 'gml')] = scalingRange
    params[('tune', 'soma', 'ch_KvAngf', 'gmn')] = scalingRange
    params[('tune', 'soma', 'ch_KvCaB', 'gmax')] = scalingRange
    params[('tune', 'soma', 'ch_Navngf', 'gmax')] = scalingRange
    params[('tune', 'soma', 'hd', 'ehd')] = scalingRange
    params[('tune', 'soma', 'hd', 'elk')] = scalingRange
    params[('tune', 'soma', 'hd', 'gbar')] = scalingRange
    params[('tune', 'soma', 'hd', 'vhalfl')] = scalingRange
    # params[('tune', 'soma', 'iconc_Ca', 'caiinf')] = scalingRange
    # params[('tune', 'soma', 'iconc_Ca', 'catau')] = scalingRange
    params[('tune', 'soma', 'pas', 'e')] = scalingRange
    params[('tune', 'soma', 'pas', 'g')] = scalingRange

    params[('tune', 'dend', 'Ra')] = scalingRange
    params[('tune', 'dend', 'cm')] = scalingRange
    params[('tune', 'dend', 'ch_Kdrfastngf', 'gmax')] = scalingRange
    params[('tune', 'dend', 'ch_Navngf', 'gmax')] = scalingRange
    params[('tune', 'dend', 'pas', 'e')] = scalingRange
    params[('tune', 'dend', 'pas', 'g')] = scalingRange


    # current injection params
    interval = 1000  # 10000
    dur = 500  # ms
    durSteady = 200  # ms
    amps = list(np.arange(0.04+0.075, 0.121+0.075, 0.01))  # amplitudes
    times = list(np.arange(1000, (dur+interval) * len(amps), dur+interval))  # start times
    targetRatesOnset = [43., 52., 68., 80., 96., 110., 119., 131., 139.]
    targetRatesSteady = [22., 24., 27., 30., 33., 35., 37., 39., 41.]

    stimWeights = [10, 50, 100, 150]
    stimRate = 80
    stimDur = 2000
    stimTimes = [times[-1] + x for x in list(np.arange(interval, (stimDur + interval) * len(stimWeights), stimDur + interval))]
    stimTargetSensitivity = 100  # max - min 

    # initial cfg set up
    initCfg = {} # specs.ODict()
    initCfg['duration'] = ((dur+interval) * len(amps)) + ((stimDur+interval) * len(stimWeights)) 
    initCfg[('hParams', 'celsius')] = 37

    initCfg['savePickle'] = True
    initCfg['saveJson'] = False
    initCfg['saveDataInclude'] = ['simConfig', 'netParams', 'net', 'simData']

    initCfg[('IClamp1', 'pop')] = 'NGF'
    initCfg[('IClamp1', 'amp')] = amps
    initCfg[('IClamp1', 'start')] = times
    initCfg[('IClamp1', 'dur')] = dur
    
    # iclamp
    initCfg[('analysis', 'plotTraces', 'timeRange')] = [0, initCfg['duration']] 
    initCfg[('analysis', 'plotfI', 'amps')] = amps
    initCfg[('analysis', 'plotfI', 'times')] = times
    initCfg[('analysis', 'plotfI', 'calculateOnset')] = True
    initCfg[('analysis', 'plotfI', 'dur')] = dur
    initCfg[('analysis', 'plotfI', 'durSteady')] = durSteady
    initCfg[('analysis', 'plotfI', 'targetRates')] = [] #
    initCfg[('analysis', 'plotfI', 'targetRatesOnset')] = targetRatesOnset
    initCfg[('analysis', 'plotfI', 'targetRatesSteady')] = targetRatesSteady

    # netstim 
    initCfg[('NetStim1', 'weight')] = stimWeights
    initCfg[('NetStim1', 'start')] = stimTimes
    initCfg[('NetStim1', 'interval')] = 1000.0 / stimRate 
    initCfg[('NetStim1', 'pop')] = 'NGF'
    initCfg[('NetStim1', 'sec')] = 'soma'
    initCfg[('NetStim1', 'synMech')] = ['AMPA', 'NMDA']
    initCfg[('NetStim1', 'synMechWeightFactor')] = [0.5, 0.5]
    initCfg[('NetStim1', 'number')] = 1e9
    initCfg[('NetStim1', 'noise')] = 1.0


    initCfg['removeWeightNorm'] = False
    initCfg[('analysis', 'plotRaster')] = False
    initCfg['printPopAvgRates'] = [[x, x+stimDur] for x in stimTimes]
    
    
    for k, v in params.items():
        initCfg[k] = v[0]  # initialize params in cfg so they can be modified    

    # fitness function
    fitnessFuncArgs = {}
    fitnessFuncArgs['targetRatesOnset'] = targetRatesOnset
    fitnessFuncArgs['targetRatesSteady'] = targetRatesSteady
    fitnessFuncArgs['stimTargetSensitivity'] = stimTargetSensitivity
    
    def fitnessFunc(simData, **kwargs):
        targetRatesOnset = kwargs['targetRatesOnset']
        targetRatesSteady = kwargs['targetRatesSteady']
        stimTargetSensitivity = kwargs['stimTargetSensitivity']
            
        diffRatesOnset = [abs(x-t) for x,t in zip(simData['fI_onset'], targetRatesOnset)]
        diffRatesSteady = [abs(x - t) for x, t in zip(simData['fI_steady'], targetRatesSteady)]
        stimDiffRate = np.max(simData['popRates']['NGF']) - np.min(simData['popRates']['NGF'])
        
        maxFitness = 1000
        fitness = np.mean(diffRatesOnset + diffRatesSteady) if stimDiffRate < stimTargetSensitivity else maxFitness
        
        print(' Candidate rates: ', simData['fI_onset']+simData['fI_steady'])
        print(' Target rates:    ', targetRatesOnset+targetRatesSteady)
        print(' Difference:      ', diffRatesOnset+diffRatesSteady)

        return fitness
        

    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, initCfg=initCfg)
    
    # Set output folder, grid method (all param combinations), and run configuration
    b.batchLabel = 'NGF_evol'
    b.saveFolder = 'data/'+b.batchLabel
    b.method = 'evol'
    b.runCfg = {
        'type': 'mpi_bulletin',#'hpc_slurm', 
        'script': 'init.py',
        # # options required only for hpc
        # 'mpiCommand': 'mpirun',  
        # 'nodes': 1,
        # 'coresPerNode': 2,
        # 'allocation': 'default',
        # 'email': 'salvadordura@gmail.com',
        # 'reservation': None,
        # 'folder': '/home/salvadord/evol'
        # #'custom': 'export LD_LIBRARY_PATH="$HOME/.openmpi/lib"' # only for conda users
    }
    b.evolCfg = {
        'evolAlgorithm': 'custom',
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'pop_size': 1,
        'num_elites': 1, # keep this number of parents for next generation if they are fitter than children
        'mutation_rate': 0.4,
        'crossover': 0.5,
        'maximize': False, # maximize fitness function?
        'max_generations': 1,
        'time_sleep': 10, # wait this time before checking again if sim is completed (for each generation)
        'maxiter_wait': 20, # max number of times to check if sim is completed (for each generation)
        'defaultFitness': 1000 # set fitness value in case simulation time is over
    }
    # Run batch simulations
    b.run()


# ---------------------------------------------------------------------------------------------- #
# Main code
if __name__ == '__main__':
    # evolCellITS4()
    evolCellNGF() 


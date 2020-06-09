"""
batch.py 

Batch simulation for M1 model using NetPyNE

Contributors: salvadordura@gmail.com
"""
from netpyne.batch import Batch
from netpyne import specs
import numpy as np


# ----------------------------------------------------------------------------------------------
# Weight Normalization 
# ----------------------------------------------------------------------------------------------
def bkgWeights(pops=[], weights=list(range(50))):

    params = specs.ODict()
    params['singlePop'] = pops
    params['weightBkg'] = weights

    # set initial config
    initCfg = {}
    # sim and recoring params
    initCfg['duration'] = 10.0 * 1e3
    initCfg['singleCellPops'] = True
    initCfg['singlePopForNetstim'] = True
    initCfg['removeWeightNorm'] = False
    initCfg[('analysis','plotTraces','include')] = [0]
    initCfg[('analysis','plotTraces','timeRange')] = [0, 3000]
    initCfg[('analysis', 'plotRaster')] = False

    initCfg[('rateBkg', 'exc')] = 40
    initCfg[('rateBkg', 'inh')] = 40

    ## turn off components not required
    initCfg['addBkgConn'] = True
    initCfg['addConn'] = False
    initCfg['addIntraThalamicConn'] = False
    initCfg['addIntraThalamicConn'] = False
    initCfg['addCorticoThalamicConn'] = False
    initCfg['addCoreThalamoCorticalConn'] = False
    initCfg['addMatrixThalamoCorticalConn'] = False
    initCfg['stimSubConn'] = False
    initCfg['addIClamp'] = False
    initCfg['addNetStim'] = False
 
    b = Batch(params=params, netParamsFile='netParams_bkg.py', cfgFile='cfg_cell.py', initCfg=initCfg)
    b.method = 'grid'

    return b

# ----------------------------------------------------------------------------------------------
# Weight Normalization 
# ----------------------------------------------------------------------------------------------
def bkgWeights2D(pops=[], weights=list(range(50))):

    params = specs.ODict()
    params['singlePop'] = pops
    params['weightBkgE'] = weights
    params['weightBkgI'] = weights
    params[('rateBkg', 'exc')] = [20, 40, 60, 80]
    params[('rateBkg', 'inh')] = [20, 40, 60, 80]

    # set initial config
    initCfg = {}
    # sim and recoring params
    initCfg['duration'] = 3.0 * 1e3
    initCfg['singleCellPops'] = True
    initCfg['singlePopForNetstim'] = True
    initCfg['removeWeightNorm'] = False
    initCfg[('analysis','plotTraces','include')] = [0]
    initCfg[('analysis','plotTraces','timeRange')] = [0, 3000]
    initCfg[('analysis', 'plotRaster')] = False
    initCfg['printPopAvgRates'] = [500, 3000]

    initCfg[('rateBkg', 'exc')] = 40
    initCfg[('rateBkg', 'inh')] = 40

    ## turn off components not required
    initCfg['addBkgConn'] = True
    initCfg['addConn'] = False
    initCfg['addIntraThalamicConn'] = False
    initCfg['addIntraThalamicConn'] = False
    initCfg['addCorticoThalamicConn'] = False
    initCfg['addCoreThalamoCorticalConn'] = False
    initCfg['addMatrixThalamoCorticalConn'] = False
    initCfg['stimSubConn'] = False
    initCfg['addIClamp'] = False
    initCfg['addNetStim'] = False
 

    b = Batch(params=params, netParamsFile='netParams_bkg.py', cfgFile='cfg_cell.py', initCfg=initCfg)
    b.method = 'grid'

    return b

# ----------------------------------------------------------------------------------------------
# Weight Normalization 
# ----------------------------------------------------------------------------------------------
def weightNorm(pops=[], rule = None, segs = None, allSegs = True, weights=list(np.arange(0.01, 0.2, 0.01)/100.0)):

    # Add params
    from cfg_cell import cfg
    from netParams_cell import netParams

    excludeSegs = ['axon']
    if not segs:
        secs = []
        locs = []
        for secName,sec in netParams.cellParams[rule]['secs'].items():
            if secName not in excludeSegs:
                if allSegs:
                    nseg = sec['geom']['nseg']
                    for iseg in range(nseg):
                        secs.append(secName) 
                        locs.append((iseg+1)*(1.0/(nseg+1)))
                else:
                    secs.append(secName) 
                    locs.append(0.5)

    params = specs.ODict()
    params[('NetStim1', 'pop')] = pops
    params[('NetStim1', 'sec')] = secs
    params[('NetStim1', 'loc')] = locs
    params[('NetStim1', 'weight')] = weights

    groupedParams = [('NetStim1', 'sec'), ('NetStim1', 'loc')] 

    # set initial config
    initCfg = {}
    # sim and recoring params
    initCfg['duration'] = 1.0 * 1e3
    initCfg['singleCellPops'] = True
    initCfg['removeWeightNorm'] = True
    initCfg[('analysis','plotTraces','include')] = []
    initCfg[('analysis','plotTraces','timeRange')] = [0, 1000]
    
    ## turn off components not required
    #initCfg[('analysis', 'plotRaster')] = False
    initCfg['addConn'] = False
    initCfg['addIntraThalamicConn'] = False
    initCfg['addIntraThalamicConn'] = False
    initCfg['addCorticoThalamicConn'] = False
    initCfg['addCoreThalamoCorticalConn'] = False
    initCfg['addMatrixThalamoCorticalConn'] = False
    initCfg['addBkgConn'] = False
    initCfg['stimSubConn'] = False
    initCfg['addIClamp'] = 0
 
    ## set netstim params
    initCfg['addNetStim'] = True
    initCfg[('NetStim1', 'synMech')] = ['AMPA','NMDA']
    initCfg[('NetStim1','synMechWeightFactor')] = [0.5,0.5]
    initCfg[('NetStim1', 'start')] = 700
    initCfg[('NetStim1', 'interval')] = 1000
    initCfg[('NetStim1','ynorm')] = [0.0, 2.0]
    initCfg[('NetStim1', 'noise')] = 0
    initCfg[('NetStim1', 'number')] = 1
    initCfg[('NetStim1', 'delay')] = 1
    
    
    b = Batch(params=params, netParamsFile='netParams_cell.py', cfgFile='cfg_cell.py', initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'

    return b

# ----------------------------------------------------------------------------------------------
# Exc-Inh balance
# ----------------------------------------------------------------------------------------------
def EIbalance():
    params = specs.ODict()

    params['EEGain'] = [0.5, 1.0, 1.5] 
    params['EIGain'] = [0.5, 1.0, 1.5] 
    params['IEGain'] = [0.5, 1.0, 1.5] 
    params['IIGain'] = [0.5, 1.0, 1.5]
    params[('weightBkg', 'E')] = [2.0, 3.0]
    params[('weightBkg', 'I')] = [2.0, 3.0]
    
    groupedParams =  []

    # initial config
    initCfg = {}
    initCfg['duration'] = 1.0 * 1e3
    initCfg['scaleDensity'] = 0.05
    
    b = Batch(params=params, groupedParams=groupedParams, initCfg=initCfg)

    return b


# ----------------------------------------------------------------------------------------------
# Exc-Inh balance
# ----------------------------------------------------------------------------------------------
def longBalance():
    params = specs.ODict()

    params[('ratesLong', 'TPO', 1)] = [2,4]
    params[('ratesLong', 'TVL', 1)] = [2,4]
    params[('ratesLong', 'S1', 1)] = [2,4]
    params[('ratesLong', 'S2', 1)] = [2,4]
    params[('ratesLong', 'cM1', 1)] = [2,4]
    params[('ratesLong', 'M2', 1)] = [2,4]
    params[('ratesLong', 'OC', 1)] = [2,4]

    # 
    params['IEweights'] = [[0.8,0.8,0.8], [1.0,1.0,1.0], [1.2,1.2,1.2]]
    params['IIweights'] =  [[0.8,0.8,0.80], [1.0, 1.0, 1.0], [1.2,1.2,1.2]]

    params['ihGbar'] = [0.25, 1.0]

    groupedParams = []

    # initial config
    initCfg = {}
    initCfg['duration'] = 2.0*1e3
    initCfg['ihModel'] = 'migliore'  # ih model

    initCfg['ihGbarBasal'] = 1.0 # multiplicative factor for ih gbar in PT cells
    initCfg['ihlkc'] = 0.2 # ih leak param (used in Migliore)
    initCfg['ihLkcBasal'] = 1.0 # multiplicative factor for ih lk in PT cells
    initCfg['ihLkcBelowSoma'] = 0.01 # multiplicative factor for ih lk in PT cells
    initCfg['ihlke'] = -86  # ih leak param (used in Migliore)
    initCfg['ihSlope'] = 28  # ih leak param (used in Migliore)

    initCfg['somaNa'] = 5.0  # somatic Na conduct
    initCfg['dendNa'] = 0.3  # dendritic Na conduct (reduced to avoid dend spikes) 
    initCfg['axonNa'] = 7   # axon Na conduct (increased to compensate) 
    initCfg['axonRa'] = 0.005
    initCfg['gpas'] = 0.5
    initCfg['epas'] = 0.9

    initCfg[('pulse', 'pop')] = 'S2'
    initCfg[('pulse', 'rate')] = 10.0
    initCfg[('pulse', 'start')] = 1000.0
    initCfg[('pulse', 'end')] = 1100.0
    initCfg[('pulse', 'noise')] = 0.8

    initCfg['IEdisynapticBias'] = None

    initCfg['weightNormThreshold'] = 4.0
    initCfg['IEGain'] = 1.0
    initCfg['IIGain'] = 1.0
    initCfg['IPTGain'] = 1.0

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False
    
    b = Batch(params=params, groupedParams=groupedParams, initCfg=initCfg)
    b.method = 'grid'

    return b

# ----------------------------------------------------------------------------------------------
# Long-range pop stimulation
# ----------------------------------------------------------------------------------------------
def longPopStims():
    params = specs.ODict()
    
    params['ihGbar'] = [0.25, 1.0] # [0.2, 0.25, 0.3, 1.0]
    params[('seeds', 'conn')] = [4321+(17*i) for i in range(5)]
    params[('seeds', 'stim')] = [1234+(17*i) for i in range(5)]

    params[('pulse', 'pop')] = ['None'] #, 'TPO', 'TVL', 'S2', 'M2'] #, 'OC'] # 'S1','cM1',
    #params[('pulse', 'end')] = [1100, 1500]

    groupedParams = []

    # initial config
    initCfg = {}
    initCfg['duration'] = 51*1e3 #2.5*1e3
    initCfg['ihModel'] = 'migliore'  # ih model

    initCfg['ihGbarBasal'] = 1.0 # multiplicative factor for ih gbar in PT cells
    initCfg['ihlkc'] = 0.2 # ih leak param (used in Migliore)
    initCfg['ihLkcBasal'] = 1.0 # multiplicative factor for ih lk in PT cells
    initCfg['ihLkcBelowSoma'] = 0.01 # multiplicative factor for ih lk in PT cells
    initCfg['ihlke'] = -86  # ih leak param (used in Migliore)
    initCfg['ihSlope'] = 28  # ih leak param (used in Migliore)

    initCfg['somaNa'] = 5.0  # somatic Na conduct
    initCfg['dendNa'] = 0.3  # dendritic Na conduct (reduced to avoid dend spikes) 
    initCfg['axonNa'] = 7   # axon Na conduct (increased to compensate) 
    initCfg['axonRa'] = 0.005
    initCfg['gpas'] = 0.5
    initCfg['epas'] = 0.9

    #initCfg[('pulse', 'pop')] = 'None'
    initCfg[('pulse', 'rate')] = 10.0
    initCfg[('pulse', 'start')] = 1000.0
    initCfg[('pulse', 'end')] = 1100.0
    initCfg[('pulse', 'noise')] = 0.8

    initCfg['IEdisynapticBias'] = None

    initCfg['weightNormThreshold'] = 4.0
    initCfg['EEGain'] = 0.5 
    initCfg['IEGain'] = 1.0
    initCfg['IIGain'] = 1.0
    initCfg['IPTGain'] = 1.0

    initCfg[('ratesLong', 'TPO', 1)] = 5 	
    initCfg[('ratesLong', 'TVL', 1)] = 2.5
    initCfg[('ratesLong', 'S1', 1)] = 5
    initCfg[('ratesLong', 'S2', 1)] = 5 
    initCfg[('ratesLong', 'cM1', 1)] = 2.5
    initCfg[('ratesLong', 'M2', 1)] = 2.5
    initCfg[('ratesLong', 'OC', 1)] = 5	

    # # L2/3+4
    initCfg[('IEweights',0)] =  0.8
    initCfg[('IIweights',0)] =  1.2 
    # L5
    initCfg[('IEweights',1)] = 0.8   
    initCfg[('IIweights',1)] = 1.0
    # L6
    initCfg[('IEweights',2)] =  1.0  
    initCfg[('IIweights',2)] =  1.0

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False

    groupedParams = [] #('IEweights',0), ('IIweights',0), ('IEweights',1), ('IIweights',1), ('IEweights',2), ('IIweights',2)]

    b = Batch(params=params, initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'

    return b

# ----------------------------------------------------------------------------------------------
# Simultaenous long-range pop stimulations
# ----------------------------------------------------------------------------------------------
def simultLongPopStims():
    params = specs.ODict()
    
    params[('pulse', 'pop')] = ['TPO', 'M2', 'TVL', 'S2', 'S2', 'M2', 'TVL', 'TPO']
    params[('pulse2', 'pop')] = ['M2', 'TPO', 'S2', 'TVL', 'M2', 'S2', 'TPO', 'TVL']
    params[('pulse2', 'start')] = list(np.arange(1500, 2020, 20))
    params['ihGbar'] = [0.25, 1.0]


    # initial config
    initCfg = {}
    initCfg['duration'] = 3.0*1e3
    initCfg['ihModel'] = 'migliore'  # ih model

    initCfg['ihGbarBasal'] = 1.0 # multiplicative factor for ih gbar in PT cells
    initCfg['ihlkc'] = 0.2 # ih leak param (used in Migliore)
    initCfg['ihLkcBasal'] = 1.0 # multiplicative factor for ih lk in PT cells
    initCfg['ihLkcBelowSoma'] = 0.01 # multiplicative factor for ih lk in PT cells
    initCfg['ihlke'] = -86  # ih leak param (used in Migliore)
    initCfg['ihSlope'] = 28  # ih leak param (used in Migliore)

    initCfg['somaNa'] = 5.0  # somatic Na conduct
    initCfg['dendNa'] = 0.3  # dendritic Na conduct (reduced to avoid dend spikes) 
    initCfg['axonNa'] = 7   # axon Na conduct (increased to compensate) 
    initCfg['axonRa'] = 0.005
    initCfg['gpas'] = 0.5
    initCfg['epas'] = 0.9

    #initCfg[('pulse', 'pop')] = 'None'
    initCfg[('pulse', 'rate')] = 10.0
    initCfg[('pulse', 'start')] = 1500.0
    initCfg[('pulse', 'end')] = 1700.0
    initCfg[('pulse', 'noise')] = 0.8

    #initCfg[('pulse2', 'start')] = 1500.0
    initCfg[('pulse2', 'rate')] = 10.0
    initCfg[('pulse2', 'duration')] = 200.0
    initCfg[('pulse2', 'noise')] = 0.8


    initCfg['IEdisynapticBias'] = None

    initCfg['weightNormThreshold'] = 4.0
    initCfg['EEGain'] = 0.5 
    initCfg['IEGain'] = 1.0
    initCfg['IIGain'] = 1.0
    initCfg['IPTGain'] = 1.0

    initCfg[('ratesLong', 'TPO', 1)] = 5 	
    initCfg[('ratesLong', 'TVL', 1)] = 2.5
    initCfg[('ratesLong', 'S1', 1)] = 5
    initCfg[('ratesLong', 'S2', 1)] = 5 
    initCfg[('ratesLong', 'cM1', 1)] = 2.5
    initCfg[('ratesLong', 'M2', 1)] = 2.5
    initCfg[('ratesLong', 'OC', 1)] = 5	

    # # L2/3+4
    initCfg[('IEweights',0)] =  0.8
    initCfg[('IIweights',0)] =  1.2 
    # L5
    initCfg[('IEweights',1)] = 0.8   
    initCfg[('IIweights',1)] = 1.0
    # L6
    initCfg[('IEweights',2)] =  1.0  
    initCfg[('IIweights',2)] =  1.0

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False

    groupedParams = [('pulse', 'pop'),('pulse2', 'pop')] 
    b = Batch(params=params, initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'

    return b



# ----------------------------------------------------------------------------------------------
# Recorded stimulation
# ----------------------------------------------------------------------------------------------
def recordedLongPopStims():
    params = specs.ODict()
    
    high = 'cells/ssc-3_spikes.json'
    low  = 'cells/ssc-3_lowrate_spikes.json'
    low2 = 'cells/ssc-3_lowrate2_spikes.json'


    # 1) normal, 2) S2high+lowbkg, 3) S2low+bkg0.1, 4) S2low2+bkg0.1, 5) S2low2+M2low+bkg0.1, 6) S2low, 
    # 7) S2high, 8) S1high, 9) S1low, 10) M2low, 11) M2high
    params[('ratesLong','S2')] =  [[0,2]]#, high,    low,     low2,	 low2,		high, 	low,   [0,2], [0,2], [0,2], [0,2]]
    params[('ratesLong','S1')] =  [[0,2]]#, [0,0.1], [0,0.1], [0,0.1], [0,0.1],	[0,2], 	[0,2], high,  low,   [0,2], [0,2]]
    params[('ratesLong','M2')] =  [[0,2]]#, [0,0.1], [0,0.1], [0,0.1], low, 		[0,2], 	[0,2], [0,2], [0,2], high, 	low]
    params[('ratesLong','TPO')] = [[0,4]]#, [0,0.1], [0,0.1], [0,0.1], [0,0.1],	[0,4],	[0,4], [0,4], [0,4], [0,4], [0,4]]
    params[('ratesLong','TVL')] = [[0,4]]#, [0,0.1], [0,0.1], [0,0.1], [0,0.1],	[0,4],	[0,4], [0,4], [0,4], [0,4], [0,4]]
    params[('ratesLong','cM1')] = [[0,4]]#, [0,0.1], [0,0.1], [0,0.1], [0,0.1],	[0,4],	[0,4], [0,4], [0,4], [0,4], [0,4]]
    params[('ratesLong','OC')] =  [[0,2]]#, [0,0.1], [0,0.1], [0,0.1], [0,0.1],	[0,2], 	[0,2], [0,2], [0,2], [0,2], [0,2]]
    #params['ihGbar'] = [0.3, 0.4, 0.5, 1.0]
    params['ihGbar'] = [0.3] #, 1.0]
    
    # initial config
    initCfg = {}

    initCfg['duration'] = 6.0*1e3
    initCfg['ihModel'] = 'migliore'  # ih model

    initCfg['ihGbarBasal'] = 1.0 # multiplicative factor for ih gbar in PT cells
    initCfg['ihlkc'] = 0.2 # ih leak param (used in Migliore)
    initCfg['ihLkcBasal'] = 1.0 # multiplicative factor for ih lk in PT cells
    initCfg['ihLkcBelowSoma'] = 0.01 # multiplicative factor for ih lk in PT cells
    initCfg['ihlke'] = -86  # ih leak param (used in Migliore)
    initCfg['ihSlope'] = 28  # ih leak param (used in Migliore)

    initCfg['somaNa'] = 5.0
    initCfg['dendNa'] = 0.3  # dendritic Na conduct (reduced to avoid dend spikes) 
    initCfg['axonNa'] = 7   # axon Na conduct (increased to compensate) 
    initCfg['axonRa'] = 0.005
    initCfg['gpas'] = 0.5
    initCfg['epas'] = 0.9

    initCfg[('analysis','plotRaster','timeRange')] = [500, 5500]

    initCfg['weightNormThreshold'] = 4.0

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False

    initCfg['IEGain'] = 1.0
    initCfg['IIGain'] = 1.0
    initCfg['IEdisynapticBias'] = None

    # 1101 222

    # # L2/3+4
    initCfg[('IEweights',0)] = 1.2
    initCfg[('IIweights',0)] =  1.0  
    # L5
    initCfg[('IEweights',1)] = 1.2
    initCfg[('IIweights',1)] = 1.0
    # L6
    initCfg[('IEweights',2)] =  1.2  
    initCfg[('IIweights',2)] =  1.0

    # groupedParams = [('ratesLong','S2'), ('ratesLong','S1'), ('ratesLong','M2'), 
    # 				('ratesLong','TPO'), ('ratesLong','TVL'), ('ratesLong','cM1'), ('ratesLong','OC')]
    groupedParams = []

    b = Batch(params=params, initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'

    return b



# ----------------------------------------------------------------------------------------------
# Frequency stimulation
# ----------------------------------------------------------------------------------------------
def freqStims():
    params = specs.ODict()

    params[('NetStim1', 'interval')] = [1000.0/f for f in [4,8,12,16,20,24,28,32,36,40]]
    params[('NetStim1', 'number')] = [f for f in [4,8,12,16,20,24,28,32,36,40]]	
    params[('NetStim1', 'start')] = [500, 550]
    params['ihGbar'] = [0.5, 1.0]
    params[('NetStim1', 'ynorm', 1)] = [0.15+x*(0.31-0.12) for x in [0.1, 0.2, 0.3]]  # 10, 20, 30% of cells; L23 NCD = 0.12 - 0.31


    # initial config
    initCfg = {}
    initCfg['addNetStim'] = True
    initCfg[('NetStim1', 'pop')] = 'IT2'
    initCfg[('NetStim1', 'ynorm', 0)] = 0.15
    initCfg[('NetStim1', 'weight')] = 30.0	
    initCfg[('NetStim1', 'noise')] = 0.01	

    initCfg['duration'] = 2.0*1e3
    initCfg['ihModel'] = 'migliore'  # ih model

    initCfg['ihGbarBasal'] = 1.0 # multiplicative factor for ih gbar in PT cells
    initCfg['ihlkc'] = 0.2 # ih leak param (used in Migliore)
    initCfg['ihLkcBasal'] = 1.0 # multiplicative factor for ih lk in PT cells
    initCfg['ihLkcBelowSoma'] = 0.01 # multiplicative factor for ih lk in PT cells
    initCfg['ihlke'] = -86  # ih leak param (used in Migliore)
    initCfg['ihSlope'] = 28  # ih leak param (used in Migliore)

    initCfg['somaNa'] = 5.0
    initCfg['dendNa'] = 0.3  # dendritic Na conduct (reduced to avoid dend spikes) 
    initCfg['axonNa'] = 7   # axon Na conduct (increased to compensate) 
    initCfg['axonRa'] = 0.005
    initCfg['gpas'] = 0.5
    initCfg['epas'] = 0.9

    initCfg['weightNormThreshold'] = 4.0

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False

    initCfg['IEGain'] = 1.0
    initCfg['IIGain'] = 1.0
    initCfg['IEdisynapticBias'] = None


    # 1101 222
    initCfg[('ratesLong', 'TPO', 1)] = 4
    initCfg[('ratesLong', 'TVL', 1)] = 4
    initCfg[('ratesLong', 'S1', 1)] = 2
    initCfg[('ratesLong', 'cM1', 1)] = 4

    # # L2/3+4
    initCfg[('IEweights',0)] = 1.2
    initCfg[('IIweights',0)] =  1.0  
    # L5
    initCfg[('IEweights',1)] = 1.2
    initCfg[('IIweights',1)] = 1.0
    # L6
    initCfg[('IEweights',2)] =  1.2  
    initCfg[('IIweights',2)] =  1.0
    initCfg[('IIweights',2)] =  0.8

    groupedParams = [('NetStim1', 'interval'), ('NetStim1', 'number')] 

    b = Batch(params=params, initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'

    return b

# ----------------------------------------------------------------------------------------------
# Local pop stimulation
# ----------------------------------------------------------------------------------------------
def localPopStims():
    params = specs.ODict()

    params['ihGbar'] = [0.0, 1.0, 2.0]
    params[('NetStim1', 'pop')] = ['IT2','IT4','IT5A','IT5B','PT5B','IT6','CT6']
    params[('NetStim1', 'interval')] = [1000.0/20.0, 1000.0/30.0]

    b = Batch(params=params)
    b.method = 'grid'

    grouped = []

    for p in b.params:
        if p['label'] in grouped: 
            p['group'] = True

    return b


# ----------------------------------------------------------------------------------------------
# EPSPs via NetStim
# ----------------------------------------------------------------------------------------------
def EPSPs():
    params = specs.ODict()

    params['groupWeight'] = [x*0.05 for x in np.arange(1, 8, 1)]
    params['ihGbar'] = [0.0, 1.0]
 
    
    # initial config
    initCfg = {}
    initCfg['duration'] = 0.5*1e3
    initCfg['addIClamp'] = False
    initCfg['addNetStim'] = True
    initCfg[('GroupNetStimW1', 'pop')] = 'PT5B'
    initCfg[('analysis','plotTraces','timeRange')] = [0, 500]
    initCfg['excTau2Factor'] = 2.0
    initCfg['weightNorm'] = True
    initCfg['stimSubConn'] = False
    initCfg['ihGbarZD'] = None

    groupedParams = [] 

    b = Batch(params=params, netParamsFile='netParams_cell.py', cfgFile='cfg_cell.py', initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'

    return b


# ----------------------------------------------------------------------------------------------
# f-I curve
# ----------------------------------------------------------------------------------------------
def fIcurve(pops = [], amps = list(np.arange(0.0, 6.5, 0.5)/10.0) ):
    params = specs.ODict()

    params['singlePop'] = pops
    params[('IClamp1', 'amp')] = amps
    #params['ihGbar'] = [0.0, 1.0, 2.0]
    # params['axonNa'] = [5, 6, 7, 8] 
    # params['gpas'] = [0.6, 0.65, 0.70, 0.75] 
    # params['epas'] = [1.0, 1.05] 
    # params['ihLkcBasal'] = [0.0, 0.01, 0.1, 0.5, 1.0] 

    # initial config
    initCfg = {}
    initCfg['duration'] = 2.0*1e3
    initCfg['addIClamp'] = True
    initCfg['addNetStim'] = False
    initCfg['weightNorm'] = True
    initCfg[('IClamp1','sec')] = 'soma'
    initCfg[('IClamp1','loc')] = 0.5
    initCfg[('IClamp1','start')] = 750
    initCfg[('IClamp1','dur')] = 1000
    initCfg[('analysis', 'plotTraces', 'timeRange')] = [0, 2000]
    initCfg['printPopAvgRates'] = [750,1750]

    initCfg[('hParams', 'celsius')] = 37

    ## turn off components not required
    initCfg['addBkgConn'] = False
    initCfg['addConn'] = False
    initCfg['addIntraThalamicConn'] = False
    initCfg['addIntraThalamicConn'] = False
    initCfg['addCorticoThalamicConn'] = False
    initCfg['addCoreThalamoCorticalConn'] = False
    initCfg['addMatrixThalamoCorticalConn'] = False
    initCfg['stimSubConn'] = False
    initCfg['addNetStim'] = False

    groupedParams = [] 

    b = Batch(params=params, netParamsFile='netParams_cell.py', cfgFile='cfg_cell.py', initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'

    return b


# ----------------------------------------------------------------------------------------------
# Custom
# ----------------------------------------------------------------------------------------------
def custom():
    params = specs.ODict()

    # conn gains
    # params[('IELayerGain', '1-3')] = [1.9609935, 1.9609935 - 0.1, 1.9609935 - 0.2] 
    # params[('IELayerGain', '4')] = [1.973369532, 1.973369532 - 0.1, 1.973369532 - 0.2]
    # params[('IELayerGain', '5')] = [0.547478256, 0.547478256 - 0.1, 0.547478256 - 0.2]	
    # params[('IELayerGain', '6')] = [0.817050621, 0.817050621 - 0.1, 0.817050621 - 0.2]
    
    #params['thalamoCorticalGain'] = [1.434715802, 2.0]
    #params[('ICThalInput', 'probE')] = [0.12, 0.25]#, 0.5]
    params[('ICThalInput', 'probI')] = [0.12, 0.25, 0.5]
    
    groupedParams = [] #('IELayerGain', '1-3'), ('IELayerGain', '4'), ('IELayerGain', '5'), ('IELayerGain', '6')]

    # --------------------------------------------------------
    # initial config
    initCfg = {}
    initCfg = {}
    initCfg['duration'] = 1750
    initCfg['printPopAvgRates'] = [250, 1750] 
    initCfg['dt'] = 0.05

    initCfg['scaleDensity'] = 0.5

    # plotting and saving params
    initCfg[('analysis','plotRaster','timeRange')] = initCfg['printPopAvgRates']
    initCfg[('analysis', 'plotTraces', 'timeRange')] = initCfg['printPopAvgRates']
    initCfg[('analysis', 'plotLFP', 'timeRange')] = initCfg['printPopAvgRates']
    
    initCfg[('analysis', 'plotTraces', 'oneFigPer')] = 'trace'

    initCfg['addConn'] = True # test only bkg inputs

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False


    ''' from v23_batch8'''
    initCfg['EEGain']=0.10928952347451457
    initCfg['EIGain']=0.13089042807412776
    initCfg[('IELayerGain', '1-3')]= 1.0
    initCfg[('IELayerGain', '4')]=1.0
    initCfg[('IELayerGain', '5')]=1.0
    initCfg[('IELayerGain', '6')]=1.5743753119276733
    initCfg[('IILayerGain', '1-3')]=1.0
    initCfg[('IILayerGain', '4')]=1.0 
    initCfg[('IILayerGain', '5')]=1.0
    initCfg[('IILayerGain', '6')]=0.8550747910383769
    initCfg['thalamoCorticalGain']=1.964478741362849
    initCfg['intraThalamicGain']=0.35778625557189936
    initCfg['corticoThalamicGain'] = 1.4715575641214615
    

    ''' from v24_batch3
    initCfg['EEGain'] = 1.7930365644528616
    initCfg['EIGain'] = 1.301292631	
    
    initCfg[('IELayerGain', '1-3')] = 1.9609935 - 0.15
    initCfg[('IELayerGain', '4')] = 1.973369532	- 0.15
    initCfg[('IELayerGain', '5')] = 0.547478256	- 0.15
    initCfg[('IELayerGain', '6')] = 0.817050621	- 0.15

    initCfg[('IILayerGain', '1-3')] = 0.575910457
    initCfg[('IILayerGain', '4')] = 0.506134474	
    initCfg[('IILayerGain', '5')] = 1.140789303	
    initCfg[('IILayerGain', '6')] = 1.999973065	

    initCfg['thalamoCorticalGain'] = 1.434715802
    initCfg['intraThalamicGain'] = 1.987386358	
    initCfg['corticoThalamicGain'] = 1.354024353042513
    '''

    
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'

    return b

# ----------------------------------------------------------------------------------------------
# Evol
# ----------------------------------------------------------------------------------------------
def evolRates():
    # --------------------------------------------------------
    # parameters
    params = specs.ODict()

    # bkg inputs
    params['EEGain'] = [0.5, 2.0]
    params['EIGain'] = [0.5, 2.0]

    params[('IELayerGain', '1-3')] = [0.5, 2.0]
    params[('IELayerGain', '4')] = [0.5, 2.0]
    params[('IELayerGain', '5')] = [0.5, 2.0]
    params[('IELayerGain', '6')] = [0.5, 2.0]

    params[('IILayerGain', '1-3')] = [0.5, 2.0]
    params[('IILayerGain', '4')] = [0.5, 2.0]
    params[('IILayerGain', '5')] = [0.5, 2.0]
    params[('IILayerGain', '6')] = [0.5, 2.0]
    
    params['thalamoCorticalGain'] = [0.5, 2.0]  
    params['intraThalamicGain'] = [0.5, 2.0] 
    params['corticoThalamicGain'] = [0.5, 2.0]

    groupedParams = []

    # --------------------------------------------------------
    # initial config
    initCfg = {}
    initCfg = {}
    initCfg['duration'] = 1500
    initCfg['printPopAvgRates'] = [[500, 750], [750, 1000], [1000, 1250], [1250, 1500]]
    initCfg['dt'] = 0.05

    initCfg['scaleDensity'] = 0.5

    # plotting and saving params
    initCfg[('analysis','plotRaster','timeRange')] = [500,1500]
    initCfg[('analysis', 'plotTraces', 'timeRange')] = [500,1500]

    initCfg[('analysis', 'plotTraces', 'oneFigPer')] = 'trace'

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False


    # --------------------------------------------------------
    # fitness function
    fitnessFuncArgs = {}
    pops = {}
    
    ## Exc pops
    Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6', 'TC', 'TCM', 'HTC']  # all layers + thal + IC

    Etune = {'target': 5, 'width': 20, 'min': 0.05}
    for pop in Epops:
        pops[pop] = Etune
    
    ## Inh pops 
    Ipops = ['NGF1',                            # L1
            'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
            'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
            'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
            'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
            'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
            'PV6', 'SOM6', 'VIP6', 'NGF6',       # L6
            'IRE', 'IREM', 'TI']  # Thal 

    Itune = {'target': 10, 'width': 30, 'min': 0.05}
    for pop in Ipops:
        pops[pop] = Itune
    
    fitnessFuncArgs['pops'] = pops
    fitnessFuncArgs['maxFitness'] = 1000
    fitnessFuncArgs['tranges'] = initCfg['printPopAvgRates']


    def fitnessFunc(simData, **kwargs):
        import numpy as np
        pops = kwargs['pops']
        maxFitness = kwargs['maxFitness']
        tranges = kwargs['tranges']

        popFitnessAll = []

        for trange in tranges:
            popFitnessAll.append([min(np.exp(abs(v['target'] - simData['popRates'][k]['%d_%d'%(trange[0], trange[1])])/v['width']), maxFitness) 
                if simData['popRates'][k]['%d_%d'%(trange[0], trange[1])] > v['min'] else maxFitness for k, v in pops.items()])
        
        popFitness = np.mean(np.array(popFitnessAll), axis=0)
        
        fitness = np.mean(popFitness)

        popInfo = '; '.join(['%s rate=%.1f fit=%1.f' % (p, np.mean(list(simData['popRates'][p].values())), popFitness[i]) for i,p in enumerate(pops)])
        print('  ' + popInfo)

        return fitness

    
    #from IPython import embed; embed()

    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)

    b.method = 'evol'

    # Set evol alg configuration
    b.evolCfg = {
        'evolAlgorithm': 'custom',
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'pop_size': 100,
        'num_elites': 2,
        'mutation_rate': 0.5,
        'crossover': 0.5,
        'maximize': False, # maximize fitness function?
        'max_generations': 200,
        'time_sleep': 150, # 2.5min wait this time before checking again if sim is completed (for each generation)
        'maxiter_wait': 5, # max number of times to check if sim is completed (for each generation)
        'defaultFitness': 1000, # set fitness value in case simulation time is over
        'scancelUser': 'ext_salvadordura_gmail_com'
    }

    return b

# ----------------------------------------------------------------------------------------------
# Adaptive Stochastic Descent (ASD)
# ----------------------------------------------------------------------------------------------
def asdRates():

    # --------------------------------------------------------
    # parameters
    params = specs.ODict()

    # bkg inputs

    params['EEGain'] = [0.5, 2.0, [1.7930365644528616]]
    params['EIGain'] = [0.5, 2.0, [1.301292631]]

    params[('IELayerGain', '1-3')] = [0.5, 2.0, [1.9609935]]
    params[('IELayerGain', '4')] = [0.5, 2.0, [2.0]]
    params[('IELayerGain', '5')] = [0.5, 2.0, [0.547478256]]
    params[('IELayerGain', '6')] = [0.5, 2.0, [0.817050621]]

    params[('IILayerGain', '1-3')] = [0.5, 2.0, [0.5183194113]]
    params[('IILayerGain', '4')] = [0.5, 2.0, [0.506134474]]
    params[('IILayerGain', '5')] = [0.5, 2.0, [1.140789303]]
    params[('IILayerGain', '6')] = [0.5, 2.0, [1.999973065]]
    
    params['thalamoCorticalGain'] = [0.5, 2.0, [1.434715802]]  
    params['intraThalamicGain'] = [0.5, 2.0, [1.987386358]]
    params['corticoThalamicGain'] = [0.5, 2.0, [1.354024353042513]]


    groupedParams = []

    # --------------------------------------------------------
    # initial config
    initCfg = {}
    initCfg = {}
    initCfg['duration'] = 1500
    initCfg['printPopAvgRates'] = [[500, 750], [750, 1000], [1000, 1250], [1250, 1500]]
    initCfg['dt'] = 0.05

    initCfg['scaleDensity'] = 0.5

    # plotting and saving params
    initCfg[('analysis','plotRaster','timeRange')] = [500,1500]
    initCfg[('analysis', 'plotTraces', 'timeRange')] = [500,1500]
    initCfg[('analysis', 'plotTraces', 'oneFigPer')] = 'trace'
    initCfg['recordLFP'] = None
    initCfg[('analysis', 'plotLFP')] = False

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False


    # --------------------------------------------------------
    # fitness function
    fitnessFuncArgs = {}
    pops = {}
    
    ## Exc pops
    Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6', 'TC', 'TCM', 'HTC']  # all layers + thal + IC

    Etune = {'target': 5, 'width': 20, 'min': 0.05}
    for pop in Epops:
        pops[pop] = Etune
    
    ## Inh pops 
    Ipops = ['NGF1',                            # L1
            'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
            'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
            'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
            'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
            'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
            'PV6', 'SOM6', 'VIP6', 'NGF6',       # L6
            'IRE', 'IREM', 'TI']  # Thal 

    Itune = {'target': 10, 'width': 30, 'min': 0.05}
    for pop in Ipops:
        pops[pop] = Itune
    
    fitnessFuncArgs['pops'] = pops
    fitnessFuncArgs['maxFitness'] = 1000
    fitnessFuncArgs['tranges'] = initCfg['printPopAvgRates']


    def fitnessFunc(simData, **kwargs):
        import numpy as np
        pops = kwargs['pops']
        maxFitness = kwargs['maxFitness']
        tranges = kwargs['tranges']

        popFitnessAll = []

        for trange in tranges:
            popFitnessAll.append([min(np.exp(abs(v['target'] - simData['popRates'][k]['%d_%d'%(trange[0], trange[1])])/v['width']), maxFitness) 
                if simData['popRates'][k]['%d_%d'%(trange[0], trange[1])] > v['min'] else maxFitness for k, v in pops.items()])
        
        popFitness = np.mean(np.array(popFitnessAll), axis=0)
        
        fitness = np.mean(popFitness)

        popInfo = '; '.join(['%s rate=%.1f fit=%1.f' % (p, np.mean(list(simData['popRates'][p].values())), popFitness[i]) for i,p in enumerate(pops)])
        print('  ' + popInfo)

        return fitness
        
    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)

    b.method = 'asd'

    b.optimCfg = {
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'stepsize':     0.1,     #   Initial step size as a fraction of each parameter
        'sinc':         1.5,       #   Step size learning rate (increase)
        'sdec':         1.5,       #   Step size learning rate (decrease)
        'pinc':         2,       #   Parameter selection learning rate (increase)
        'pdec':         2,       #   Parameter selection learning rate (decrease)
        #'pinitial':     None,    #    Set initial parameter selection probabilities
        #'sinitial':     None,    #    Set initial step sizes; if empty, calculated from stepsize instead
        'maxiters':     100,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
        'maxtime':      360000,    #    Maximum time allowed, in seconds
        'abstol':       1e-6,    #    Minimum absolute change in objective function
        'reltol':       1e-3,    #    Minimum relative change in objective function
        #'stalliters':   10*len(params)*len(params),  #    Number of iterations over which to calculate TolFun (n = number of parameters)
        #'stoppingfunc': None,    #    External method that can be used to stop the calculation from the outside.
        #'randseed':     None,    #    The random seed to use
        'verbose':      2,       #    How much information to print during the run
        #'label':        None    #    A label to use to annotate the output
        'time_sleep': 60, # 2.5min wait this time before checking again if sim is completed (for each generation)
        'maxiter_wait': 30,  # max number of times to check if sim is completed (for each generation)
        'popsize': 1
    }

    return b

# ----------------------------------------------------------------------------------------------
# Run configurations
# ----------------------------------------------------------------------------------------------
def setRunCfg(b, type='mpi_bulletin'):
    if type=='mpi_bulletin':
        b.runCfg = {'type': 'mpi_bulletin', 
            'script': 'init_cell.py', 
            'skip': True}

    elif type=='mpi_direct':
        b.runCfg = {'type': 'mpi_direct',
            'nodes': 1,
            'coresPerNode': 96,
            'script': 'init.py',
            'mpiCommand': 'mpirun',
            'skip': True}

    elif type=='hpc_torque':
        b.runCfg = {'type': 'hpc_torque',
             'script': 'init.py',
             'nodes': 3,
             'ppn': 8,
             'walltime': "12:00:00",
             'queueName': 'longerq',
             'sleepInterval': 5,
             'skip': True}

    elif type=='hpc_slurm_comet':
        b.runCfg = {'type': 'hpc_slurm', 
            'allocation': 'shs100', # bridges='ib4iflp', comet m1='shs100', comet nsg='csd403'
            #'reservation': 'salva1',
            'walltime': '6:00:00',
            'nodes': 4,
            'coresPerNode': 24,  # comet=24, bridges=28
            'email': 'salvadordura@gmail.com',
            'folder': '/home/salvadord/m1/sim/',  # comet='/salvadord', bridges='/salvi82'
            'script': 'init.py', 
            'mpiCommand': 'ibrun', # comet='ibrun', bridges='mpirun'
            'skipCustom': '_raster.png'}

    elif type=='hpc_slurm_gcp':
        b.runCfg = {'type': 'hpc_slurm', 
            'allocation': 'default', # bridges='ib4iflp', comet m1='shs100', comet nsg='csd403', gcp='default'
            'walltime': '24:00:00', #'48:00:00',
            'nodes': 1,
            'coresPerNode': 96,  # comet=24, bridges=28, gcp=32
            'email': 'salvadordura@gmail.com',
            'folder': '/home/ext_salvadordura_gmail_com/A1/',  # comet,gcp='/salvadord', bridges='/salvi82'
            'script': 'init.py', 
            'mpiCommand': 'mpirun', # comet='ibrun', bridges,gcp='mpirun' 
            'skipCustom': '_raster.png'}
            #'custom': '#SBATCH --exclude=compute[17-64000]'} # only use first 16 nodes (non-preemptible for long runs )
            # --nodelist=compute1


    elif type=='hpc_slurm_bridges':
        b.runCfg = {'type': 'hpc_slurm', 
            'allocation': 'ib4iflp', # bridges='ib4iflp', comet m1='shs100', comet nsg='csd403'
            'walltime': '06:00:00',
            'nodes': 2,
            'coresPerNode': 28,  # comet=24, bridges=28
            'email': 'salvadordura@gmail.com',
            'folder': '/home/salvi82/m1/sim/',  # comet='/salvadord', bridges='/salvi82'
            'script': 'init.py', 
            'mpiCommand': 'mpirun', # comet='ibrun', bridges='mpirun'
            'skip': True}



# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':

    cellTypes = ['IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6', 'TC', 'HTC', 'IRE', 'TI']

    #b = custom()
    # b = evolRates()
    b = asdRates()
    # b = bkgWeights(pops = cellTypes, weights = list(np.arange(1,100)))
    # b = bkgWeights2D(pops = ['ITS4'], weights = list(np.arange(0,150,10)))
    #b = fIcurve(pops=['ITS4']) 

    b.batchLabel = 'v24_batch11'
    b.saveFolder = 'data/'+b.batchLabel

    setRunCfg(b, 'mpi_direct') #'mpi_bulletin') #'hpc_slurm_gcp') #'hpc_slurm_gcp') #'mpi_bulletin') #'hpc_slurm_gcp')
    b.run() # run batch


    # #Submit set of batch sims together (eg. for weight norm)

    # # for weightNorm need to group cell types by those that have the same section names (one cell rule for each) 
    # popsWeightNorm =    {#'IT2_reduced': ['CT5A', 'CT5B']}#, #'IT2', 'IT3', 'ITP4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6'],
    #                      'ITS4_reduced': ['ITS4']}
    # # #                     'PV_reduced': ['PV2', 'SOM2'],
    #                      'VIP_reduced': ['VIP2'],
    #                      #'NGF_reduced': ['NGF2']} #,
    # #                     #'RE_reduced': ['IRE', 'TC', 'HTC'],
    # #                     #'TI_reduced': ['TI']}
 
    # batchIndex = 7
    # for k, v in popsWeightNorm.items(): 
    #     b = weightNorm(pops=v, rule=k)
    #     b.batchLabel = 'v24_batch'+str(batchIndex) 
    #     b.saveFolder = 'data/'+b.batchLabel
    #     b.method = 'grid'  # evol
    #     setRunCfg(b, 'mpi_bulletin')
    #     b.run()  # run batch
    #     batchIndex += 1


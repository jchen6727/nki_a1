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

    return b


# ----------------------------------------------------------------------------------------------
# f-I curve
# ----------------------------------------------------------------------------------------------
def fIcurve():
    params = specs.ODict()

    params[('IClamp1', 'pop')] = ['IT2', 'IT4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6', 'PV2', 'SOM2']
    params[('IClamp1', 'amp')] = list(np.arange(0.0, 6.5, 0.5)/10.0) 
    #params['ihGbar'] = [0.0, 1.0, 2.0]
    # params['axonNa'] = [5, 6, 7, 8] 
    # params['gpas'] = [0.6, 0.65, 0.70, 0.75] 
    # params['epas'] = [1.0, 1.05] 
    # params['ihLkcBasal'] = [0.0, 0.01, 0.1, 0.5, 1.0] 

    # initial config
    initCfg = {}
    initCfg['duration'] = 1.5*1e3
    initCfg['addIClamp'] = True
    initCfg['addNetStim'] = False
    initCfg['weightNorm'] = True
    initCfg[('IClamp1','sec')] = 'soma'
    initCfg[('IClamp1','loc')] = 0.5
    initCfg[('IClamp1','start')] = 500
    initCfg[('IClamp1','dur')] = 1000
    initCfg[('analysis','plotTraces','timeRange')] = [0, 1500]

    groupedParams = [] 

    b = Batch(params=params, netParamsFile='netParams_cell.py', cfgFile='cfg_cell.py', initCfg=initCfg, groupedParams=groupedParams)

    return b


# ----------------------------------------------------------------------------------------------
# Custom
# ----------------------------------------------------------------------------------------------
def custom():
    params = specs.ODict()

    params[('ratesLong', 'TPO', 1)] = [5, 5] 	#[2,4,2,2,4,2,4,4]
    params[('ratesLong', 'TVL', 1)] = [2, 2.5] #[2,4,2,2,4,2,4,4]
    params[('ratesLong', 'S1', 1)] =  [5, 5] 	#[2,2,4,2,4,4,2,4]
    params[('ratesLong', 'S2', 1)] =  [5, 5] 	#[2,2,4,2,4,4,2,4]
    params[('ratesLong', 'cM1', 1)] = [2, 2.5] #[2,2,2,4,2,4,4,4]
    params[('ratesLong', 'M2', 1)] =  [2, 2.5] #[2,2,2,4,2,4,4,4]
    params[('ratesLong', 'OC', 1)] =  [5, 5]	

    # # L2/3+4
    params[('IEweights',0)] =  [0.6,0.8]
    params[('IIweights',0)] =  [1.0,1.2] #[0.8, 1.0]   
    # L5
    params[('IEweights',1)] = [0.8] #[0.8, 1.0]   
    params[('IIweights',1)] = [1.0] #[0.8, 1.0]
    # L6
    params[('IEweights',2)] =  [1.0] # [0.8, 1.0]  
    # params[('IIweights',2)] =  [0.8, 1.0, 1.2]

    params[('pulse', 'rate')] = [10.0] # [10.0, 15.0]

    params[('pulse', 'pop')] = ['S2', 'M2'] # ['TPO', 'S1', 'S2', 'TVL', 'cM1', 'M2']


    params['ihGbar'] = [0.25, 1.0] #[0.2, 0.25, 0.3, 1.0]

    groupedParams = [('ratesLong', 'TPO', 1), ('ratesLong', 'TVL', 1),
                    ('ratesLong', 'S1', 1), ('ratesLong', 'S2', 1),
                    ('ratesLong', 'cM1', 1), ('ratesLong', 'M2', 1),
                    ('ratesLong', 'OC', 1)] # ['IEGain','IIGain'] #'EEGain', 'EPVGain', 'ESOMGain', 'PVEGain', 'SOMEGain', 'PVIGain', 'SOMIGain']

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

    #initCfg[('pulse', 'pop')] = 'None'
    #initCfg[('pulse', 'rate')] = 10.0
    initCfg[('pulse', 'start')] = 1000.0
    initCfg[('pulse', 'end')] = 1100.0
    initCfg[('pulse', 'noise')] = 0.8

    initCfg['IEdisynapticBias'] = None

    initCfg['EEGain'] = 0.5 # [0.6, 0.8] #[0.8, 1.0]

    initCfg['weightNormThreshold'] = 4.0
    initCfg['IEGain'] = 1.0
    initCfg['IIGain'] = 1.0
    initCfg['IPTGain'] = 1.0
    initCfg['IIweights'] =  [1.0, 1.0, 1.0]

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False



# ----------------------------------------------------------------------------------------------
# Evol
# ----------------------------------------------------------------------------------------------
def evolRates():
    # --------------------------------------------------------
    # parameters
    params = specs.ODict()

    # long-range inputs
    params[('ratesLong', 'TPO', 1)] = [2, 5] 	#[2,4,2,2,4,2,4,4]
    params[('ratesLong', 'TVL', 1)] = [2, 5] #[2,4,2,2,4,2,4,4]
    #params[('ratesLong', 'S1', 1)] =  [2, 5] 	#[2,2,4,2,4,4,2,4]
    params[('ratesLong', 'S2', 1)] =  [2, 5] 	#[2,2,4,2,4,4,2,4]
    #params[('ratesLong', 'cM1', 1)] =  [2, 5] 
    params[('ratesLong', 'M2', 1)] =  [2, 5] #[2,2,2,4,2,4,4,4]
    params[('ratesLong', 'OC', 1)] =  [2, 5]	

    # EEgain
    params['EEGain'] = [0.5, 1.0] 

    # IEgain
    ## L2/3+4
    params[('IEweights',0)] =  [0.8, 1.2]
    ## L5
    params[('IEweights',1)] = [0.8, 1.2] #[0.8, 1.0]   
    ## L6
    params[('IEweights',2)] =  [0.8, 1.2] # [0.8, 1.0]  

    # IIGain
    params['IIGain'] = [0.8, 1.2]
    
    # ih - remove
    #params['ihGbar'] = [0.1, 0.3] #[0.2, 0.25, 0.3, 1.0] 

    groupedParams = []

    # --------------------------------------------------------
    # initial config
    initCfg = {}
    initCfg['duration'] = 1.5*1e3
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

    initCfg[('pulse', 'pop')] = 'None'
    initCfg[('pulse', 'start')] = 1000.0
    initCfg[('pulse', 'end')] = 1100.0
    initCfg[('pulse', 'noise')] = 0.8

    initCfg['IEdisynapticBias'] = None

    initCfg['weightNormThreshold'] = 4.0
    initCfg['IEGain'] = 1.0
    initCfg['IPTGain'] = 1.0
    initCfg['IIweights'] =  [1.0, 1.0, 1.0]

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False


    # --------------------------------------------------------
    # fitness function
    fitnessFuncArgs = {}
        
    ## complex net
    pops = {} 
    Etune = {'target': 10, 'width': 5, 'min': 0.5}
    pops['IT2'] = Etune
    pops['IT4'] = Etune
    pops['IT5A'] = Etune 
    pops['IT5B'] = Etune  
    pops['PT5B'] = Etune 
    pops['IT6'] =  Etune
    pops['CT6'] =  Etune

    Itune = {'target': 20, 'width': 15, 'min': 0.25}
    pops['PV2'] = Itune
    pops['SOM2'] = Itune
    pops['PV5A'] = Itune
    pops['SOM5A'] = Itune
    pops['PV5B'] = Itune
    pops['SOM5B'] = Itune
    pops['PV6'] = Itune
    pops['SOM6'] = Itune
    
    fitnessFuncArgs['pops'] = pops
    fitnessFuncArgs['maxFitness'] = 1000


    def fitnessFunc(simData, **kwargs):
        import numpy as np
        pops = kwargs['pops']
        maxFitness = kwargs['maxFitness']
        popFitness = [min(np.exp(abs(v['target'] - simData['popRates'][k])/v['width']), maxFitness) 
                if simData['popRates'][k] > v['min'] else maxFitness for k,v in pops.iteritems()]
        fitness = np.mean(popFitness)

        popInfo = '; '.join(['%s rate=%.1f fit=%1.f'%(p, simData['popRates'][p], popFitness[i]) for i,p in enumerate(pops)])
        print('  '+popInfo)
        return fitness
    
    #from IPython import embed; embed()

    b = Batch(params=params, groupedParams=groupedParams, initCfg=initCfg)

    # Set evol alg configuration
    b.evolCfg = {
        'evolAlgorithm': 'custom',
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'pop_size': 30,
        'num_elites': 2,
        'mutation_rate': 0.4,
        'crossover': 0.5,
        'maximize': False, # maximize fitness function?
        'max_generations': 200,
        'time_sleep': 90*10, # (15 min) wait this time before checking again if sim is completed (for each generation)
        'maxiter_wait': 48*4, # (48h) max number of times to check if sim is completed (for each generation)
        'defaultFitness': 1000, # set fitness value in case simulation time is over
        'scancelUser': 'salvadord'
    }


    return b


# ----------------------------------------------------------------------------------------------
# Run configurations
# ----------------------------------------------------------------------------------------------
def setRunCfg(b, type='mpi_bulletin'):
    if type=='mpi_bulletin':
        b.runCfg = {'type': 'mpi_bulletin', 
            'script': 'init.py', 
            'skip': True}

    elif type=='mpi_direct':
        b.runCfg = {'type': 'mpi_direct',
            'cores': 4,
            'script': 'init_cell.py',
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
            'walltime': '48:00:30', #'48:00:00',
            'nodes': 16,
            'coresPerNode': 8,  # comet=24, bridges=28, gcp=32
            'email': 'salvadordura@gmail.com',
            'folder': '/home/salvadord/m1/sim/',  # comet,gcp='/salvadord', bridges='/salvi82'
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

    b = EIbalance()
    # b = longBalance()
    # b = longPopStims()
    # b = recordedLongPopStims() 
    # b = simultLongPopStims()
    # b = freqStims()
    # b = localPopStims()
    # b = fIcurve()
    # b = EPSPs()
    # b = custom()
    #b = evolRates()

    # b.batchLabel = 'v11_batch7' 
    # b.saveFolder = 'data/'+b.batchLabel
    # b.method = 'grid'  # evol
    # setRunCfg(b, 'mpi_bulletin')
    # b.run() # run batch

    # for weightNorm need to group cell types by those that have the same section names (one cell rule for each) 
    popsWeightNorm =    {#'IT2_A1': ['IT2', 'IT3', 'ITP4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6'],
    #                     'ITS4_reduced': ['ITS4'],
    #                     'PV_reduced': ['PV2', 'SOM2'],
    #                     'VIP_reduced': ['VIP2'],
    #                     'NGF_reduced': ['NGF2'],
                         'RE_reduced': ['IRE', 'TC', 'HTC']}
 
    batchIndex = 6
    for k, v in popsWeightNorm.items(): 
        b = weightNorm(pops=v, rule=k)
        b.batchLabel = 'v11_batch'+str(batchIndex) 
        b.saveFolder = 'data/'+b.batchLabel
        b.method = 'grid'  # evol
        setRunCfg(b, 'mpi_bulletin')
        b.run()  # run batch
        batchIndex += 1


"""
batch.py 

Batch simulation for M1 model using NetPyNE

Contributors: salvadordura@gmail.com, samuel.neymotin@nki.rfmh.org
"""
from netpyne.batch import Batch
from netpyne import specs
import numpy as np
import pickle

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
    # NGF
    # initCfg['tune'] = { "L": 0.5292094921519722,
    #                     "Ra": 1.4789488212279527,
    #                     "ch_CavL": {
    #                         "gmax": 0.8294009810179838
    #                     },
    #                     "ch_CavN": {
    #                         "gmax": 1.8302558031992033
    #                     },
    #                     "ch_KCaS": {
    #                         "gmax": 1.9966915506612826
    #                     },
    #                     "ch_Kdrfastngf": {
    #                         "gmax": 1.9718693540420718
    #                     },
    #                     "ch_KvAngf": {
    #                         "gmax": 1.9995297406303094
    #                     },
    #                     "ch_KvCaB": {
    #                         "gmax": 1.386572752588786
    #                     },
    #                     "ch_Navngf": {
    #                         "gmax": 1.9747105482249578
    #                     },
    #                     "cm": 1.6538025216246877,
    #                     "diam": 0.9280430551297493,
    #                     "hd": {
    #                         "gbar": 1.359096548635795
    #                     },
    #                     "pas": {
    #                         "e": 0.6789102769330073,
    #                         "g": 0.8144720557328889
    #                     }}
    #ITS4
    initCfg['tune'] = { "L": 1.3118623085123142,
            "Nca": {
                "gmax": 3.5838808831900515
            },
            "Ra": 0.6872551957221755,
            "cm": 2.2202371016519873,
            "diam": 2.7938082173487877,
            "kca": {
                "gbar": 0.40727688068272255
            },
            "km": {
                "gbar": 1.3266426599463168
            },
            "kv": {
                "gbar": 0.2713752431207888
            },
            "naz": {
                "gmax": 3.635282589182774
            },
            "pas": {
                "e": 1.0950678422289308,
                "g": 1.6892125376050984
            }}
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
def fIcurve(pops = [], amps = list(np.arange(0.0, 12.5, 0.5)/10.0) ):
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
# Custom spont
# ----------------------------------------------------------------------------------------------
def custom_spont(filename):
    params = specs.ODict()
    if not filename:
        filename = 'data/v34_batch25/trial_2142/trial_2142_cfg.json'  # 'data/simDataFiles/spont/jsonCfgFiles/A1_v34_batch27_v34_batch27_0_3_cfg.json'
    # from prev 
    import json
    with open(filename, 'rb') as f:
        cfgLoad = json.load(f)['simConfig']
    cfgLoad2 = cfgLoad    
    #params['thalamoCorticalGain'] = [cfgLoad['thalamoCorticalGain']] # [cfgLoad['thalamoCorticalGain']*0.75, cfgLoad['thalamoCorticalGain'], cfgLoad['thalamoCorticalGain']*1.25]    
    params[('seeds', 'conn')] = [2] #list(range(3)) #list(range(5)) #[3, 3]  #[3]#[3, 3] #list(range(1)) #[4321+(17*i) for i in range(5)]
    params[('seeds', 'stim')] = [3] #list(range(5)) #[2, 3]  #[4]#[2, 3] #list(range(1)) #[1234+(17*i) for i in range(5)]
    # params['ihGbar'] = [0.25, 0.5] #[0.75, 1.0, 1.25]
    # params['KgbarFactor'] = [0.25, 0.5] #[0.75, 1.0, 1.25]
    groupedParams = [] #('ICThalInput', 'probE'), ('ICThalInput', 'probI')] #('IELayerGain', '1-3'), ('IELayerGain', '4'), ('IELayerGain', '5'), ('IELayerGain', '6')]
    # --------------------------------------------------------
    # initial config
    initCfg = {} # set default options from prev sim    
    initCfg['duration'] = 11500                                      # 11500
    initCfg['printPopAvgRates'] = [1500, initCfg['duration']] 
    initCfg['scaleDensity'] = 1.0 #0.25                              # 1.0
    initCfg['recordStep'] = 0.05
    ### I DON'T KNOW IF I SHOULD HAVE THIS COMMENTED OR UNCOMMENTED???
    # initCfg[('seeds', 'conn')] = 0
    # initCfg[('seeds', 'stim')] = 0
    # plotting and saving params
    # initCfg[('analysis','plotRaster','timeRange')] = initCfg['printPopAvgRates']
    #initCfg[('analysis', 'plotTraces', 'timeRange')] = initCfg['printPopAvgRates']
    #initCfg[('analysis', 'plotSpikeStats', 'timeRange')] = initCfg['printPopAvgRates']
    #initCfg[('analysis', 'plotLFP', 'timeRange')] = initCfg['printPopAvgRates']
    #initCfg[('analysis', 'plotCSD', 'timeRange')] = [1500, 1700]
    # changed directly in cfg.py    
    #initCfg[('analysis', 'plotCSD')] = {'spacing_um': 100, 'timeRange': initCfg['printPopAvgRates'], 'LFP_overlay': 1, 'layer_lines': 1, 'saveFig': 1, 'showFig': 0}
    #initCfg['recordLFP'] = [[100, y, 100] for y in range(0, 2000, 100)]
    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False    
    # from prev - best of 50% cell density
    updateParams = ['EEGain', 'EIGain', 'IEGain', 'IIGain',
                    ('EICellTypeGain', 'PV'), ('EICellTypeGain', 'SOM'), ('EICellTypeGain', 'VIP'), ('EICellTypeGain', 'NGF'),
                    ('IECellTypeGain', 'PV'), ('IECellTypeGain', 'SOM'), ('IECellTypeGain', 'VIP'), ('IECellTypeGain', 'NGF'),
                    ('EILayerGain', '1'), ('IILayerGain', '1'),
                    ('EELayerGain', '2'), ('EILayerGain', '2'),  ('IELayerGain', '2'), ('IILayerGain', '2'), 
                    ('EELayerGain', '3'), ('EILayerGain', '3'), ('IELayerGain', '3'), ('IILayerGain', '3'), 
                    ('EELayerGain', '4'), ('EILayerGain', '4'), ('IELayerGain', '4'), ('IILayerGain', '4'), 
                    ('EELayerGain', '5A'), ('EILayerGain', '5A'), ('IELayerGain', '5A'), ('IILayerGain', '5A'), 
                    ('EELayerGain', '5B'), ('EILayerGain', '5B'), ('IELayerGain', '5B'), ('IILayerGain', '5B'), 
                    ('EELayerGain', '6'), ('EILayerGain', '6'), ('IELayerGain', '6'), ('IILayerGain', '6')] 
    for p in updateParams:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})
    # good thal params for 100% cell density 
    updateParams2 = ['thalamoCorticalGain', 'intraThalamicGain', 'EbkgThalamicGain', 'IbkgThalamicGain', 'wmat']
    for p in updateParams2:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad2[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad2[p]})
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'
    return b

# ----------------------------------------------------------------------------------------------
# Custom speech stim
# ----------------------------------------------------------------------------------------------
def custom_speech(filename):
    params = specs.ODict()
    if not filename:
        filename = 'data/v34_batch25/trial_2142/trial_2142_cfg.json'
    # from prev 
    import json
    with open(filename, 'rb') as f:
        cfgLoad = json.load(f)['simConfig']
    cfgLoad2 = cfgLoad
    '''
    params[('ICThalInput', 'probE')] = [0.26] #[0.12, 0.26] # 0,1,2
    params[('ICThalInput', 'probI')] = [0.12, 0.26] # 0,1,2
    params[('ICThalInput', 'weightE')] = [0.25, 0.5]
    params[('ICThalInput', 'weightI')] = [0.25, 0.5]
    params['thalamoCorticalGain'] = [cfgLoad['thalamoCorticalGain'] * x for x in [0.9, 0.95, 1.0, 1.1, 1.2]]# , 1.5, 1.75, 2.0]]  #[0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    '''
    #### COMMENTING OUT LINES 829-834 (EYG 9/18/22) FOR ATTEMPT AT SPEECH-EEG RUN ON CINECA!! #### 
    # params[('wmat', 'TC', 'ITS4')] = [0.7, 0.8]
    # params[('wmat', 'TC', 'ITP4')] = [0.7, 0.8]
    # params[('wmat', 'HTC', 'ITS4')] = [0.7, 0.8]
    # params[('wmat', 'HTC', 'ITP4')] = [0.7, 0.8]
    # params[('wmat', 'TC', 'PV4')] = [0.2, 0.3, 0.4, 0.5]
    #params[('wmat', 'HTC', 'PV4')] = [0.3, 0.4, 0.5]
    #### NOT COMMENTING OUT LINE 838 (EYG 9/22/22) FOR ATTEMPT AT SPEECH-EEG RUN ON CINECA!! #### 
    params[('ICThalInput', 'startTime')] = [[2500, 4000, 5500, 7000]] #[[2000, 3500, 5000, 6500, 8000], [2500, 4000, 5500, 7000]] #[2500, 2550] # [2500, 2550, 2600, 2650]
    # conn gains 
    #params['thalamoCorticalGain'] = [cfgLoad['thalamoCorticalGain']] # [cfgLoad['thalamoCorticalGain']*0.75, cfgLoad['thalamoCorticalGain'], cfgLoad['thalamoCorticalGain']*1.25]
    # # Different dt and recordSteps -- No longer necessary; bug fixed 
    # params['dt'] = [0.05, 0.1]
    # params['recordStep'] = [0.05, 0.1]
    # params[('seeds', 'conn')] = list(range(1)) #[4321+(17*i) for i in range(5)] # list(range(5)) 
    # params[('seeds', 'stim')] = list(range(1)) #[1234+(17*i) for i in range(5)] # list(range(5)) 
    ###### SETTING GROUPEDPARAMS TO [] (EYG, 9/18/22) FOR ATTEMPT AT SPEECH-EEG RUN ON CINECA!! #### 
    groupedParams = [] #[('wmat', 'TC', 'ITS4'), ('wmat', 'TC', 'ITP4'), ('wmat', 'HTC', 'ITS4'), ('wmat', 'HTC', 'ITP4')] #('ICThalInput', 'probE'), ('ICThalInput', 'probI')] #('IELayerGain', '1-3'), ('IELayerGain', '4'), ('IELayerGain', '5'), ('IELayerGain', '6')]
    # --------------------------------------------------------
    # initial config
    initCfg = {} # set default options from prev sim    
    initCfg['duration'] = 10000 # 7500 # 4500
    initCfg['printPopAvgRates'] = [1500, 10000] #[1500, 7500] # [1500, 9500]  # [1500, 4500] 
    initCfg['scaleDensity'] = 1.0
    initCfg['recordStep'] = 0.05
    # plotting and saving params
    # initCfg[('analysis','plotRaster','timeRange')] = initCfg['printPopAvgRates'] # MAY NEED TO BE 'plotting' rather than 'analysis'
    #initCfg[('analysis', 'plotTraces', 'timeRange')] = initCfg['printPopAvgRates']
    #initCfg[('analysis', 'plotSpikeStats', 'timeRange')] = initCfg['printPopAvgRates']
    #initCfg[('analysis', 'plotLFP', 'timeRange')] = initCfg['printPopAvgRates']
    #initCfg[('analysis', 'plotCSD', 'timeRange')] = [1500, 1700]
    # 'startTime': 2500, 
    initCfg['ICThalInput'] = {'file': 'data/ICoutput/speechTrials/ICoutput_CF_9600_10400_wav_01_ba_peter_BEZ2018.mat', 
                            'startTime': 2500, 
                            'weightE': 0.25,#1.0, 
                            'weightI': 0.25,#1.0, 
                            'probE': 0.12, 
                            'probI': 0.12, #0.25 
                            'seed': 1}  
    ## orig file: 'data/ICoutput/ICoutput_CF_9600_10400_wav_01_ba_peter.mat' ## EYG 12/02/22 
    ## RECORDING EEG / DIPOLE (EYG, 9/18/22):
    # initCfg['recordDipole'] = True
    # initCfg['saveDipoleCells'] = ['all']
    # initCfg['saveDipolePops'] = cfg.allpops # or is it initCfg['allpops']
    # changed directly in cfg.py    
    #initCfg[('analysis', 'plotCSD')] = {'spacing_um': 100, 'timeRange': initCfg['printPopAvgRates'], 'LFP_overlay': 1, 'layer_lines': 1, 'saveFig': 1, 'showFig': 0}
    #initCfg['recordLFP'] = [[100, y, 100] for y in range(0, 2000, 100)]
    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False    
    # from prev - best of 50% cell density
    updateParams = ['EEGain', 'EIGain', 'IEGain', 'IIGain',
                    ('EICellTypeGain', 'PV'), ('EICellTypeGain', 'SOM'), ('EICellTypeGain', 'VIP'), ('EICellTypeGain', 'NGF'),
                    ('IECellTypeGain', 'PV'), ('IECellTypeGain', 'SOM'), ('IECellTypeGain', 'VIP'), ('IECellTypeGain', 'NGF'),
                    ('EILayerGain', '1'), ('IILayerGain', '1'),
                    ('EELayerGain', '2'), ('EILayerGain', '2'),  ('IELayerGain', '2'), ('IILayerGain', '2'), 
                    ('EELayerGain', '3'), ('EILayerGain', '3'), ('IELayerGain', '3'), ('IILayerGain', '3'), 
                    ('EELayerGain', '4'), ('EILayerGain', '4'), ('IELayerGain', '4'), ('IILayerGain', '4'), 
                    ('EELayerGain', '5A'), ('EILayerGain', '5A'), ('IELayerGain', '5A'), ('IILayerGain', '5A'), 
                    ('EELayerGain', '5B'), ('EILayerGain', '5B'), ('IELayerGain', '5B'), ('IILayerGain', '5B'), 
                    ('EELayerGain', '6'), ('EILayerGain', '6'), ('IELayerGain', '6'), ('IILayerGain', '6')] 
    for p in updateParams:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})
    # good thal params for 100% cell density 
    updateParams2 = ['thalamoCorticalGain', 'intraThalamicGain', 'EbkgThalamicGain', 'IbkgThalamicGain', 'wmat']
    for p in updateParams2:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad2[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad2[p]})
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'
    return b

# ----------------------------------------------------------------------------------------------
# Custom BBN stim
# ----------------------------------------------------------------------------------------------
def custom_BBN(filename):
    params = specs.ODict()
    if not filename:
        filename = 'data/v34_batch25/trial_2142/trial_2142_cfg.json'
    # from prev 
    import json
    with open(filename, 'rb') as f:
        cfgLoad = json.load(f)['simConfig']
    cfgLoad2 = cfgLoad
    '''
    params['thalamoCorticalGain'] = [cfgLoad['thalamoCorticalGain'] * x for x in [0.9, 0.95, 1.0, 1.1, 1.2]]# , 1.5, 1.75, 2.0]]  #[0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    params['thalamoCorticalGain'] = [cfgLoad['thalamoCorticalGain']] # [cfgLoad['thalamoCorticalGain']*0.75, cfgLoad['thalamoCorticalGain'], cfgLoad['thalamoCorticalGain']*1.25]
    '''
    # params[('ICThalInput', 'probE')] = [0.12, 0.26]     # [0.26]    # 0,1,2  
    # params[('ICThalInput', 'probI')] = [0.12, 0.26]                 # 0,1,2
    #params[('ICThalInput', 'weightE')] = [0.25, 0.5]
    #params[('ICThalInput', 'weightI')] = [0.25, 0.5]
    #### UNCOMMENT THIS FOR LOOPED STIMULUS INPUT:  
    # BBN_stimTimes = list(np.arange(2500, 11300, 624.5))
    # BBN_stimTimes_shorterSOA = list(np.arange(2500, 11300, 400))
    # BBN_stimTimes_longerSOA = list(np.arange(2500, 11300, 850))
    # BBN_deltaSOA_startTime2500 = list(np.arange(2500, 12000, 300))
    # BBN_deltaSOA_startTime3450 = list(np.arange(3450, 12000, 300))
    # BBN_deltaSOA_startTime1000 = list(np.arange(1000, 12000, 300))
    stim_noStim_startTime5000 = list(np.arange(5000, 9000, 300))    # 0-1s: discard / 1-5s: quiet   / 5-9s: BBN stim
    stim_noStim_startTime5500 = list(np.arange(5500, 10000, 300))   # 0-1s: discard / 1-5.5s: quiet / 5.5-10s: BBN stim
    stim_noStim_startTime5750 = list(np.arange(5750, 10500, 300))    # 0-1s: discard / 1-5.75s: quiet / 5.75-10.5s: BBN stim
    params[('ICThalInput', 'startTime')] = [stim_noStim_startTime5000, stim_noStim_startTime5500, stim_noStim_startTime5750] #[BBN_deltaSOA_startTime2500, BBN_deltaSOA_startTime3450, BBN_deltaSOA_startTime1000]
    # #### SET CONN AND STIM SEEDS #### 
    # params[('seeds', 'conn')] = [0] #[0,1] #[0,1,4] # list(range(1)) # list(range(5)) 
    # params[('seeds', 'stim')] = [0] #[0,1] #[0,1,4] # list(range(1)) # list(range(5)) 
    #### GROUPED PARAMS #### 
    groupedParams = [] 
    # --------------------------------------------------------
    # initial config
    initCfg = {} # set default options from prev sim    
    initCfg['duration'] = 12000 #11500 
    initCfg['printPopAvgRates'] = [1500, 11500]
    initCfg['scaleDensity'] = 1.0 
    initCfg['recordStep'] = 0.05
    # SET SEEDS FOR CONN AND STIM 
    initCfg[('seeds', 'conn')] = 0
    initCfg[('seeds', 'stim')] = 0
    # # plotting and saving params
    # initCfg[('analysis', 'plotRaster','timeRange')] = initCfg['printPopAvgRates'] # MAY NEED TO BE 'plotting' rather than 'analysis' now? 
    # initCfg[('analysis', 'plotTraces', 'timeRange')] = initCfg['printPopAvgRates']
    # initCfg[('analysis', 'plotSpikeStats', 'timeRange')] = initCfg['printPopAvgRates']
    # initCfg[('analysis', 'plotLFP', 'timeRange')] = initCfg['printPopAvgRates']
    # initCfg[('analysis', 'plotCSD', 'timeRange')] = [1500, 1700]
    ## BBN STIMULUS FOR ICThalInput ## 
    initCfg['ICThalInput'] = {'file': 'data/ICoutput/ICoutput_CF_5256_6056_wav_BBN_100ms_burst.mat', # BBN_trials/ICoutput_CF_9600_10400_wav_BBN_100ms_burst_AN.mat', 
                            'startTime': 2500,
                            'weightE': 0.25,
                            'weightI': 0.25,
                            'probE': 0.12, 
                            'probI': 0.12,
                            'seed': 1}  # SHOULD THIS BE ZERO? 
    ### OPTION TO RECORD EEG / DIPOLE ###
    initCfg['recordDipole'] = False
    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False    
    # from prev - best of 50% cell density
    updateParams = ['EEGain', 'EIGain', 'IEGain', 'IIGain',
                    ('EICellTypeGain', 'PV'), ('EICellTypeGain', 'SOM'), ('EICellTypeGain', 'VIP'), ('EICellTypeGain', 'NGF'),
                    ('IECellTypeGain', 'PV'), ('IECellTypeGain', 'SOM'), ('IECellTypeGain', 'VIP'), ('IECellTypeGain', 'NGF'),
                    ('EILayerGain', '1'), ('IILayerGain', '1'),
                    ('EELayerGain', '2'), ('EILayerGain', '2'),  ('IELayerGain', '2'), ('IILayerGain', '2'), 
                    ('EELayerGain', '3'), ('EILayerGain', '3'), ('IELayerGain', '3'), ('IILayerGain', '3'), 
                    ('EELayerGain', '4'), ('EILayerGain', '4'), ('IELayerGain', '4'), ('IILayerGain', '4'), 
                    ('EELayerGain', '5A'), ('EILayerGain', '5A'), ('IELayerGain', '5A'), ('IILayerGain', '5A'), 
                    ('EELayerGain', '5B'), ('EILayerGain', '5B'), ('IELayerGain', '5B'), ('IILayerGain', '5B'), 
                    ('EELayerGain', '6'), ('EILayerGain', '6'), ('IELayerGain', '6'), ('IILayerGain', '6')] 
    for p in updateParams:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})
    # good thal params for 100% cell density 
    updateParams2 = ['thalamoCorticalGain', 'intraThalamicGain', 'EbkgThalamicGain', 'IbkgThalamicGain', 'wmat']
    for p in updateParams2:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad2[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad2[p]})
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'
    return b

# ----------------------------------------------------------------------------------------------
# Custom pure click stim
# ----------------------------------------------------------------------------------------------
def custom_click(filename):
    params = specs.ODict()

    if not filename:
        filename = 'data/v34_batch25/trial_2142/trial_2142_cfg.json'

    # from prev 
    import json
    with open(filename, 'rb') as f:
        cfgLoad = json.load(f)['simConfig']
    cfgLoad2 = cfgLoad


    #### UNCOMMENT THIS FOR LOOPED STIMULUS INPUT:  
    # click_stimTimes = list(np.arange(2500, 11300, 624.5))
    # click_stimTimes_shorterSOA = list(np.arange(2500, 11300, 400))
    # click_stimTimes_longerSOA = list(np.arange(2500, 11300, 850))
    # params[('ICThalInput', 'startTime')] = [5000, click_stimTimes, click_stimTimes_shorterSOA, click_stimTimes_longerSOA]

    ## ICThalInput file 
    params[('ICThalInput', 'file')] = ['data/ICoutput/ICoutput_CF_3600_4400_wav_click_25ms_burst.mat', 'data/ICoutput/ICoutput_CF_10913_11713_wav_click_25ms_burst.mat' ]

    #### SET CONN AND STIM SEEDS #### 
    params[('seeds', 'conn')] = [0] #[0,1] #[0,1,4] 
    params[('seeds', 'stim')] = [0] #[0,1] #[0,1,4] 

    #### GROUPED PARAMS #### 
    groupedParams = [] 

    # --------------------------------------------------------
    # initial config
    initCfg = {} # set default options from prev sim
    
    initCfg['duration'] = 12000 #11500 
    initCfg['printPopAvgRates'] = [1500, 10000]
    initCfg['scaleDensity'] = 1.0 
    initCfg['recordStep'] = 0.05

    # # plotting and saving params
    # initCfg[('analysis', 'plotRaster','timeRange')] = initCfg['printPopAvgRates'] # MAY NEED TO BE 'plotting' rather than 'analysis' now? 
    # initCfg[('analysis', 'plotTraces', 'timeRange')] = initCfg['printPopAvgRates']
    # initCfg[('analysis', 'plotSpikeStats', 'timeRange')] = initCfg['printPopAvgRates']
    # initCfg[('analysis', 'plotLFP', 'timeRange')] = initCfg['printPopAvgRates']
    # initCfg[('analysis', 'plotCSD', 'timeRange')] = [1500, 1700]

    ## BBN STIMULUS FOR ICThalInput ## 
    initCfg['ICThalInput'] = {'file': 'data/ICoutput/ICoutput_CF_5256_6056_wav_click_25ms_burst.mat',
                            'startTime': 5000, #2500, 
                            'weightE': 0.25,
                            'weightI': 0.25, 
                            'probE': 0.12, 
                            'probI': 0.12,
                            'seed': 1}  # SHOULD THIS BE ZERO? 

    ### OPTION TO RECORD EEG / DIPOLE ###
    initCfg['recordDipole'] = False

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False
    
    # from prev - best of 50% cell density
    updateParams = ['EEGain', 'EIGain', 'IEGain', 'IIGain',
                    ('EICellTypeGain', 'PV'), ('EICellTypeGain', 'SOM'), ('EICellTypeGain', 'VIP'), ('EICellTypeGain', 'NGF'),
                    ('IECellTypeGain', 'PV'), ('IECellTypeGain', 'SOM'), ('IECellTypeGain', 'VIP'), ('IECellTypeGain', 'NGF'),
                    ('EILayerGain', '1'), ('IILayerGain', '1'),
                    ('EELayerGain', '2'), ('EILayerGain', '2'),  ('IELayerGain', '2'), ('IILayerGain', '2'), 
                    ('EELayerGain', '3'), ('EILayerGain', '3'), ('IELayerGain', '3'), ('IILayerGain', '3'), 
                    ('EELayerGain', '4'), ('EILayerGain', '4'), ('IELayerGain', '4'), ('IILayerGain', '4'), 
                    ('EELayerGain', '5A'), ('EILayerGain', '5A'), ('IELayerGain', '5A'), ('IILayerGain', '5A'), 
                    ('EELayerGain', '5B'), ('EILayerGain', '5B'), ('IELayerGain', '5B'), ('IILayerGain', '5B'), 
                    ('EELayerGain', '6'), ('EILayerGain', '6'), ('IELayerGain', '6'), ('IILayerGain', '6')] 

    for p in updateParams:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})

    # good thal params for 100% cell density 
    updateParams2 = ['thalamoCorticalGain', 'intraThalamicGain', 'EbkgThalamicGain', 'IbkgThalamicGain', 'wmat']

    for p in updateParams2:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad2[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad2[p]})


    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'

    return b


# ----------------------------------------------------------------------------------------------
# Custom pure tone stim
# ----------------------------------------------------------------------------------------------
def custom_tone(filename):
    params = specs.ODict()

    if not filename:
        filename = 'data/v34_batch25/trial_2142/trial_2142_cfg.json'

    # from prev 
    import json
    with open(filename, 'rb') as f:
        cfgLoad = json.load(f)['simConfig']
    cfgLoad2 = cfgLoad


    #### UNCOMMENT THIS FOR LOOPED STIMULUS INPUT:  
    pureTone_stimTimes = list(np.arange(2500, 11300, 624.5))
    pureTone_stimTimes_shorterSOA = list(np.arange(2500, 11300, 200))
    # pureTone_stimTimes_longerSOA = list(np.arange(2500, 11300, 850))
    params[('ICThalInput', 'startTime')] = [pureTone_stimTimes, pureTone_stimTimes_shorterSOA] #[5000, pureTone_stimTimes, pureTone_stimTimes_shorterSOA] #, pureTone_stimTimes_longerSOA]
    params[('ICThalInput', 'file')] = ['data/ICoutput/ICoutput_CF_300_700_wav_pure_tone_500Hz_25ms.mat', 'data/ICoutput/ICoutput_CF_5456_5856_wav_pure_tone_5656Hz_25ms.mat'] #['data/ICoutput/ICoutput_CF_300_700_wav_pure_tone_500Hz_25ms.mat', 'data/ICoutput/ICoutput_CF_2628_3028_wav_pure_tone_2828Hz_25ms.mat', 'data/ICoutput/ICoutput_CF_5456_5856_wav_pure_tone_5656Hz_25ms.mat', 'data/ICoutput/ICoutput_CF_11112_11512_wav_pure_tone_11312Hz_25ms.mat']

    #### SET CONN AND STIM SEEDS #### 
    params[('seeds', 'conn')] = [0] #[0,1] #[0,1,4] 
    params[('seeds', 'stim')] = [0] #[0,1] #[0,1,4] 

    #### GROUPED PARAMS #### 
    groupedParams = [] 

    # --------------------------------------------------------
    # initial config
    initCfg = {} # set default options from prev sim
    
    initCfg['duration'] = 12000 #11500 
    initCfg['printPopAvgRates'] = [1500, 10000]
    initCfg['scaleDensity'] = 1.0 
    initCfg['recordStep'] = 0.05


    # # plotting and saving params
    # initCfg[('analysis', 'plotRaster','timeRange')] = initCfg['printPopAvgRates'] # MAY NEED TO BE 'plotting' rather than 'analysis' now? 
    # initCfg[('analysis', 'plotTraces', 'timeRange')] = initCfg['printPopAvgRates']
    # initCfg[('analysis', 'plotSpikeStats', 'timeRange')] = initCfg['printPopAvgRates']
    # initCfg[('analysis', 'plotLFP', 'timeRange')] = initCfg['printPopAvgRates']
    # initCfg[('analysis', 'plotCSD', 'timeRange')] = [1500, 1700]

    ## BBN STIMULUS FOR ICThalInput ## 
    initCfg['ICThalInput'] = {'file': 'data/ICoutput/ICoutput_CF_1214_1614_wav_pure_tone_25ms.mat',
                            'startTime': 2500, 
                            'weightE': 0.25,
                            'weightI': 0.25, 
                            'probE': 0.12, 
                            'probI': 0.12,
                            'seed': 1}  # SHOULD THIS BE ZERO? 

    ### OPTION TO RECORD EEG / DIPOLE ###
    initCfg['recordDipole'] = False

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False
    
    # from prev - best of 50% cell density
    updateParams = ['EEGain', 'EIGain', 'IEGain', 'IIGain',
                    ('EICellTypeGain', 'PV'), ('EICellTypeGain', 'SOM'), ('EICellTypeGain', 'VIP'), ('EICellTypeGain', 'NGF'),
                    ('IECellTypeGain', 'PV'), ('IECellTypeGain', 'SOM'), ('IECellTypeGain', 'VIP'), ('IECellTypeGain', 'NGF'),
                    ('EILayerGain', '1'), ('IILayerGain', '1'),
                    ('EELayerGain', '2'), ('EILayerGain', '2'),  ('IELayerGain', '2'), ('IILayerGain', '2'), 
                    ('EELayerGain', '3'), ('EILayerGain', '3'), ('IELayerGain', '3'), ('IILayerGain', '3'), 
                    ('EELayerGain', '4'), ('EILayerGain', '4'), ('IELayerGain', '4'), ('IILayerGain', '4'), 
                    ('EELayerGain', '5A'), ('EILayerGain', '5A'), ('IELayerGain', '5A'), ('IILayerGain', '5A'), 
                    ('EELayerGain', '5B'), ('EILayerGain', '5B'), ('IELayerGain', '5B'), ('IILayerGain', '5B'), 
                    ('EELayerGain', '6'), ('EILayerGain', '6'), ('IELayerGain', '6'), ('IILayerGain', '6')] 

    for p in updateParams:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})

    # good thal params for 100% cell density 
    updateParams2 = ['thalamoCorticalGain', 'intraThalamicGain', 'EbkgThalamicGain', 'IbkgThalamicGain', 'wmat']

    for p in updateParams2:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad2[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad2[p]})


    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'

    return b

# ----------------------------------------------------------------------------------------------
# Custom
# ----------------------------------------------------------------------------------------------
def custom_stim(filename):
    params = specs.ODict()
    if not filename:
        filename = 'data/v34_batch25/trial_2142/trial_2142_cfg.json'
    # from prev 
    import json
    with open(filename, 'rb') as f:
        cfgLoad = json.load(f)['simConfig']
    cfgLoad2 = cfgLoad
    # conn gains 
    params[('NetStim1', 'weight')] = [1, 5, 10, 20] # 50 Hz
    params[('NetStim1', 'interval')] = [1000.0/50.0, 1000.0/100.0] # 50 Hz
    params[('NetStim1', 'noise')] = [0.5, 1.0] # 50 Hz
    groupedParams = [] #('ICThalInput', 'probE'), ('ICThalInput', 'probI')] #('IELayerGain', '1-3'), ('IELayerGain', '4'), ('IELayerGain', '5'), ('IELayerGain', '6')]
    # --------------------------------------------------------
    # initial config
    initCfg = {} # set default options from prev sim
    #initCfg['dt'] = 0.025 
    #initCfg['recordStep'] = 0.1 # 0.025 
    ## record Current as well
    # initCfg['use_fast_imem'] = True
    # initCfg['recordTraces'] = {'I_soma':{'sec':'soma','loc':0.5,'var':'i_membrane_'}} # 'V_soma':{'sec':'soma','loc':0.5,'var':'v'},  # 'sec':'soma','loc':0.5,
    initCfg['duration'] = 6000                                  # 11500 
    initCfg['printPopAvgRates'] = [1500, initCfg['duration']]   # [1500, 11500]
    initCfg['scaleDensity'] = 1.0                               # 0.25  # 0.075 
    initCfg['recordStep'] = 0.05
    initCfg['addNetStim'] = True
    initCfg[('NetStim1', 'pop')] = ['TC', 'TCM', 'HTC']
    initCfg[('NetStim1', 'ynorm')] = [0.0, 3.0]
    initCfg[('NetStim1', 'sec')] = 'soma'
    initCfg[('NetStim1', 'loc')] = 0.5
    initCfg[('NetStim1', 'start')] = 3000
    #initCfg[('NetStim1', 'interval')] = 1000.0/50.0 # 50 Hz
    #initCfg[('NetStim1', 'noise')] = 0.5 # 50 Hz
    initCfg[('NetStim1', 'number')] = 100 * 10 # enough spikes for 10 seconds
    # save LFP trace from first cell of each pop 
    #initCfg['saveLFPCells'] = [('IT5A', [0]), ('CT5A', [0]), ('SOM5A', [0]), ('PV5A', [0]), ('VIP5A', [0]), ('NGF5A', [0]), 
                                #('IT5B', [0]), ('PT5B', [0]), ('CT5B', [0]), ('SOM5B', [0]), ('PV5B', [0]), ('VIP5B', [0]), ('NGF5B', [0])]#, #[('IT3', [0]), ('SOM3', [0]), ('PV3', [0]), ('VIP3', [0]), ('NGF3', [0]),
                                #('ITP4', [0]), ('ITS4', [0]), ('SOM4', [0]), ('PV4', [0]), ('VIP4', [0]), ('NGF4', [0])]#,
                                # ('IT5A', [0]), ('CT5A', [0]), ('SOM5A', [0]), ('PV5A', [0]), ('VIP5A', [0]), ('NGF5A', [0]), 
                                # ('IT5B', [0]), ('PT5B', [0]), ('CT5B', [0]), ('SOM5B', [0]), ('PV5B', [0]), ('VIP5B', [0]), ('NGF5B', [0]), 
                                # ('IT6', [0]), ('CT6', [0]), ('SOM6', [0]), ('PV6', [0]), ('VIP6', [0]), ('NGF6', [0])] # [3, 'PYR', ('PV2', 5)] # True # Default is False ! 
                                # [('IT2', [0]), ('PV2', [0]), ('SOM2', [0]), ('VIP2', [0]), ('NGF2', [0])],
    # plotting and saving params
    initCfg[('analysis','plotRaster','timeRange')] = initCfg['printPopAvgRates']
    #initCfg[('analysis', 'plotTraces', 'timeRange')] = initCfg['printPopAvgRates']
    #initCfg[('analysis', 'plotSpikeStats', 'timeRange')] = initCfg['printPopAvgRates']
    #initCfg[('analysis', 'plotLFP', 'timeRange')] = initCfg['printPopAvgRates']
    #initCfg[('analysis', 'plotCSD', 'timeRange')] = [1500, 1700]
    # changed directly in cfg.py    
    initCfg[('analysis', 'plotCSD')] = {'spacing_um': 100, 'timeRange': initCfg['printPopAvgRates'], 'layer_lines': 0, 'saveFig': 1, 'showFig': 0} # 'overlay': 'LFP', # 'LFP_overlay'
    initCfg['recordLFP'] = [[100, y, 100] for y in range(0, 2000, 100)]
    # save LFP traces from populations 
    #initCfg['saveLFPPops'] = ['IT2', 'CT6'] #['PV2', 'SOM2', 'VIP2', 'NGF2', 
            #'PV3', 'SOM3', 'VIP3', 'NGF3',
            #'PV4', 'SOM4', 'VIP4', 'NGF4',
            #'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',
            #'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',
            #'PV6', 'SOM6', 'VIP6', 'NGF6'] #['IT2', 'PT5B'] #['IT6', 'CT6']	#['IT5B', 'CT5B']	#['IT3', 'ITP4', 'ITS4']	#['NGF1', 'IT5A', 'CT5A'] #['IT2', 'PT5B']
    #initCfg[('analysis', 'plotLFP')] = {'pop': 'PT5B', 'includeAxon': False, 'figSize': (6,10), 'timeRange': [100,3000], 'saveFig': True}
    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False
    # from prev - best of 50% cell density
    updateParams = ['EEGain', 'EIGain', 'IEGain', 'IIGain',
                    ('EICellTypeGain', 'PV'), ('EICellTypeGain', 'SOM'), ('EICellTypeGain', 'VIP'), ('EICellTypeGain', 'NGF'),
                    ('IECellTypeGain', 'PV'), ('IECellTypeGain', 'SOM'), ('IECellTypeGain', 'VIP'), ('IECellTypeGain', 'NGF'),
                    ('EILayerGain', '1'), ('IILayerGain', '1'),
                    ('EELayerGain', '2'), ('EILayerGain', '2'),  ('IELayerGain', '2'), ('IILayerGain', '2'), 
                    ('EELayerGain', '3'), ('EILayerGain', '3'), ('IELayerGain', '3'), ('IILayerGain', '3'), 
                    ('EELayerGain', '4'), ('EILayerGain', '4'), ('IELayerGain', '4'), ('IILayerGain', '4'), 
                    ('EELayerGain', '5A'), ('EILayerGain', '5A'), ('IELayerGain', '5A'), ('IILayerGain', '5A'), 
                    ('EELayerGain', '5B'), ('EILayerGain', '5B'), ('IELayerGain', '5B'), ('IILayerGain', '5B'), 
                    ('EELayerGain', '6'), ('EILayerGain', '6'), ('IELayerGain', '6'), ('IILayerGain', '6')] 
    for p in updateParams:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})
    # good thal params for 100% cell density 
    updateParams2 = ['thalamoCorticalGain', 'intraThalamicGain', 'EbkgThalamicGain', 'IbkgThalamicGain', 'wmat']
    for p in updateParams2:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad2[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad2[p]})
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'
    return b
  
# ----------------------------------------------------------------------------------------------
# Optuna optimization for ERP 
# ----------------------------------------------------------------------------------------------
def optunaERP (): 
    # --------------------------------------------------------
    # parameters
    params = specs.ODict()
    # these params control IC -> Thal
    params['ICThalweightECore'] = [0.75, 1.25]
    params['ICThalweightICore'] = [0.1875, 0.3125]
    params['ICThalprobECore'] = [0.1425, 0.2375]
    params['ICThalprobICore'] = [0.09, 0.15]
    params['ICThalMatrixCoreFactor'] = [0.075, 0.125]
    # these params added from Christoph Metzner branch
    params['thalL4PV'] = [0.1875, 0.3125]
    params['thalL4SOM'] = [0.1875, 0.3125]
    params['thalL4E'] = [1.5, 2.5]
    # ADD: parameters to vary 
    # groupedParams = []
    # --------------------------------------------------------
    # initial config
    initCfg = {}
    initCfg['duration'] = 5000 
    initCfg['printPopAvgRates'] = [300, 4000]
    initCfg['scaleDensity'] = 1.0 
    initCfg['recordStep'] = 0.05
    # SET SEEDS FOR CONN AND STIM 
    initCfg[('seeds', 'conn')] = 0    
    # --------------------------------------------------------
    # fitness function
    d = pickle.load(open('opt/2-rb023024011@os.mat_20kHz_avgERP.pkl','rb'))
    ttavgERPNHP = d['ttavg']
    avgCSDNHP = d['avgCSD'] # s2, g, i1 channels for primary CSD sinks are at indices 10, 14, 19
    fitnessFuncArgs = {}
    fitnessFuncArgs['maxFitness'] = 1.0
    groupedParams = []    
    def fitnessFunc(simData, **kwargs):
        print('fitness func')
        from csd import getCSDa1dat as getCSD
        from scipy.stats import pearsonr
        from erp import getAvgERP
        def ms2index (ms, sampr): return int(sampr*ms/1e3)
        LFP = simData['LFP']
        LFP = np.array(LFP)
        CSD = getCSD(LFP, 1e3/0.05)
        CSD.shape # (18, 220000)
        lchan = [4, 10, 15]
        lnhpchan = [11-1, 15-1, 20-1]
        bbnT = np.arange(3000, 4000, 300)
        dt = 0.05
        sampr = 1e3/dt
        bbnTrigIdx = [ms2index(x,sampr) for x in bbnT]
        ttERP,avgERP = getAvgERP(CSD, sampr, bbnTrigIdx, 0, 150)
        fitness = -pearsonr(avgCSDNHP[lnhpchan[1],:],avgERP[lchan[1]])[0]
        print('fitness is', fitness)
        return fitness
    # --------------------------------------------------------
    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg)#, groupedParams=groupedParams)
    b.method = 'optuna'    
    b.optimCfg = {
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'maxFitness': fitnessFuncArgs['maxFitness'],
        'maxiters':     100,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
        'maxtime':      None,    #    Maximum time allowed, in seconds
        'maxiter_wait': 500,
        'time_sleep': 120,
        'popsize': 1  # unused - run with mpi 
    }
    """
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
        'max_generations': 100,
        'time_sleep': 150, # 2.5min wait this time before checking again if sim is completed (for each generation)
        'maxiter_wait': 5, # max number of times to check if sim is completed (for each generation)
        'defaultFitness': 1.0, # set fitness value in case simulation time is over
        'scancelUser': 'samnemo_gmail_com'
    }  
    """
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
        'scancelUser': 'samnemo_gmail_com'
    }

    return b

# ----------------------------------------------------------------------------------------------
# Adaptive Stochastic Descent (ASD)
# ----------------------------------------------------------------------------------------------
def asdRates():

    # --------------------------------------------------------
    # parameters
    params = specs.ODict()

    x0 = [[1.4338777102469733, 1.1625828604622033, 1.091037051695174, 1.8712819675755956, 0.7397134465049761, 1.367569320349433, 1.418339439423966, 0.6274719645228012, 0.5675561437477121, 1.5174286644853214, 1.6851404284735372, 1.063075099977045, 0.673804518651403],
    [1.4117825668705553, 1.4562645192525767, 0.6966421717946888, 1.1564048776911902, 0.5082691576672945, 1.8650994583365461, 0.5247660780373347, 1.3887063888108127, 0.8359523062412009, 0.786403002769916, 1.0872681212597493, 1.9355702398668257, 0.8456162169403141],
    [1.4796339232563818, 1.2494919865726666, 1.2106074885592537, 0.5914377334878493, 0.7956691184253843, 1.1044833499655324, 1.8970275010959088, 1.2806598565853817, 1.0339389242169903, 1.2449536297089994, 1.653463860326919, 0.5816973165681442, 1.8408576413375228],
    [1.3154950966436703, 1.0095763680475387, 1.3046938357412072, 1.337690869825955, 1.3419352351670506, 2.0, 1.806386376748424, 1.785015289487499, 1.3006272106913037, 1.6797508518217605, 1.5625342091955938, 0.9733859948789619, 0.8423443321780072],
    [1.4081465013179777, 0.6909751558458218, 1.476256983214676, 1.4388900372032694, 0.5, 1.4292511768559795, 0.6980418301090989, 1.1884408015079058, 1.8830229460800467, 1.1514878860870101, 0.9636536753602729, 1.283310375368901, 1.2234380160367202]]

    # bkg inputs
    params['EEGain'] = [0.5, 2.0, [x[0] for x in x0]]
    params['EIGain'] = [0.5, 2.0, [x[1] for x in x0]]

    params[('IELayerGain', '1-3')] = [0.5, 2.0, [x[2] for x in x0]]
    params[('IELayerGain', '4')] = [0.5, 2.0, [x[3] for x in x0]]
    params[('IELayerGain', '5')] = [0.5, 2.0, [x[4] for x in x0]]
    params[('IELayerGain', '6')] = [0.5, 2.0, [x[5] for x in x0]]

    params[('IILayerGain', '1-3')] = [0.5, 2.0, [x[6] for x in x0]]
    params[('IILayerGain', '4')] = [0.5, 2.0, [x[7] for x in x0]]
    params[('IILayerGain', '5')] = [0.5, 2.0, [x[8] for x in x0]]
    params[('IILayerGain', '6')] = [0.5, 2.0, [x[9] for x in x0]]
    
    params['thalamoCorticalGain'] = [0.5, 2.0, [x[10] for x in x0]]
    params['intraThalamicGain'] = [0.5, 2.0, [x[11] for x in x0]]
    params['corticoThalamicGain'] = [0.5, 2.0, [x[12] for x in x0]]


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
        'maxiters':     300,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
        'maxtime':      360000,    #    Maximum time allowed, in seconds
        'abstol':       1e-6,    #    Minimum absolute change in objective function
        'reltol':       1e-3,    #    Minimum relative change in objective function
        #'stalliters':   10*len(params)*len(params),  #    Number of iterations over which to calculate TolFun (n = number of parameters)
        #'stoppingfunc': None,    #    External method that can be used to stop the calculation from the outside.
        #'randseed':     None,    #    The random seed to use
        'verbose':      2,       #    How much information to print during the run
        #'label':        None    #    A label to use to annotate the output
        'time_sleep': 60, # 1min wait this time before checking again if sim is completed (for each generation)
        'maxiter_wait': 12,  # max number of times to check if sim is completed (for each generation)
        'popsize': 5
    }

    return b


# ----------------------------------------------------------------------------------------------
# Adaptive Stochastic Descent (ASD)
# ----------------------------------------------------------------------------------------------
def optunaRates():

    # --------------------------------------------------------
    # parameters
    params = specs.ODict()

    # bkg inputs
    params['EEGain'] = [0.1, 1.0]

    params[('EILayerGain', '1-3')] = [0.1, 3.0]
    params[('EILayerGain', '4')] = [0.1, 3.0]
    params[('EILayerGain', '5')] = [0.1, 3.0]
    params[('EILayerGain', '6')] = [0.1, 3.0]

    params[('IELayerGain', '1-3')] = [0.1, 3.0]
    params[('IELayerGain', '4')] = [0.1, 3.0]
    params[('IELayerGain', '5')] = [0.1, 3.0]
    params[('IELayerGain', '6')] = [0.1, 3.0]

    params[('IILayerGain', '1-3')] = [0.1, 3.0]
    params[('IILayerGain', '4')] = [0.1, 3.0]
    params[('IILayerGain', '5')] = [0.1, 3.0]
    params[('IILayerGain', '6')] = [0.1, 3.0]

    params[('EICellTypeGain', 'PV')] = [max(cfgLoad['EICellTypeGain']['PV']-rangeV2, minV), min(cfgLoad['EICellTypeGain']['PV']+rangeV2, maxV)]
    params[('EICellTypeGain', 'SOM')] = [max(cfgLoad['EICellTypeGain']['SOM']-rangeV2, minV), min(cfgLoad['EICellTypeGain']['SOM']+rangeV2, maxV)]
    params[('EICellTypeGain', 'VIP')] = [max(cfgLoad['EICellTypeGain']['VIP']-rangeV2, minV), min(cfgLoad['EICellTypeGain']['VIP']+rangeV2, maxV)]
    params[('EICellTypeGain', 'NGF')] = [max(cfgLoad['EICellTypeGain']['NGF']-rangeV2, minV), min(cfgLoad['EICellTypeGain']['NGF']+rangeV2, maxV)]

    # params['thalamoCorticalGain'] = [0.25, 2.0]
    # params['intraThalamicGain'] = [0.25, 2.0]
    # params['corticoThalamicGain'] = [0.25, 2.0]


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

    b.method = 'optuna'

    b.optimCfg = {
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'maxFitness': fitnessFuncArgs['maxFitness'],
        'maxiters':     1e6,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
        'maxtime':      None,    #    Maximum time allowed, in seconds
        'maxiter_wait': 16,
        'time_sleep': 60,
        'popsize': 1  # unused - run with mpi 
    }

    return b



# ----------------------------------------------------------------------------------------------
# Optuna optimization
# ----------------------------------------------------------------------------------------------
def optunaRatesLayers():

    '''
    # from prev
    import json
    with open('data/v32_batch14/trial_981/trial_981_cfg.json', 'rb') as f:
        cfgLoad = json.load(f)['simConfig']
    '''

    # best from grid search
    import json
    # with open('data/v32_batch4/trial_15057/trial_15057_cfg.json', 'rb') as f:
    #     cfgLoad = json.load(f)['simConfig']

    # good thal params for 100% cell density 
    with open('data/v34_batch5/v34_batch5_0_2_0_0_2_0_2_2_cfg.json', 'rb') as f:
        cfgLoad2 = json.load(f)['simConfig']

    cfgLoad = cfgLoad2

    # --------------------------------------------------------
    # parameters
    params = specs.ODict()

    rangeV = 0.25
    rangeV2 = 1.0
    minV = 0.1
    maxV = 4.0


    # params based on v32_batch8 grid search best solutions; plus added L2 and L3 IE specific gains since those were problematic layers
    params['EEGain'] = [0.1, 0.5] 
    params['EIGain'] = [1.3, 1.7] 
    params['IEGain'] = [0.8, 1.7] 
    params['IIGain'] = [0.3, 0.7] 
    params[('EICellTypeGain', 'PV')] = [0.8, 1.7] 
    params[('EICellTypeGain', 'SOM')] = [0.3, 0.7] 
    params[('EICellTypeGain', 'VIP')] = [1.3, 1.7] 
    params[('EICellTypeGain', 'NGF')] = [1.3, 1.7] 

    params[('IELayerGain', '2')] = [max(cfgLoad2['IELayerGain']['2']-rangeV, minV), min(cfgLoad2['IELayerGain']['2']+rangeV, maxV)]
    params[('IELayerGain', '3')] = [max(cfgLoad2['IELayerGain']['3']-rangeV, minV), min(cfgLoad2['IELayerGain']['3']+rangeV, maxV)]
    params[('IELayerGain', '4')] = [max(cfgLoad2['IELayerGain']['4']-rangeV, minV), min(cfgLoad2['IELayerGain']['4']+rangeV, maxV)]
    params[('IELayerGain', '5A')] = [max(cfgLoad2['IELayerGain']['5A']-rangeV, minV), min(cfgLoad2['IELayerGain']['5A']+rangeV, maxV)]
    params[('IELayerGain', '5B')] = [max(cfgLoad2['IELayerGain']['5B']-rangeV, minV), min(cfgLoad2['IELayerGain']['5B']+rangeV, maxV)]

    
    '''

    params[('EICellTypeGain', 'PV')] = [max(cfgLoad['EICellTypeGain']['PV']-rangeV2, minV), min(cfgLoad['EICellTypeGain']['PV']+rangeV2, maxV)]
    params[('EICellTypeGain', 'SOM')] = [max(cfgLoad['EICellTypeGain']['SOM']-rangeV2, minV), min(cfgLoad['EICellTypeGain']['SOM']+rangeV2, maxV)]
    params[('EICellTypeGain', 'VIP')] = [max(cfgLoad['EICellTypeGain']['VIP']-rangeV2, minV), min(cfgLoad['EICellTypeGain']['VIP']+rangeV2, maxV)]
    params[('EICellTypeGain', 'NGF')] = [max(cfgLoad['EICellTypeGain']['NGF']-rangeV2, minV), min(cfgLoad['EICellTypeGain']['NGF']+rangeV2, maxV)]

    params[('IECellTypeGain', 'PV')] = [max(cfgLoad['IECellTypeGain']['PV']-rangeV, minV), min(cfgLoad['IECellTypeGain']['PV']+rangeV, maxV)]
    params[('IECellTypeGain', 'SOM')] = [max(cfgLoad['IECellTypeGain']['SOM']-rangeV, minV), min(cfgLoad['IECellTypeGain']['SOM']+rangeV, maxV)]
    params[('IECellTypeGain', 'VIP')] = [max(cfgLoad['IECellTypeGain']['VIP']-rangeV, minV), min(cfgLoad['IECellTypeGain']['VIP']+rangeV, maxV)]
    params[('IECellTypeGain', 'NGF')] = [max(cfgLoad['IECellTypeGain']['NGF']-rangeV, minV), min(cfgLoad['IECellTypeGain']['NGF']+rangeV, maxV)]

    params[('EILayerGain', '1')] = [minV, maxV] #(cfgLoad['EILayerGain']['1']-rangeV2, minV), min(cfgLoad['EILayerGain']['1']+rangeV, maxV)]
    params[('IILayerGain', '1')] = [minV, maxV] #(cfgLoad['IILayerGain']['1']-rangeV2, minV), min(cfgLoad['IILayerGain']['1']+rangeV, maxV)]

    params[('EELayerGain', '2')] = [max(cfgLoad['EELayerGain']['2']-rangeV, minV), min(cfgLoad['EELayerGain']['2']+rangeV, maxV)]
    params[('EILayerGain', '2')] = [max(cfgLoad['EILayerGain']['2']-rangeV, minV), min(cfgLoad['EILayerGain']['2']+rangeV, maxV)]
    params[('IELayerGain', '2')] = [max(cfgLoad['IELayerGain']['2']-rangeV, minV), min(cfgLoad['IELayerGain']['2']+rangeV, maxV)]
    params[('IILayerGain', '2')] = [max(cfgLoad['IILayerGain']['2']-rangeV, minV), min(cfgLoad['IILayerGain']['2']+rangeV, maxV)]

    params[('EELayerGain', '3')] = [max(cfgLoad['EELayerGain']['3']-rangeV, minV), min(cfgLoad['EELayerGain']['3']+rangeV, maxV)]
    params[('EILayerGain', '3')] = [max(cfgLoad['EILayerGain']['3']-rangeV, minV), min(cfgLoad['EILayerGain']['3']+rangeV, maxV)]
    params[('IELayerGain', '3')] = [max(cfgLoad['IELayerGain']['3']-rangeV, minV), min(cfgLoad['IELayerGain']['3']+rangeV, maxV)]
    params[('IILayerGain', '3')] = [max(cfgLoad['IILayerGain']['3']-rangeV, minV), min(cfgLoad['IILayerGain']['3']+rangeV, maxV)]

    params[('EELayerGain', '4')] = [max(cfgLoad['EELayerGain']['4']-rangeV, minV), min(cfgLoad['EELayerGain']['4']+rangeV, maxV)]
    params[('EILayerGain', '4')] = [max(cfgLoad['EILayerGain']['4']-rangeV, minV), min(cfgLoad['EILayerGain']['4']+rangeV, maxV)]
    params[('IELayerGain', '4')] = [max(cfgLoad['IELayerGain']['4']-rangeV, minV), min(cfgLoad['IELayerGain']['4']+rangeV, maxV)]
    params[('IILayerGain', '4')] = [max(cfgLoad['IILayerGain']['4']-rangeV, minV), min(cfgLoad['IILayerGain']['4']+rangeV, maxV)]

    params[('EELayerGain', '5A')] = [minV, maxV] # [max(cfgLoad['EELayerGain']['5A']-rangeV2, minV), min(cfgLoad['EELayerGain']['5A']+rangeV, maxV)]
    params[('EILayerGain', '5A')] = [minV, maxV] # [max(cfgLoad['EILayerGain']['5A']-rangeV2, minV), min(cfgLoad['EILayerGain']['5A']+rangeV, maxV)]
    params[('IELayerGain', '5A')] = [minV, maxV] # [max(cfgLoad['IELayerGain']['5A']-rangeV2, minV), min(cfgLoad['IELayerGain']['5A']+rangeV, maxV)]
    params[('IILayerGain', '5A')] = [minV, maxV] # [max(cfgLoad['IILayerGain']['5A']-rangeV2, minV), min(cfgLoad['IILayerGain']['5A']+rangeV, maxV)]

    params[('EELayerGain', '5B')] = [minV, maxV] # [max(cfgLoad['EELayerGain']['5B']-rangeV2, minV), min(cfgLoad['EELayerGain']['5B']+rangeV, maxV)]
    params[('EILayerGain', '5B')] = [minV, maxV] # [max(cfgLoad['EILayerGain']['5B']-rangeV2, minV), min(cfgLoad['EILayerGain']['5B']+rangeV, maxV)]
    params[('IELayerGain', '5B')] = [minV, maxV] # [max(cfgLoad['IELayerGain']['5B']-rangeV2, minV), min(cfgLoad['IELayerGain']['5B']+rangeV, maxV)]
    params[('IILayerGain', '5B')] = [minV, maxV] # [max(cfgLoad['IILayerGain']['5B']-rangeV2, minV), min(cfgLoad['IILayerGain']['5B']+rangeV, maxV)]

    params[('EELayerGain', '6')] = [minV, maxV] # [max(cfgLoad['EELayerGain']['6']-rangeV2, minV), min(cfgLoad['EELayerGain']['6']+rangeV, maxV)]
    params[('EILayerGain', '6')] = [minV, maxV] # [max(cfgLoad['EILayerGain']['6']-rangeV2, minV), min(cfgLoad['EILayerGain']['6']+rangeV, maxV)]
    params[('IELayerGain', '6')] = [minV, maxV] # [max(cfgLoad['IELayerGain']['6']-rangeV2, minV), min(cfgLoad['IELayerGain']['6']+rangeV, maxV)]
    params[('IILayerGain', '6')] = [minV, maxV] # [max(cfgLoad['IILayerGain']['6']-rangeV2, minV), min(cfgLoad['IILayerGain']['6']+rangeV, maxV)]
    '''

    groupedParams = []

    # --------------------------------------------------------
    # initial config
    initCfg = {}
    initCfg = {}
    initCfg['duration'] = 2500
    initCfg['printPopAvgRates'] = [[1500, 1750], [1750, 2000], [2000, 2250], [2250, 2500]]
    initCfg['dt'] = 0.05

    initCfg['scaleDensity'] = 1.0

    # plotting and saving params
    initCfg[('analysis','plotRaster','markerSize')] = 10

    initCfg[('analysis','plotRaster','timeRange')] = [1500, 2500]
    initCfg[('analysis', 'plotTraces', 'timeRange')] = [1500, 2500]
    initCfg[('analysis', 'plotTraces', 'oneFigPer')] = 'trace'
    initCfg['recordLFP'] = None
    initCfg[('analysis', 'plotLFP')] = False

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False
    
    '''
    initCfg['EEGain'] = 1.0 	
    initCfg['EIGain'] = 1.0 	
    initCfg['IEGain'] = 1.0 	
    initCfg['IIGain'] = 1.0 	

    initCfg.update({'thalamoCorticalGain': cfgLoad['thalamoCorticalGain'],
                    'intraThalamicGain': cfgLoad['intraThalamicGain'],
                    'EbkgThalamicGain': cfgLoad['EbkgThalamicGain'],
                    'IbkgThalamicGain': cfgLoad['IbkgThalamicGain']})

    print(initCfg)
    '''

    # from prev - best of 50% cell density
    updateParams = ['EEGain', 'EIGain', 'IEGain', 'IIGain',
                    ('EICellTypeGain', 'PV'), ('EICellTypeGain', 'SOM'), ('EICellTypeGain', 'VIP'), ('EICellTypeGain', 'NGF'),
                    ('IECellTypeGain', 'PV'), ('IECellTypeGain', 'SOM'), ('IECellTypeGain', 'VIP'), ('IECellTypeGain', 'NGF'),
                    ('EILayerGain', '1'), ('IILayerGain', '1'),
                    ('EELayerGain', '2'), ('EILayerGain', '2'),  ('IELayerGain', '2'), ('IILayerGain', '2'), 
                    ('EELayerGain', '3'), ('EILayerGain', '3'), ('IELayerGain', '3'), ('IILayerGain', '3'), 
                    ('EELayerGain', '4'), ('EILayerGain', '4'), ('IELayerGain', '4'), ('IILayerGain', '4'), 
                    ('EELayerGain', '5A'), ('EILayerGain', '5A'), ('IELayerGain', '5A'), ('IILayerGain', '5A'), 
                    ('EELayerGain', '5B'), ('EILayerGain', '5B'), ('IELayerGain', '5B'), ('IILayerGain', '5B'), 
                    ('EELayerGain', '6'), ('EILayerGain', '6'), ('IELayerGain', '6'), ('IILayerGain', '6')] 

    for p in updateParams:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})

    # good thal params for 100% cell density 
    updateParams2 = ['thalamoCorticalGain', 'intraThalamicGain', 'EbkgThalamicGain', 'IbkgThalamicGain']

    for p in updateParams2:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad2[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad2[p]})


    # --------------------------------------------------------
    # fitness function
    fitnessFuncArgs = {}
    pops = {}
    
    ## Exc pops
    Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6', 'TC', 'TCM', 'HTC']  # all layers + thal + IC
    #Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'TC', 'TCM', 'HTC']  # all layers + thal + IC

    #Etune = {'target': 5, 'width': 20, 'min': 0.05}
    Etune = {'target': 5, 'width': 5, 'min': 0.5}
    
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
    # Ipops = [#'NGF1',  
    #         'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
    #         'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
    #         'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
    #         #'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A 
    #         #'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
    #         #'PV6', 'SOM6', 'VIP6', 'NGF6',  # L6
    #         'IRE', 'IREM', 'TI']  # Thal 

    #Itune = {'target': 10, 'width': 30, 'min': 0.05}
    Itune = {'target': 10, 'width': 15, 'min': 0.5}

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
                if simData['popRates'][k]['%d_%d'%(trange[0], trange[1])] >= v['min'] else maxFitness for k, v in pops.items()])
        
        popFitness = np.mean(np.array(popFitnessAll), axis=0)
        
        fitness = np.mean(popFitness)

        popInfo = '; '.join(['%s rate=%.1f fit=%1.f' % (p, np.mean(list(simData['popRates'][p].values())), popFitness[i]) for i,p in enumerate(pops)])
        print('  ' + popInfo)

        return fitness
        
    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)

    b.method = 'optuna'

    b.optimCfg = {
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'maxFitness': fitnessFuncArgs['maxFitness'],
        'maxiters':     1e6,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
        'maxtime':      None,    #    Maximum time allowed, in seconds
        'maxiter_wait': 45,
        'time_sleep': 120,
        'popsize': 1  # unused - run with mpi 
    }

    return b



# ----------------------------------------------------------------------------------------------
# Optuna optimization
# ----------------------------------------------------------------------------------------------
def optunaRatesLayersThalL4():

    # # from prev
    # import json
    # with open('data/v32_batch4/trial_15057/trial_15057_cfg.json', 'rb') as f:
    #     cfgLoad = json.load(f)['simConfig']


    # --------------------------------------------------------
    # parameters
    params = specs.ODict()

    rangeV = 1.0
    rangeV2 = 1.0
    minV = 0.1
    maxV = 4.0

    params[('EICellTypeGain', 'PV')] = [minV, maxV] # [max(cfgLoad['EICellTypeGain']['PV']-rangeV, minV), min(cfgLoad['EICellTypeGain']['PV']+rangeV, maxV)]
    params[('EICellTypeGain', 'SOM')] = [minV, maxV] # [max(cfgLoad['EICellTypeGain']['SOM']-rangeV, minV), min(cfgLoad['EICellTypeGain']['SOM']+rangeV, maxV)]
    params[('EICellTypeGain', 'VIP')] = [minV, maxV] # [max(cfgLoad['EICellTypeGain']['VIP']-rangeV, minV), min(cfgLoad['EICellTypeGain']['VIP']+rangeV, maxV)]
    params[('EICellTypeGain', 'NGF')] = [minV, maxV] # [max(cfgLoad['EICellTypeGain']['NGF']-rangeV, minV), min(cfgLoad['EICellTypeGain']['NGF']+rangeV, maxV)]

    params[('IECellTypeGain', 'PV')] = [minV, maxV] # [max(cfgLoad['IECellTypeGain']['PV']-rangeV, minV), min(cfgLoad['IECellTypeGain']['PV']+rangeV, maxV)]
    params[('IECellTypeGain', 'SOM')] = [minV, maxV] # [max(cfgLoad['IECellTypeGain']['SOM']-rangeV, minV), min(cfgLoad['IECellTypeGain']['SOM']+rangeV, maxV)]
    params[('IECellTypeGain', 'VIP')] = [minV, maxV] # [max(cfgLoad['IECellTypeGain']['VIP']-rangeV, minV), min(cfgLoad['IECellTypeGain']['VIP']+rangeV, maxV)]
    params[('IECellTypeGain', 'NGF')] = [minV, maxV] # [max(cfgLoad['IECellTypeGain']['NGF']-rangeV, minV), min(cfgLoad['IECellTypeGain']['NGF']+rangeV, maxV)]

    # params[('EILayerGain', '1')] = [max(cfgLoad['EILayerGain']['1']-rangeV2, minV), min(cfgLoad['EILayerGain']['1']+rangeV, maxV)]
    # params[('IILayerGain', '1')] = [max(cfgLoad['IILayerGain']['1']-rangeV2, minV), min(cfgLoad['IILayerGain']['1']+rangeV, maxV)]

    # params[('EELayerGain', '2')] = [max(cfgLoad['EELayerGain']['2']-rangeV2, minV), min(cfgLoad['EELayerGain']['2']+rangeV, maxV)]
    # params[('EILayerGain', '2')] = [max(cfgLoad['EILayerGain']['2']-rangeV2, minV), min(cfgLoad['EILayerGain']['2']+rangeV, maxV)]
    # params[('IELayerGain', '2')] = [max(cfgLoad['IELayerGain']['2']-rangeV2, minV), min(cfgLoad['IELayerGain']['2']+rangeV, maxV)]
    # params[('IILayerGain', '2')] = [max(cfgLoad['IILayerGain']['2']-rangeV2, minV), min(cfgLoad['IILayerGain']['2']+rangeV, maxV)]

    # params[('EELayerGain', '3')] = [max(cfgLoad['EELayerGain']['3']-rangeV2, minV), min(cfgLoad['EELayerGain']['3']+rangeV, maxV)]
    # params[('EILayerGain', '3')] = [max(cfgLoad['EILayerGain']['3']-rangeV2, minV), min(cfgLoad['EILayerGain']['3']+rangeV, maxV)]
    # params[('IELayerGain', '3')] = [max(cfgLoad['IELayerGain']['3']-rangeV2, minV), min(cfgLoad['IELayerGain']['3']+rangeV, maxV)]
    # params[('IILayerGain', '3')] = [max(cfgLoad['IILayerGain']['3']-rangeV2, minV), min(cfgLoad['IILayerGain']['3']+rangeV, maxV)]

    params[('EELayerGain', '4')] = [minV, maxV] #[max(cfgLoad['EELayerGain']['4']-rangeV2, minV), min(cfgLoad['EELayerGain']['4']+rangeV, maxV)]
    params[('EILayerGain', '4')] = [minV, maxV] #[max(cfgLoad['EILayerGain']['4']-rangeV2, minV), min(cfgLoad['EILayerGain']['4']+rangeV, maxV)]
    params[('IELayerGain', '4')] = [minV, maxV] #[max(cfgLoad['IELayerGain']['4']-rangeV2, minV), min(cfgLoad['IELayerGain']['4']+rangeV, maxV)]
    params[('IILayerGain', '4')] = [minV, maxV] #[max(cfgLoad['IILayerGain']['4']-rangeV2, minV), min(cfgLoad['IILayerGain']['4']+rangeV, maxV)]

    params['thalamoCorticalGain'] = [minV, maxV]
    params['intraThalamicGain'] = [minV, maxV]
    params['EbkgThalamicGain'] = [minV, maxV]
    params['IbkgThalamicGain'] = [minV, maxV]

    # params[('EELayerGain', '5A')] = [max(cfgLoad['EELayerGain']['5A']-rangeV2, minV), min(cfgLoad['EELayerGain']['5A']+rangeV, maxV)]
    # params[('EILayerGain', '5A')] = [max(cfgLoad['EILayerGain']['5A']-rangeV2, minV), min(cfgLoad['EILayerGain']['5A']+rangeV, maxV)]
    # params[('IELayerGain', '5A')] = [max(cfgLoad['IELayerGain']['5A']-rangeV2, minV), min(cfgLoad['IELayerGain']['5A']+rangeV, maxV)]
    # params[('IILayerGain', '5A')] = [max(cfgLoad['IILayerGain']['5A']-rangeV2, minV), min(cfgLoad['IILayerGain']['5A']+rangeV, maxV)]

    # params[('EELayerGain', '5B')] = [max(cfgLoad['EELayerGain']['5B']-rangeV2, minV), min(cfgLoad['EELayerGain']['5B']+rangeV, maxV)]
    # params[('EILayerGain', '5B')] = [max(cfgLoad['EILayerGain']['5B']-rangeV2, minV), min(cfgLoad['EILayerGain']['5B']+rangeV, maxV)]
    # params[('IELayerGain', '5B')] = [max(cfgLoad['IELayerGain']['5B']-rangeV2, minV), min(cfgLoad['IELayerGain']['5B']+rangeV, maxV)]
    # params[('IILayerGain', '5B')] = [max(cfgLoad['IILayerGain']['5B']-rangeV2, minV), min(cfgLoad['IILayerGain']['5B']+rangeV, maxV)]

    # params[('EELayerGain', '6')] = [max(cfgLoad['EELayerGain']['6']-rangeV2, minV), min(cfgLoad['EELayerGain']['6']+rangeV, maxV)]
    # params[('EILayerGain', '6')] = [max(cfgLoad['EILayerGain']['6']-rangeV2, minV), min(cfgLoad['EILayerGain']['6']+rangeV, maxV)]
    # params[('IELayerGain', '6')] = [max(cfgLoad['IELayerGain']['6']-rangeV2, minV), min(cfgLoad['IELayerGain']['6']+rangeV, maxV)]
    # params[('IILayerGain', '6')] = [max(cfgLoad['IILayerGain']['6']-rangeV2, minV), min(cfgLoad['IILayerGain']['6']+rangeV, maxV)]


    groupedParams = []

    # --------------------------------------------------------
    # initial config
    initCfg = {}
    initCfg = {}
    initCfg['duration'] = 2500
    initCfg['printPopAvgRates'] = [[1500, 1750], [1750, 2000], [2000, 2250], [2250, 2500]]
    initCfg['dt'] = 0.05

    initCfg['scaleDensity'] = 1.0

    # plotting and saving params
    initCfg[('analysis','plotRaster','markerSize')] = 10

    initCfg[('analysis','plotRaster','timeRange')] = [1500, 2500]
    initCfg[('analysis', 'plotTraces', 'timeRange')] = [1500, 2500]
    initCfg[('analysis', 'plotTraces', 'oneFigPer')] = 'trace'
    initCfg['recordLFP'] = None
    initCfg[('analysis', 'plotLFP')] = False

    initCfg['EEGain'] = 1.0 	
    initCfg['EIGain'] = 1.0 	
    initCfg['IEGain'] = 1.0 	
    initCfg['IIGain'] = 1.0 	

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False
    
    # initCfg.update({'thalamoCorticalGain': cfgLoad['thalamoCorticalGain'],
    #                 'intraThalamicGain': cfgLoad['intraThalamicGain'],
    #                 'EbkgThalamicGain': cfgLoad['EbkgThalamicGain'],
    #                 'IbkgThalamicGain': cfgLoad['IbkgThalamicGain']})

    print(initCfg)


    # --------------------------------------------------------
    # fitness function
    fitnessFuncArgs = {}
    pops = {}
    
    ## Exc pops
    #Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6', 'TC', 'TCM', 'HTC']  # all layers + thal + IC
    Epops = ['ITP4', 'ITS4', 'TC', 'TCM', 'HTC']  # all layers + thal + IC

    #Etune = {'target': 5, 'width': 20, 'min': 0.05}
    Etune = {'target': 5, 'width': 5, 'min': 0.5}
    
    for pop in Epops:
        pops[pop] = Etune
    
    ## Inh pops 
    # Ipops = ['NGF1',                            # L1
    #         'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
    #         'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
    #         'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
    #         'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
    #         'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
    #         'PV6', 'SOM6', 'VIP6', 'NGF6',       # L6
    #         'IRE', 'IREM', 'TI']  # Thal 
    Ipops = [#'NGF1',  
            #'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
            #'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
            'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
            #'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A 
            #'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
            #'PV6', 'SOM6', 'VIP6', 'NGF6',  # L6
            'IRE', 'IREM', 'TI']  # Thal 

    #Itune = {'target': 10, 'width': 30, 'min': 0.05}
    Itune = {'target': 10, 'width': 15, 'min': 0.5}

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
                if simData['popRates'][k]['%d_%d'%(trange[0], trange[1])] >= v['min'] else maxFitness for k, v in pops.items()])
        
        popFitness = np.mean(np.array(popFitnessAll), axis=0)
        
        fitness = np.mean(popFitness)

        popInfo = '; '.join(['%s rate=%.1f fit=%1.f' % (p, np.mean(list(simData['popRates'][p].values())), popFitness[i]) for i,p in enumerate(pops)])
        print('  ' + popInfo)

        return fitness
        
    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)

    b.method = 'optuna'

    b.optimCfg = {
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'maxFitness': fitnessFuncArgs['maxFitness'],
        'maxiters':     1e6,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
        'maxtime':      None,    #    Maximum time allowed, in seconds
        'maxiter_wait': 45,
        'time_sleep': 120,
        'popsize': 1  # unused - run with mpi 
    }

    return b



# ----------------------------------------------------------------------------------------------
# Optuna optimization
# ----------------------------------------------------------------------------------------------
def optunaRatesLayersThalL234():

    # from prev
    import json
    with open('data/v34_batch1/trial_462/trial_462_cfg.json', 'rb') as f:
        cfgLoad = json.load(f)['simConfig']


    # --------------------------------------------------------
    # parameters
    params = specs.ODict()

    rangeV = 0.25
    rangeV2 = 0.1
    minV = 0.1
    maxV = 4.0

    params[('EICellTypeGain', 'PV')] = [max(cfgLoad['EICellTypeGain']['PV']-rangeV, minV), min(cfgLoad['EICellTypeGain']['PV']+rangeV, maxV)]
    params[('EICellTypeGain', 'SOM')] = [max(cfgLoad['EICellTypeGain']['SOM']-rangeV, minV), min(cfgLoad['EICellTypeGain']['SOM']+rangeV, maxV)]
    params[('EICellTypeGain', 'VIP')] = [max(cfgLoad['EICellTypeGain']['VIP']-rangeV, minV), min(cfgLoad['EICellTypeGain']['VIP']+rangeV, maxV)]
    params[('EICellTypeGain', 'NGF')] = [max(cfgLoad['EICellTypeGain']['NGF']-rangeV, minV), min(cfgLoad['EICellTypeGain']['NGF']+rangeV, maxV)]

    params[('IECellTypeGain', 'PV')] = [max(cfgLoad['IECellTypeGain']['PV']-rangeV, minV), min(cfgLoad['IECellTypeGain']['PV']+rangeV, maxV)]
    params[('IECellTypeGain', 'SOM')] = [max(cfgLoad['IECellTypeGain']['SOM']-rangeV, minV), min(cfgLoad['IECellTypeGain']['SOM']+rangeV, maxV)]
    params[('IECellTypeGain', 'VIP')] = [max(cfgLoad['IECellTypeGain']['VIP']-rangeV, minV), min(cfgLoad['IECellTypeGain']['VIP']+rangeV, maxV)]
    params[('IECellTypeGain', 'NGF')] = [max(cfgLoad['IECellTypeGain']['NGF']-rangeV, minV), min(cfgLoad['IECellTypeGain']['NGF']+rangeV, maxV)]

    # params[('EILayerGain', '1')] = [max(cfgLoad['EILayerGain']['1']-rangeV2, minV), min(cfgLoad['EILayerGain']['1']+rangeV, maxV)]
    # params[('IILayerGain', '1')] = [max(cfgLoad['IILayerGain']['1']-rangeV2, minV), min(cfgLoad['IILayerGain']['1']+rangeV, maxV)]

    params[('EELayerGain', '2')] = [minV, maxV]
    params[('EILayerGain', '2')] = [minV, maxV]
    params[('IELayerGain', '2')] = [minV, maxV]
    params[('IILayerGain', '2')] = [minV, maxV]

    params[('EELayerGain', '3')] = [minV, maxV]
    params[('EILayerGain', '3')] = [minV, maxV]
    params[('IELayerGain', '3')] = [minV, maxV]
    params[('IILayerGain', '3')] = [minV, maxV]

    params[('EELayerGain', '4')] = [max(cfgLoad['EELayerGain']['4']-rangeV2, minV), min(cfgLoad['EELayerGain']['4']+rangeV2, maxV)]
    params[('EILayerGain', '4')] = [max(cfgLoad['EILayerGain']['4']-rangeV2, minV), min(cfgLoad['EILayerGain']['4']+rangeV2, maxV)]
    params[('IELayerGain', '4')] = [max(cfgLoad['IELayerGain']['4']-rangeV2, minV), min(cfgLoad['IELayerGain']['4']+rangeV2, maxV)]
    params[('IILayerGain', '4')] = [max(cfgLoad['IILayerGain']['4']-rangeV2, minV), min(cfgLoad['IILayerGain']['4']+rangeV2, maxV)]

    # params[('EELayerGain', '5A')] = [max(cfgLoad['EELayerGain']['5A']-rangeV2, minV), min(cfgLoad['EELayerGain']['5A']+rangeV, maxV)]
    # params[('EILayerGain', '5A')] = [max(cfgLoad['EILayerGain']['5A']-rangeV2, minV), min(cfgLoad['EILayerGain']['5A']+rangeV, maxV)]
    # params[('IELayerGain', '5A')] = [max(cfgLoad['IELayerGain']['5A']-rangeV2, minV), min(cfgLoad['IELayerGain']['5A']+rangeV, maxV)]
    # params[('IILayerGain', '5A')] = [max(cfgLoad['IILayerGain']['5A']-rangeV2, minV), min(cfgLoad['IILayerGain']['5A']+rangeV, maxV)]

    # params[('EELayerGain', '5B')] = [max(cfgLoad['EELayerGain']['5B']-rangeV2, minV), min(cfgLoad['EELayerGain']['5B']+rangeV, maxV)]
    # params[('EILayerGain', '5B')] = [max(cfgLoad['EILayerGain']['5B']-rangeV2, minV), min(cfgLoad['EILayerGain']['5B']+rangeV, maxV)]
    # params[('IELayerGain', '5B')] = [max(cfgLoad['IELayerGain']['5B']-rangeV2, minV), min(cfgLoad['IELayerGain']['5B']+rangeV, maxV)]
    # params[('IILayerGain', '5B')] = [max(cfgLoad['IILayerGain']['5B']-rangeV2, minV), min(cfgLoad['IILayerGain']['5B']+rangeV, maxV)]

    # params[('EELayerGain', '6')] = [max(cfgLoad['EELayerGain']['6']-rangeV2, minV), min(cfgLoad['EELayerGain']['6']+rangeV, maxV)]
    # params[('EILayerGain', '6')] = [max(cfgLoad['EILayerGain']['6']-rangeV2, minV), min(cfgLoad['EILayerGain']['6']+rangeV, maxV)]
    # params[('IELayerGain', '6')] = [max(cfgLoad['IELayerGain']['6']-rangeV2, minV), min(cfgLoad['IELayerGain']['6']+rangeV, maxV)]
    # params[('IILayerGain', '6')] = [max(cfgLoad['IILayerGain']['6']-rangeV2, minV), min(cfgLoad['IILayerGain']['6']+rangeV, maxV)]


    groupedParams = []

    # --------------------------------------------------------
    # initial config
    initCfg = {}
    initCfg = {}
    initCfg['duration'] = 2500
    initCfg['printPopAvgRates'] = [[1500, 1750], [1750, 2000], [2000, 2250], [2250, 2500]]
    initCfg['dt'] = 0.05

    initCfg['scaleDensity'] = 1.0

    # plotting and saving params
    initCfg[('analysis','plotRaster','markerSize')] = 10

    initCfg[('analysis','plotRaster','timeRange')] = [1500, 2500]
    initCfg[('analysis', 'plotTraces', 'timeRange')] = [1500, 2500]
    initCfg[('analysis', 'plotTraces', 'oneFigPer')] = 'trace'
    initCfg['recordLFP'] = None
    initCfg[('analysis', 'plotLFP')] = False

    initCfg['EEGain'] = 1.0 	
    initCfg['EIGain'] = 1.0 	
    initCfg['IEGain'] = 1.0 	
    initCfg['IIGain'] = 1.0 	

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False
    
    initCfg.update({'thalamoCorticalGain': cfgLoad['thalamoCorticalGain'],
                    'intraThalamicGain': cfgLoad['intraThalamicGain'],
                    'EbkgThalamicGain': cfgLoad['EbkgThalamicGain'],
                    'IbkgThalamicGain': cfgLoad['IbkgThalamicGain']})

    print(initCfg)


    # --------------------------------------------------------
    # fitness function
    fitnessFuncArgs = {}
    pops = {}
    
    ## Exc pops
    #Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6', 'TC', 'TCM', 'HTC']  # all layers + thal + IC
    Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'TC', 'TCM', 'HTC']  # all layers + thal + IC

    #Etune = {'target': 5, 'width': 20, 'min': 0.05}
    Etune = {'target': 5, 'width': 5, 'min': 0.5}
    
    for pop in Epops:
        pops[pop] = Etune
    
    ## Inh pops 
    # Ipops = ['NGF1',                            # L1
    #         'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
    #         'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
    #         'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
    #         'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
    #         'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
    #         'PV6', 'SOM6', 'VIP6', 'NGF6',       # L6
    #         'IRE', 'IREM', 'TI']  # Thal 
    Ipops = [#'NGF1',  
            'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
            'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
            'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
            #'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A 
            #'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
            #'PV6', 'SOM6', 'VIP6', 'NGF6',  # L6
            'IRE', 'IREM', 'TI']  # Thal 

    #Itune = {'target': 10, 'width': 30, 'min': 0.05}
    Itune = {'target': 10, 'width': 15, 'min': 0.5}

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
                if simData['popRates'][k]['%d_%d'%(trange[0], trange[1])] >= v['min'] else maxFitness for k, v in pops.items()])
        
        popFitness = np.mean(np.array(popFitnessAll), axis=0)
        
        fitness = np.mean(popFitness)

        popInfo = '; '.join(['%s rate=%.1f fit=%1.f' % (p, np.mean(list(simData['popRates'][p].values())), popFitness[i]) for i,p in enumerate(pops)])
        print('  ' + popInfo)

        return fitness
        
    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)

    b.method = 'optuna'

    b.optimCfg = {
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'maxFitness': fitnessFuncArgs['maxFitness'],
        'maxiters':     1e6,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
        'maxtime':      None,    #    Maximum time allowed, in seconds
        'maxiter_wait': 45,
        'time_sleep': 120,
        'popsize': 1  # unused - run with mpi 
    }

    return b



# ----------------------------------------------------------------------------------------------
# Optuna optimization
# ----------------------------------------------------------------------------------------------
def optunaRatesLayersThalL2345A():

    # from prev
    import json
    with open('data/v34_batch4/v34_batch4_2_2_2_1_0_0_2_2_cfg.json', 'rb') as f:
        cfgLoad = json.load(f)['simConfig']


    # --------------------------------------------------------
    # parameters
    params = specs.ODict()

    rangeV = 0.25
    rangeV2 = 0.1
    minV = 0.1
    maxV = 4.0

    params[('EICellTypeGain', 'PV')] = [max(cfgLoad['EICellTypeGain']['PV']-rangeV, minV), min(cfgLoad['EICellTypeGain']['PV']+rangeV, maxV)]
    params[('EICellTypeGain', 'SOM')] = [max(cfgLoad['EICellTypeGain']['SOM']-rangeV, minV), min(cfgLoad['EICellTypeGain']['SOM']+rangeV, maxV)]
    params[('EICellTypeGain', 'VIP')] = [max(cfgLoad['EICellTypeGain']['VIP']-rangeV, minV), min(cfgLoad['EICellTypeGain']['VIP']+rangeV, maxV)]
    params[('EICellTypeGain', 'NGF')] = [max(cfgLoad['EICellTypeGain']['NGF']-rangeV, minV), min(cfgLoad['EICellTypeGain']['NGF']+rangeV, maxV)]

    params[('IECellTypeGain', 'PV')] = [max(cfgLoad['IECellTypeGain']['PV']-rangeV, minV), min(cfgLoad['IECellTypeGain']['PV']+rangeV, maxV)]
    params[('IECellTypeGain', 'SOM')] = [max(cfgLoad['IECellTypeGain']['SOM']-rangeV, minV), min(cfgLoad['IECellTypeGain']['SOM']+rangeV, maxV)]
    params[('IECellTypeGain', 'VIP')] = [max(cfgLoad['IECellTypeGain']['VIP']-rangeV, minV), min(cfgLoad['IECellTypeGain']['VIP']+rangeV, maxV)]
    params[('IECellTypeGain', 'NGF')] = [max(cfgLoad['IECellTypeGain']['NGF']-rangeV, minV), min(cfgLoad['IECellTypeGain']['NGF']+rangeV, maxV)]

    # params[('EILayerGain', '1')] = [max(cfgLoad['EILayerGain']['1']-rangeV2, minV), min(cfgLoad['EILayerGain']['1']+rangeV, maxV)]
    # params[('IILayerGain', '1')] = [max(cfgLoad['IILayerGain']['1']-rangeV2, minV), min(cfgLoad['IILayerGain']['1']+rangeV, maxV)]

    params[('EELayerGain', '2')] = [max(cfgLoad['EELayerGain']['4']-rangeV2, minV), min(cfgLoad['EELayerGain']['4']+rangeV2, maxV)]
    params[('EILayerGain', '2')] = [max(cfgLoad['EELayerGain']['4']-rangeV2, minV), min(cfgLoad['EELayerGain']['4']+rangeV2, maxV)]
    params[('IELayerGain', '2')] = [max(cfgLoad['EELayerGain']['4']-rangeV2, minV), min(cfgLoad['EELayerGain']['4']+rangeV2, maxV)]
    params[('IILayerGain', '2')] = [max(cfgLoad['EELayerGain']['4']-rangeV2, minV), min(cfgLoad['EELayerGain']['4']+rangeV2, maxV)]

    params[('EELayerGain', '3')] = [max(cfgLoad['EELayerGain']['4']-rangeV2, minV), min(cfgLoad['EELayerGain']['4']+rangeV2, maxV)]
    params[('EILayerGain', '3')] = [max(cfgLoad['EELayerGain']['4']-rangeV2, minV), min(cfgLoad['EELayerGain']['4']+rangeV2, maxV)]
    params[('IELayerGain', '3')] = [max(cfgLoad['EELayerGain']['4']-rangeV2, minV), min(cfgLoad['EELayerGain']['4']+rangeV2, maxV)]
    params[('IILayerGain', '3')] = [max(cfgLoad['EELayerGain']['4']-rangeV2, minV), min(cfgLoad['EELayerGain']['4']+rangeV2, maxV)]

    params[('EELayerGain', '4')] = [max(cfgLoad['EELayerGain']['4']-rangeV2, minV), min(cfgLoad['EELayerGain']['4']+rangeV2, maxV)]
    params[('EILayerGain', '4')] = [max(cfgLoad['EILayerGain']['4']-rangeV2, minV), min(cfgLoad['EILayerGain']['4']+rangeV2, maxV)]
    params[('IELayerGain', '4')] = [max(cfgLoad['IELayerGain']['4']-rangeV2, minV), min(cfgLoad['IELayerGain']['4']+rangeV2, maxV)]
    params[('IILayerGain', '4')] = [max(cfgLoad['IILayerGain']['4']-rangeV2, minV), min(cfgLoad['IILayerGain']['4']+rangeV2, maxV)]

    params[('EELayerGain', '5A')] = [minV, maxV]
    params[('EILayerGain', '5A')] = [minV, maxV]
    params[('IELayerGain', '5A')] = [minV, maxV]
    params[('IILayerGain', '5A')] = [minV, maxV]

    # params[('EELayerGain', '5B')] = [max(cfgLoad['EELayerGain']['5B']-rangeV2, minV), min(cfgLoad['EELayerGain']['5B']+rangeV, maxV)]
    # params[('EILayerGain', '5B')] = [max(cfgLoad['EILayerGain']['5B']-rangeV2, minV), min(cfgLoad['EILayerGain']['5B']+rangeV, maxV)]
    # params[('IELayerGain', '5B')] = [max(cfgLoad['IELayerGain']['5B']-rangeV2, minV), min(cfgLoad['IELayerGain']['5B']+rangeV, maxV)]
    # params[('IILayerGain', '5B')] = [max(cfgLoad['IILayerGain']['5B']-rangeV2, minV), min(cfgLoad['IILayerGain']['5B']+rangeV, maxV)]

    # params[('EELayerGain', '6')] = [max(cfgLoad['EELayerGain']['6']-rangeV2, minV), min(cfgLoad['EELayerGain']['6']+rangeV, maxV)]
    # params[('EILayerGain', '6')] = [max(cfgLoad['EILayerGain']['6']-rangeV2, minV), min(cfgLoad['EILayerGain']['6']+rangeV, maxV)]
    # params[('IELayerGain', '6')] = [max(cfgLoad['IELayerGain']['6']-rangeV2, minV), min(cfgLoad['IELayerGain']['6']+rangeV, maxV)]
    # params[('IILayerGain', '6')] = [max(cfgLoad['IILayerGain']['6']-rangeV2, minV), min(cfgLoad['IILayerGain']['6']+rangeV, maxV)]


    groupedParams = []

    # --------------------------------------------------------
    # initial config
    initCfg = {}
    initCfg = {}
    initCfg['duration'] = 2500
    initCfg['printPopAvgRates'] = [[1500, 1750], [1750, 2000], [2000, 2250], [2250, 2500]]
    initCfg['dt'] = 0.05

    initCfg['scaleDensity'] = 1.0

    # plotting and saving params
    initCfg[('analysis','plotRaster','markerSize')] = 10

    initCfg[('analysis','plotRaster','timeRange')] = [1500, 2500]
    initCfg[('analysis', 'plotTraces', 'timeRange')] = [1500, 2500]
    initCfg[('analysis', 'plotTraces', 'oneFigPer')] = 'trace'
    initCfg['recordLFP'] = None
    initCfg[('analysis', 'plotLFP')] = False

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False


    # from prev - best of 50% cell density
    updateParams = ['EEGain', 'EIGain', 'IEGain', 'IIGain',
                    ('EICellTypeGain', 'PV'), ('EICellTypeGain', 'SOM'), ('EICellTypeGain', 'VIP'), ('EICellTypeGain', 'NGF'),
                    ('IECellTypeGain', 'PV'), ('IECellTypeGain', 'SOM'), ('IECellTypeGain', 'VIP'), ('IECellTypeGain', 'NGF'),
                    ('EILayerGain', '1'), ('IILayerGain', '1'),
                    ('EELayerGain', '2'), ('EILayerGain', '2'),  ('IELayerGain', '2'), ('IILayerGain', '2'), 
                    ('EELayerGain', '3'), ('EILayerGain', '3'), ('IELayerGain', '3'), ('IILayerGain', '3'), 
                    ('EELayerGain', '4'), ('EILayerGain', '4'), ('IELayerGain', '4'), ('IILayerGain', '4'), 
                    ('EELayerGain', '5A'), ('EILayerGain', '5A'), ('IELayerGain', '5A'), ('IILayerGain', '5A'), 
                    ('EELayerGain', '5B'), ('EILayerGain', '5B'), ('IELayerGain', '5B'), ('IILayerGain', '5B'), 
                    ('EELayerGain', '6'), ('EILayerGain', '6'), ('IELayerGain', '6'), ('IILayerGain', '6')] 

    for p in updateParams:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})

    # good thal params for 100% cell density 
    updateParams2 = ['thalamoCorticalGain', 'intraThalamicGain', 'EbkgThalamicGain', 'IbkgThalamicGain']

    for p in updateParams2:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})

    print(initCfg)


    # --------------------------------------------------------
    # fitness function
    fitnessFuncArgs = {}
    pops = {}
    
    ## Exc pops
    #Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6', 'TC', 'TCM', 'HTC']  # all layers + thal + IC
    Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'TC', 'TCM', 'HTC']  # all layers + thal + IC

    #Etune = {'target': 5, 'width': 20, 'min': 0.05}
    Etune = {'target': 5, 'width': 5, 'min': 0.5}
    
    for pop in Epops:
        pops[pop] = Etune
    
    ## Inh pops 
    # Ipops = ['NGF1',                            # L1
    #         'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
    #         'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
    #         'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
    #         'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
    #         'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
    #         'PV6', 'SOM6', 'VIP6', 'NGF6',       # L6
    #         'IRE', 'IREM', 'TI']  # Thal 
    Ipops = [#'NGF1',  
            'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
            'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
            'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
            'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A 
            #'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
            #'PV6', 'SOM6', 'VIP6', 'NGF6',  # L6
            'IRE', 'IREM', 'TI']  # Thal 

    #Itune = {'target': 10, 'width': 30, 'min': 0.05}
    Itune = {'target': 10, 'width': 15, 'min': 0.5}

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
                if simData['popRates'][k]['%d_%d'%(trange[0], trange[1])] >= v['min'] else maxFitness for k, v in pops.items()])
        
        popFitness = np.mean(np.array(popFitnessAll), axis=0)
        
        fitness = np.mean(popFitness)

        popInfo = '; '.join(['%s rate=%.1f fit=%1.f' % (p, np.mean(list(simData['popRates'][p].values())), popFitness[i]) for i,p in enumerate(pops)])
        print('  ' + popInfo)

        return fitness
        
    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)

    b.method = 'optuna'

    b.optimCfg = {
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'maxFitness': fitnessFuncArgs['maxFitness'],
        'maxiters':     1e6,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
        'maxtime':      None,    #    Maximum time allowed, in seconds
        'maxiter_wait': 45,
        'time_sleep': 120,
        'popsize': 1  # unused - run with mpi 
    }

    return b


# ----------------------------------------------------------------------------------------------
# Optuna optimization
# ----------------------------------------------------------------------------------------------
def optunaRatesLayersThalL2345A5B():

    # from prev
    import json
    with open('data/v34_batch11/trial_2150/trial_2150_cfg.json', 'rb') as f:
        cfgLoad = json.load(f)['simConfig']


    # --------------------------------------------------------
    # parameters
    params = specs.ODict()

    rangeV1 = 0.1
    rangeV2 = 0.25
    rangeV3 = 2.0
    minV = 0.1
    maxV = 4.0

    ''' potentially:
    layer-specific EE,EI,IE,II gains for L2, with [prev-0.25:prev+0.25]
    layer-specific EE,EI,IE,II gains for L3, with [prev-0.25:prev+025]
    layer-specific EE,EI,IE,II gains for L4, with [prev-0.1:prev+0.1]
    layer-specific EE,EI,IE,II gains for L5A, with param range [prev-0.1:prev+0.1]
    layer-specific EE,EI,IE,II gains for L5B, with broad param range [0.1:4.0]
    E-PV, E-SOM, E-NGF, E-VIP (common to all layers) with range [prev-0.25:prev+0.25]
    PV-E, SOM-E, NGF-E, VIP-E (common to all layers) with range [0.1:4.0]

    '''

    # 0.25
    params[('EICellTypeGain', 'PV')] = [max(cfgLoad['EICellTypeGain']['PV']-rangeV2, minV), min(cfgLoad['EICellTypeGain']['PV']+rangeV2, maxV)]
    params[('EICellTypeGain', 'SOM')] = [max(cfgLoad['EICellTypeGain']['SOM']-rangeV2, minV), min(cfgLoad['EICellTypeGain']['SOM']+rangeV2, maxV)]
    params[('EICellTypeGain', 'VIP')] = [max(cfgLoad['EICellTypeGain']['VIP']-rangeV2, minV), min(cfgLoad['EICellTypeGain']['VIP']+rangeV2, maxV)]
    params[('EICellTypeGain', 'NGF')] = [max(cfgLoad['EICellTypeGain']['NGF']-rangeV2, minV), min(cfgLoad['EICellTypeGain']['NGF']+rangeV3, maxV)]

    # 0.1
    params[('IECellTypeGain', 'PV')] = [minV, maxV] # [max(cfgLoad['IECellTypeGain']['PV']-rangeV3, minV), min(cfgLoad['IECellTypeGain']['PV']+rangeV3, maxV)]
    params[('IECellTypeGain', 'SOM')] = [minV, maxV] # [max(cfgLoad['IECellTypeGain']['SOM']-rangeV3, minV), min(cfgLoad['IECellTypeGain']['SOM']+rangeV3, maxV)]
    params[('IECellTypeGain', 'VIP')] = [minV, maxV] # [max(cfgLoad['IECellTypeGain']['VIP']-rangeV3, minV), min(cfgLoad['IECellTypeGain']['VIP']+rangeV3, maxV)]
    params[('IECellTypeGain', 'NGF')] = [minV, maxV] # [max(cfgLoad['IECellTypeGain']['NGF']-rangeV3, minV), min(cfgLoad['IECellTypeGain']['NGF']+rangeV3, maxV)]

    # params[('EILayerGain', '1')] = [max(cfgLoad['EILayerGain']['1']-rangeV2, minV), min(cfgLoad['EILayerGain']['1']+rangeV, maxV)]
    # params[('IILayerGain', '1')] = [max(cfgLoad['IILayerGain']['1']-rangeV2, minV), min(cfgLoad['IILayerGain']['1']+rangeV, maxV)]

    # 0.25
    params[('EELayerGain', '2')] = [max(cfgLoad['EELayerGain']['2']-rangeV2, minV), min(cfgLoad['EELayerGain']['2']+rangeV2, maxV)]
    params[('EILayerGain', '2')] = [max(cfgLoad['EELayerGain']['2']-rangeV2, minV), min(cfgLoad['EELayerGain']['2']+rangeV2, maxV)]
    params[('IELayerGain', '2')] = [max(cfgLoad['EELayerGain']['2']-rangeV2, minV), min(cfgLoad['EELayerGain']['2']+rangeV2, maxV)]
    params[('IILayerGain', '2')] = [max(cfgLoad['EELayerGain']['2']-rangeV2, minV), min(cfgLoad['EELayerGain']['2']+rangeV2, maxV)]

    # 0.1
    params[('EELayerGain', '3')] = [max(cfgLoad['EELayerGain']['3']-rangeV2, minV), min(cfgLoad['EELayerGain']['3']+rangeV2, maxV)]
    params[('EILayerGain', '3')] = [max(cfgLoad['EELayerGain']['3']-rangeV2, minV), min(cfgLoad['EELayerGain']['3']+rangeV2, maxV)]
    params[('IELayerGain', '3')] = [max(cfgLoad['EELayerGain']['3']-rangeV2, minV), min(cfgLoad['EELayerGain']['3']+rangeV2, maxV)]
    params[('IILayerGain', '3')] = [max(cfgLoad['EELayerGain']['3']-rangeV2, minV), min(cfgLoad['EELayerGain']['3']+rangeV2, maxV)]

    # 0.1
    params[('EELayerGain', '4')] = [max(cfgLoad['EELayerGain']['4']-rangeV1, minV), min(cfgLoad['EELayerGain']['4']+rangeV1, maxV)]
    params[('EILayerGain', '4')] = [max(cfgLoad['EILayerGain']['4']-rangeV1, minV), min(cfgLoad['EILayerGain']['4']+rangeV1, maxV)]
    params[('IELayerGain', '4')] = [max(cfgLoad['IELayerGain']['4']-rangeV1, minV), min(cfgLoad['IELayerGain']['4']+rangeV1, maxV)]
    params[('IILayerGain', '4')] = [max(cfgLoad['IILayerGain']['4']-rangeV1, minV), min(cfgLoad['IILayerGain']['4']+rangeV1, maxV)]
    
    # 0.1
    params[('EELayerGain', '5A')] = [max(cfgLoad['EELayerGain']['5A']-rangeV1, minV), min(cfgLoad['EELayerGain']['5A']+rangeV1, maxV)]
    params[('EILayerGain', '5A')] = [max(cfgLoad['EILayerGain']['5A']-rangeV1, minV), min(cfgLoad['EILayerGain']['5A']+rangeV1, maxV)]
    params[('IELayerGain', '5A')] = [max(cfgLoad['IELayerGain']['5A']-rangeV1, minV), min(cfgLoad['IELayerGain']['5A']+rangeV1, maxV)]
    params[('IILayerGain', '5A')] = [max(cfgLoad['IILayerGain']['5A']-rangeV1, minV), min(cfgLoad['IILayerGain']['5A']+rangeV1, maxV)]

    # 0.25
    params[('EELayerGain', '5B')] = [minV, maxV] #[max(cfgLoad['EELayerGain']['5B']-rangeV2, minV), min(cfgLoad['EELayerGain']['5B']+rangeV2, maxV)]
    params[('EILayerGain', '5B')] = [minV, maxV] # [max(cfgLoad['EILayerGain']['5B']-rangeV2, minV), min(cfgLoad['EILayerGain']['5B']+rangeV2, maxV)]
    params[('IELayerGain', '5B')] = [minV, maxV] #[max(cfgLoad['IELayerGain']['5B']-rangeV2, minV), min(cfgLoad['IELayerGain']['5B']+rangeV2, maxV)]
    params[('IILayerGain', '5B')] = [minV, maxV] #[max(cfgLoad['IILayerGain']['5B']-rangeV2, minV), min(cfgLoad['IILayerGain']['5B']+rangeV2, maxV)]

    # params[('EELayerGain', '6')] = [max(cfgLoad['EELayerGain']['6']-rangeV2, minV), min(cfgLoad['EELayerGain']['6']+rangeV, maxV)]
    # params[('EILayerGain', '6')] = [max(cfgLoad['EILayerGain']['6']-rangeV2, minV), min(cfgLoad['EILayerGain']['6']+rangeV, maxV)]
    # params[('IELayerGain', '6')] = [max(cfgLoad['IELayerGain']['6']-rangeV2, minV), min(cfgLoad['IELayerGain']['6']+rangeV, maxV)]
    # params[('IILayerGain', '6')] = [max(cfgLoad['IILayerGain']['6']-rangeV2, minV), min(cfgLoad['IILayerGain']['6']+rangeV, maxV)]


    groupedParams = []

    # --------------------------------------------------------
    # initial config
    initCfg = {}
    initCfg = {}
    initCfg['duration'] = 2500
    initCfg['printPopAvgRates'] = [[1500, 1750], [1750, 2000], [2000, 2250], [2250, 2500]]
    initCfg['dt'] = 0.05

    initCfg['scaleDensity'] = 1.0

    # plotting and saving params
    initCfg[('analysis','plotRaster','markerSize')] = 10

    initCfg[('analysis','plotRaster','timeRange')] = [1500, 2500]
    initCfg[('analysis', 'plotTraces', 'timeRange')] = [1500, 2500]
    initCfg[('analysis', 'plotTraces', 'oneFigPer')] = 'trace'
    initCfg['recordLFP'] = None
    initCfg[('analysis', 'plotLFP')] = False

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False


    # from prev - best of 50% cell density
    updateParams = ['EEGain', 'EIGain', 'IEGain', 'IIGain',
                    ('EICellTypeGain', 'PV'), ('EICellTypeGain', 'SOM'), ('EICellTypeGain', 'VIP'), ('EICellTypeGain', 'NGF'),
                    ('IECellTypeGain', 'PV'), ('IECellTypeGain', 'SOM'), ('IECellTypeGain', 'VIP'), ('IECellTypeGain', 'NGF'),
                    ('EILayerGain', '1'), ('IILayerGain', '1'),
                    ('EELayerGain', '2'), ('EILayerGain', '2'),  ('IELayerGain', '2'), ('IILayerGain', '2'), 
                    ('EELayerGain', '3'), ('EILayerGain', '3'), ('IELayerGain', '3'), ('IILayerGain', '3'), 
                    ('EELayerGain', '4'), ('EILayerGain', '4'), ('IELayerGain', '4'), ('IILayerGain', '4'), 
                    ('EELayerGain', '5A'), ('EILayerGain', '5A'), ('IELayerGain', '5A'), ('IILayerGain', '5A'), 
                    ('EELayerGain', '5B'), ('EILayerGain', '5B'), ('IELayerGain', '5B'), ('IILayerGain', '5B'), 
                    ('EELayerGain', '6'), ('EILayerGain', '6'), ('IELayerGain', '6'), ('IILayerGain', '6')] 

    for p in updateParams:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})

    # good thal params for 100% cell density 
    updateParams2 = ['thalamoCorticalGain', 'intraThalamicGain', 'EbkgThalamicGain', 'IbkgThalamicGain']

    for p in updateParams2:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})

    print(initCfg)


    # --------------------------------------------------------
    # fitness function
    fitnessFuncArgs = {}
    pops = {}
    
    ## Exc pops
    #Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6', 'TC', 'TCM', 'HTC']  # all layers + thal + IC
    Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'TC', 'TCM', 'HTC']  # all layers + thal + IC

    #Etune = {'target': 5, 'width': 20, 'min': 0.05}
    Etune = {'target': 5, 'width': 5, 'min': 0.5}
    
    for pop in Epops:
        pops[pop] = Etune
    
    ## Inh pops 
    # Ipops = ['NGF1',                            # L1
    #         'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
    #         'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
    #         'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
    #         'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
    #         'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
    #         'PV6', 'SOM6', 'VIP6', 'NGF6',       # L6
    #         'IRE', 'IREM', 'TI']  # Thal 
    Ipops = [#'NGF1',  
            'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
            'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
            'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
            'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A 
            'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
            #'PV6', 'SOM6', 'VIP6', 'NGF6',  # L6
            'IRE', 'IREM', 'TI']  # Thal 

    #Itune = {'target': 10, 'width': 30, 'min': 0.05}
    Itune = {'target': 10, 'width': 15, 'min': 0.5}

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
                if simData['popRates'][k]['%d_%d'%(trange[0], trange[1])] >= v['min'] else maxFitness for k, v in pops.items()])
        
        popFitness = np.mean(np.array(popFitnessAll), axis=0)
        
        fitness = np.mean(popFitness)

        popInfo = '; '.join(['%s rate=%.1f fit=%1.f' % (p, np.mean(list(simData['popRates'][p].values())), popFitness[i]) for i,p in enumerate(pops)])
        print('  ' + popInfo)

        return fitness
        
    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)

    b.method = 'optuna'

    b.optimCfg = {
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'maxFitness': fitnessFuncArgs['maxFitness'],
        'maxiters':     1e6,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
        'maxtime':      None,    #    Maximum time allowed, in seconds
        'maxiter_wait': 45,
        'time_sleep': 120,
        'popsize': 1  # unused - run with mpi 
    }

    return b


# ----------------------------------------------------------------------------------------------
# Optuna optimization
# ----------------------------------------------------------------------------------------------
def optunaRatesLayersThalL12345A5B6():

    # from prev
    import json
    with open('data/v34_batch15/trial_5955/trial_5955_cfg.json', 'rb') as f:
        cfgLoad = json.load(f)['simConfig']


    # --------------------------------------------------------
    # parameters
    params = specs.ODict()

    rangeV1 = 0.2
    rangeV2 = 0.25
    rangeV3 = 2.0

    scaleLow = 0.9
    scaleHigh = 1.1

    scaleLow2 = 0.5
    scaleHigh2 = 1.5

    minV = 0.1
    maxV = 5.0

    ''' potentially:
    layer-specific EE,EI,IE,II gains for L2, with [prev-0.25:prev+0.25]
    layer-specific EE,EI,IE,II gains for L3, with [prev-0.25:prev+025]
    layer-specific EE,EI,IE,II gains for L4, with [prev-0.1:prev+0.1]
    layer-specific EE,EI,IE,II gains for L5A, with param range [prev-0.1:prev+0.1]
    layer-specific EE,EI,IE,II gains for L5B, with broad param range [0.1:4.0]
    E-PV, E-SOM, E-NGF, E-VIP (common to all layers) with range [prev-0.25:prev+0.25]
    PV-E, SOM-E, NGF-E, VIP-E (common to all layers) with range [0.1:4.0]

    '''

    # E->I cell-type-specific
    params[('EICellTypeGain', 'PV')] = [max(cfgLoad['EICellTypeGain']['PV']*scaleLow2, minV), min(cfgLoad['EICellTypeGain']['PV']*scaleHigh2, maxV)]
    params[('EICellTypeGain', 'SOM')] = [max(cfgLoad['EICellTypeGain']['SOM']*scaleLow2, minV), min(cfgLoad['EICellTypeGain']['SOM']*scaleHigh2, maxV)]
    params[('EICellTypeGain', 'VIP')] = [max(cfgLoad['EICellTypeGain']['VIP']*scaleLow2, minV), min(cfgLoad['EICellTypeGain']['VIP']*scaleHigh2, maxV)]
    params[('EICellTypeGain', 'NGF')] = [max(cfgLoad['EICellTypeGain']['NGF']*scaleLow2, minV), min(cfgLoad['EICellTypeGain']['NGF']*scaleHigh2, maxV)]

    # I->E cell-type-specific
    params[('IECellTypeGain', 'PV')] = [max(cfgLoad['IECellTypeGain']['PV']*scaleLow, minV), min(cfgLoad['IECellTypeGain']['PV']*scaleHigh, maxV)]
    params[('IECellTypeGain', 'SOM')] = [max(cfgLoad['IECellTypeGain']['SOM']*scaleLow, minV), min(cfgLoad['IECellTypeGain']['SOM']*scaleHigh, maxV)]
    params[('IECellTypeGain', 'VIP')] = [max(cfgLoad['IECellTypeGain']['VIP']*scaleLow, minV), min(cfgLoad['IECellTypeGain']['VIP']*scaleHigh, maxV)]
    params[('IECellTypeGain', 'NGF')] = [max(cfgLoad['IECellTypeGain']['NGF']*scaleLow, minV), min(cfgLoad['IECellTypeGain']['NGF']*scaleHigh, maxV)]

    # L1
    params[('EILayerGain', '1')] = [max(cfgLoad['EILayerGain']['1']*scaleLow, minV), min(cfgLoad['EILayerGain']['1']*scaleHigh, maxV)]
    params[('IILayerGain', '1')] = [max(cfgLoad['IILayerGain']['1']*scaleLow, minV), min(cfgLoad['IILayerGain']['1']*scaleHigh, maxV)]

    # L2
    params[('EELayerGain', '2')] = [max(cfgLoad['EELayerGain']['2']*scaleLow2, minV), min(cfgLoad['EELayerGain']['2']*scaleHigh2, maxV)]
    params[('EILayerGain', '2')] = [max(cfgLoad['EILayerGain']['2']*scaleLow2, minV), min(cfgLoad['EILayerGain']['2']*scaleHigh2, maxV)]
    params[('IELayerGain', '2')] = [max(cfgLoad['IELayerGain']['2']*scaleLow2, minV), min(cfgLoad['IELayerGain']['2']*scaleHigh2, maxV)]
    params[('IILayerGain', '2')] = [max(cfgLoad['IILayerGain']['2']*scaleLow2, minV), min(cfgLoad['IILayerGain']['2']*scaleHigh2, maxV)]

    # L3
    params[('EELayerGain', '3')] = [max(cfgLoad['EELayerGain']['3']*scaleLow2, minV), min(cfgLoad['EELayerGain']['3']*scaleHigh2, maxV)]
    params[('EILayerGain', '3')] = [max(cfgLoad['EILayerGain']['3']*scaleLow2, minV), min(cfgLoad['EILayerGain']['3']*scaleHigh2, maxV)]
    params[('IELayerGain', '3')] = [max(cfgLoad['IELayerGain']['3']*scaleLow2, minV), min(cfgLoad['IELayerGain']['3']*scaleHigh2, maxV)]
    params[('IILayerGain', '3')] = [max(cfgLoad['IILayerGain']['3']*scaleLow2, minV), min(cfgLoad['IILayerGain']['3']*scaleHigh2, maxV)]

    # L4
    params[('EELayerGain', '4')] = [max(cfgLoad['EELayerGain']['4']*scaleLow, minV), min(cfgLoad['EELayerGain']['4']*scaleHigh, maxV)]
    params[('EILayerGain', '4')] = [max(cfgLoad['EILayerGain']['4']*scaleLow, minV), min(cfgLoad['EILayerGain']['4']*scaleHigh, maxV)]
    params[('IELayerGain', '4')] = [max(cfgLoad['IELayerGain']['4']*scaleLow, minV), min(cfgLoad['IELayerGain']['4']*scaleHigh, maxV)]
    params[('IILayerGain', '4')] = [max(cfgLoad['IILayerGain']['4']*scaleLow, minV), min(cfgLoad['IILayerGain']['4']*scaleHigh, maxV)]
    
    # L5A
    params[('EELayerGain', '5A')] = [max(cfgLoad['EELayerGain']['5A']*scaleLow, minV), min(cfgLoad['EELayerGain']['5A']*scaleHigh, maxV)]
    params[('EILayerGain', '5A')] = [max(cfgLoad['EILayerGain']['5A']*scaleLow, minV), min(cfgLoad['EILayerGain']['5A']*scaleHigh, maxV)]
    params[('IELayerGain', '5A')] = [max(cfgLoad['IELayerGain']['5A']*scaleLow, minV), min(cfgLoad['IELayerGain']['5A']*scaleHigh, maxV)]
    params[('IILayerGain', '5A')] = [max(cfgLoad['IILayerGain']['5A']*scaleLow, minV), min(cfgLoad['IILayerGain']['5A']*scaleHigh, maxV)]

    # L5B
    params[('EELayerGain', '5B')] = [max(cfgLoad['EELayerGain']['5B']*scaleLow, minV), min(cfgLoad['EELayerGain']['5B']*scaleHigh, maxV)]
    params[('EILayerGain', '5B')] = [max(cfgLoad['EILayerGain']['5B']*scaleLow, minV), min(cfgLoad['EILayerGain']['5B']*scaleHigh, maxV)]
    params[('IELayerGain', '5B')] = [max(cfgLoad['IELayerGain']['5B']*scaleLow, minV), min(cfgLoad['IELayerGain']['5B']*scaleHigh, maxV)]
    params[('IILayerGain', '5B')] = [max(cfgLoad['IILayerGain']['5B']*scaleLow, minV), min(cfgLoad['IILayerGain']['5B']*scaleHigh, maxV)]

    # L6
    params[('EELayerGain', '6')] = [max(cfgLoad['EELayerGain']['6']*scaleLow, minV), min(cfgLoad['EELayerGain']['6']*scaleHigh, maxV)]
    params[('EILayerGain', '6')] = [max(cfgLoad['EILayerGain']['6']*scaleLow, minV), min(cfgLoad['EILayerGain']['6']*scaleHigh, maxV)]
    params[('IELayerGain', '6')] = [max(cfgLoad['IELayerGain']['6']*scaleLow, minV), min(cfgLoad['IELayerGain']['6']*scaleHigh, maxV)]
    params[('IILayerGain', '6')] = [max(cfgLoad['IILayerGain']['6']*scaleLow, minV), min(cfgLoad['IILayerGain']['6']*scaleHigh, maxV)]


    groupedParams = []

    # --------------------------------------------------------
    # initial config
    initCfg = {}
    initCfg = {}
    initCfg['duration'] = 2500
    initCfg['printPopAvgRates'] = [[1500, 1750], [1750, 2000], [2000, 2250], [2250, 2500]]
    initCfg['dt'] = 0.05

    initCfg['scaleDensity'] = 1.0

    # plotting and saving params
    initCfg[('analysis','plotRaster','markerSize')] = 10

    initCfg[('analysis','plotRaster','timeRange')] = [1500, 2500]
    initCfg[('analysis', 'plotTraces', 'timeRange')] = [1500, 2500]
    initCfg[('analysis', 'plotTraces', 'oneFigPer')] = 'trace'
    initCfg['recordLFP'] = None
    initCfg[('analysis', 'plotLFP')] = False

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False


    # from prev - best of 50% cell density
    updateParams = ['EEGain', 'EIGain', 'IEGain', 'IIGain',
                    ('EICellTypeGain', 'PV'), ('EICellTypeGain', 'SOM'), ('EICellTypeGain', 'VIP'), ('EICellTypeGain', 'NGF'),
                    ('IECellTypeGain', 'PV'), ('IECellTypeGain', 'SOM'), ('IECellTypeGain', 'VIP'), ('IECellTypeGain', 'NGF'),
                    ('EILayerGain', '1'), ('IILayerGain', '1'),
                    ('EELayerGain', '2'), ('EILayerGain', '2'),  ('IELayerGain', '2'), ('IILayerGain', '2'), 
                    ('EELayerGain', '3'), ('EILayerGain', '3'), ('IELayerGain', '3'), ('IILayerGain', '3'), 
                    ('EELayerGain', '4'), ('EILayerGain', '4'), ('IELayerGain', '4'), ('IILayerGain', '4'), 
                    ('EELayerGain', '5A'), ('EILayerGain', '5A'), ('IELayerGain', '5A'), ('IILayerGain', '5A'), 
                    ('EELayerGain', '5B'), ('EILayerGain', '5B'), ('IELayerGain', '5B'), ('IILayerGain', '5B'), 
                    ('EELayerGain', '6'), ('EILayerGain', '6'), ('IELayerGain', '6'), ('IILayerGain', '6')] 

    for p in updateParams:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})

    # good thal params for 100% cell density 
    updateParams2 = ['thalamoCorticalGain', 'intraThalamicGain', 'EbkgThalamicGain', 'IbkgThalamicGain']

    for p in updateParams2:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})

    print(initCfg)


    # --------------------------------------------------------
    # fitness function
    fitnessFuncArgs = {}
    pops = {}
    
    ## Exc pops
    Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6', 'TC', 'TCM', 'HTC']  # all layers + thal + IC
    # Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'TC', 'TCM', 'HTC']  # all layers + thal + IC

    Etune = {'target': 5, 'width': 20, 'min': 0.05}
    #Etune = {'target': 5, 'width': 5, 'min': 0.5}
    
    for pop in Epops:
        pops[pop] = Etune
    
    ## Inh pops 
    # Ipops = ['NGF1',                            # L1
    #         'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
    #         'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
    #         'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
    #         'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
    #         'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
    #         'PV6', 'SOM6', 'VIP6', 'NGF6',       # L6
    #         'IRE', 'IREM', 'TI']  # Thal 
    Ipops = ['NGF1',  
            'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
            'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
            'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
            'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A 
            'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
            'PV6', 'SOM6', 'VIP6', 'NGF6',  # L6
            'IRE', 'IREM', 'TI']  # Thal 

    Itune = {'target': 10, 'width': 30, 'min': 0.05}
    #Itune = {'target': 10, 'width': 15, 'min': 0.5}

    for pop in Ipops:
        pops[pop] = Itune
    
    fitnessFuncArgs['pops'] = pops
    fitnessFuncArgs['maxFitness'] = 2000
    fitnessFuncArgs['tranges'] = initCfg['printPopAvgRates']


    def fitnessFunc(simData, **kwargs):
        import numpy as np
        pops = kwargs['pops']
        maxFitness = kwargs['maxFitness']
        tranges = kwargs['tranges']

        popFitnessAll = []

        for trange in tranges:
            popFitnessAll.append([min(np.exp(abs(v['target'] - simData['popRates'][k]['%d_%d'%(trange[0], trange[1])])/v['width']), maxFitness) 
                if simData['popRates'][k]['%d_%d'%(trange[0], trange[1])] >= v['min'] else maxFitness for k, v in pops.items()])
        
        popFitness = np.mean(np.array(popFitnessAll), axis=0)
        
        fitness = np.mean(popFitness)

        popInfo = '; '.join(['%s rate=%.1f fit=%1.f' % (p, np.mean(list(simData['popRates'][p].values())), popFitness[i]) for i,p in enumerate(pops)])
        print('  ' + popInfo)

        return fitness
        
    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)

    b.method = 'optuna'

    b.optimCfg = {
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'maxFitness': fitnessFuncArgs['maxFitness'],
        'maxiters':     1e6,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
        'maxtime':      None,    #    Maximum time allowed, in seconds
        'maxiter_wait': 60,
        'time_sleep': 150,
        'popsize': 1  # unused - run with mpi 
    }

    return b



# ----------------------------------------------------------------------------------------------
# Optuna optimization
# ----------------------------------------------------------------------------------------------
def optunaRatesLayersWmat():

    # from prev
    import json
    #with open('data/v34_batch25/trial_2142/trial_2142_cfg.json', 'rb') as f:
    #    cfgLoad = json.load(f)['simConfig']

    with open('data/v34_batch23/trial_1937/trial_1937_cfg.json', 'rb') as f:
        cfgLoad = json.load(f)['simConfig']


    # --------------------------------------------------------
    # parameters
    params = specs.ODict()

    scaleLow = 0.75
    scaleHigh = 1.25

    #scaleLow2 = 0.1
    #scaleHigh2 = 10.0

    scaleLow2 = 0.5
    scaleHigh2 = 2.0

    # import pickle
    # with open('conn/conn.pkl', 'rb') as fileObj: connData = pickle.load(fileObj)
    # wmat = connData['wmat']
    wmat = cfgLoad['wmat']    

    # ALL 36
    # weightsScale = [['IT6', 'NGF6'],
    #                 ['CT6', 'NGF6']
    #                 ['PV6', 'NGF6'],
    #                 ['SOM6', 'NGF6'],
    #                 ['VIP6', 'NGF6'],
    #                 ['NGF6', 'NGF6'],
    #                 ['IT6', 'SOM6'],
    #                 ['CT6', 'SOM6'],
    #                 ['PV6', 'SOM6'],
    #                 ['SOM6', 'SOM6'],
    #                 ['VIP6', 'SOM6'],
    #                 ['NGF6', 'SOM6'],         
    #                 ['IT6', 'PV6'],
    #                 ['CT6', 'PV6']
    #                 ['PV6', 'PV6'],
    #                 ['SOM6', 'PV6'],
    #                 ['VIP6', 'PV6'],
    #                 ['NGF6', 'PV6']
    #                 ['IT6', 'IT6'],
    #                 ['CT6', 'IT6']
    #                 ['PV6', 'IT6'],
    #                 ['SOM6', 'IT6'],
    #                 ['VIP6', 'IT6'],
    #                 ['NGF6', 'IT6'],
    #                 ['IT6', 'CT6'],
    #                 ['CT6', 'CT6']
    #                 ['PV6', 'CT6'],
    #                 ['SOM6', 'CT6'],
    #                 ['VIP6', 'CT6'],
    #                 ['NGF6', 'CT6'],
    #                 ['IT6', 'VIP6'],
    #                 ['CT6', 'VIP6']
    #                 ['PV6', 'VIP6'],
    #                 ['SOM6', 'VIP6'],
    #                 ['VIP6', 'VIP6'],
    #                 ['NGF6', 'VIP6']]        
    
    # only those with pmat > 0.08
    # weightsScale =  [['IT6', 'PV6'], 
    #                 ['IT6', 'SOM6'], 
    #                 ['IT6', 'VIP6'], 
    #                 ['IT6', 'NGF6'], 
    #                 ['CT6', 'PV6'], 
    #                 ['CT6', 'SOM6'], 
    #                 ['CT6', 'VIP6'], 
    #                 ['CT6', 'NGF6'], 
    #                 ['PV6', 'IT6'], 
    #                 ['PV6', 'CT6'], 
    #                 ['PV6', 'PV6'], 
    #                 ['PV6', 'SOM6'], 
    #                 ['PV6', 'VIP6'], 
    #                 ['PV6', 'NGF6'], 
    #                 ['SOM6', 'IT6'], 
    #                 ['SOM6', 'CT6'], 
    #                 ['VIP6', 'PV6'], 
    #                 ['VIP6', 'SOM6'], 
    #                 ['VIP6', 'VIP6'], 
    #                 ['VIP6', 'NGF6'], 
    #                 ['NGF6', 'IT6'], 
    #                 ['NGF6', 'CT6']]


    weightsScale = [['IT2', 'PV2'],
                    ['IT2', 'SOM2'], 
                    ['IT3', 'PV2'],
                    ['IT3', 'SOM2'],
                    ['PV2', 'PV2'],
                    ['PV2', 'VIP2'],
                    ['PV3', 'PV2'],
                    ['PV3', 'VIP2'],
                    ['SOM2', 'PV2'],
                    ['SOM2', 'VIP2'],
                    ['SOM3', 'PV2'],
                    ['SOM3', 'VIP2'],
                    ['VIP2', 'SOM2'],
                    ['VIP3', 'SOM2'],
                    ['IT2', 'SOM3'], 
                    ['IT3', 'SOM3'],
                    ['VIP2', 'SOM3'],
                    ['VIP3', 'SOM3'],
                    
                    ['IT6', 'PV6'], 
                    ['IT6', 'SOM6'], 
                    ['IT6', 'VIP6'], 
                    ['IT6', 'NGF6']]
                    

    for ws in weightsScale:
        params[('wmat', ws[0], ws[1])] = [wmat[ws[0]][ws[1]] * scaleLow, wmat[ws[0]][ws[1]] * scaleHigh]

    weightsScale2 = []

    for ws in weightsScale:
        params[('wmat', ws[0], ws[1])] = [wmat[ws[0]][ws[1]] * scaleLow2, wmat[ws[0]][ws[1]] * scaleHigh2]



    groupedParams = []

    # --------------------------------------------------------
    # initial config
    initCfg = {}
    initCfg = {}
    initCfg['duration'] = 2500
    initCfg['printPopAvgRates'] = [[1500, 1750], [1750, 2000], [2000, 2250], [2250, 2500]]
    initCfg['dt'] = 0.05

    initCfg['scaleDensity'] = 1.0

    # plotting and saving params
    initCfg[('analysis','plotRaster','markerSize')] = 10

    initCfg[('analysis','plotRaster','timeRange')] = [1500, 2500]
    initCfg[('analysis', 'plotTraces', 'timeRange')] = [1500, 2500]
    initCfg[('analysis', 'plotTraces', 'oneFigPer')] = 'trace'
    initCfg['recordLFP'] = None
    initCfg[('analysis', 'plotLFP')] = False

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False


    # from prev - best of 50% cell density
    updateParams = ['EEGain', 'EIGain', 'IEGain', 'IIGain',
                    ('EICellTypeGain', 'PV'), ('EICellTypeGain', 'SOM'), ('EICellTypeGain', 'VIP'), ('EICellTypeGain', 'NGF'),
                    ('IECellTypeGain', 'PV'), ('IECellTypeGain', 'SOM'), ('IECellTypeGain', 'VIP'), ('IECellTypeGain', 'NGF'),
                    ('EILayerGain', '1'), ('IILayerGain', '1'),
                    ('EELayerGain', '2'), ('EILayerGain', '2'),  ('IELayerGain', '2'), ('IILayerGain', '2'), 
                    ('EELayerGain', '3'), ('EILayerGain', '3'), ('IELayerGain', '3'), ('IILayerGain', '3'), 
                    ('EELayerGain', '4'), ('EILayerGain', '4'), ('IELayerGain', '4'), ('IILayerGain', '4'), 
                    ('EELayerGain', '5A'), ('EILayerGain', '5A'), ('IELayerGain', '5A'), ('IILayerGain', '5A'), 
                    ('EELayerGain', '5B'), ('EILayerGain', '5B'), ('IELayerGain', '5B'), ('IILayerGain', '5B'), 
                    ('EELayerGain', '6'), ('EILayerGain', '6'), ('IELayerGain', '6'), ('IILayerGain', '6')] 

    for p in updateParams:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})

    # good thal params for 100% cell density 
    updateParams2 = ['thalamoCorticalGain', 'intraThalamicGain', 'EbkgThalamicGain', 'IbkgThalamicGain']

    for p in updateParams2:
        if isinstance(p, tuple):
            initCfg.update({p: cfgLoad[p[0]][p[1]]})
        else:
            initCfg.update({p: cfgLoad[p]})

    print(initCfg)


    # --------------------------------------------------------
    # fitness function
    fitnessFuncArgs = {}
    pops = {}
    
    ## Exc pops
    Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6', 'TC', 'TCM', 'HTC']  # all layers + thal + IC
    # Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'TC', 'TCM', 'HTC']  # all layers + thal + IC

    Etune = {'target': 5, 'width': 20, 'min': 0.05}
    #Etune = {'target': 5, 'width': 5, 'min': 0.5}
    
    for pop in Epops:
        pops[pop] = Etune
    
    ## Inh pops 
    # Ipops = ['NGF1',                            # L1
    #         'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
    #         'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
    #         'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
    #         'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
    #         'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
    #         'PV6', 'SOM6', 'VIP6', 'NGF6',       # L6
    #         'IRE', 'IREM', 'TI']  # Thal 
    Ipops = ['NGF1',  
            'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
            'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
            'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
            'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A 
            'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
            'PV6', 'SOM6', 'VIP6', 'NGF6',  # L6
            'IRE', 'IREM', 'TI']  # Thal 

    Itune = {'target': 10, 'width': 30, 'min': 0.05}
    #Itune = {'target': 10, 'width': 15, 'min': 0.5}

    for pop in Ipops:
        pops[pop] = Itune
    
    fitnessFuncArgs['pops'] = pops
    fitnessFuncArgs['maxFitness'] = 2000
    fitnessFuncArgs['tranges'] = initCfg['printPopAvgRates']


    def fitnessFunc(simData, **kwargs):
        import numpy as np
        pops = kwargs['pops']
        maxFitness = kwargs['maxFitness']
        tranges = kwargs['tranges']

        popFitnessAll = []

        for trange in tranges:
            popFitnessAll.append([min(np.exp(abs(v['target'] - simData['popRates'][k]['%d_%d'%(trange[0], trange[1])])/v['width']), maxFitness) 
                if simData['popRates'][k]['%d_%d'%(trange[0], trange[1])] >= v['min'] else maxFitness for k, v in pops.items()])
        
        popFitness = np.mean(np.array(popFitnessAll), axis=0)
        
        fitness = np.mean(popFitness)

        popInfo = '; '.join(['%s rate=%.1f fit=%1.f' % (p, np.mean(list(simData['popRates'][p].values())), popFitness[i]) for i,p in enumerate(pops)])
        print('  ' + popInfo)

        return fitness
        
    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)

    b.method = 'optuna'

    b.optimCfg = {
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'maxFitness': fitnessFuncArgs['maxFitness'],
        'maxiters':     1e6,    #    Maximum number of iterations (1 iteration = 1 function evaluation)
        'maxtime':      None,    #    Maximum time allowed, in seconds
        'maxiter_wait': 60,
        'time_sleep': 150,
        'popsize': 1  # unused - run with mpi 
    }

    return b

# ----------------------------------------------------------------------------------------------
# Run configurations
# ----------------------------------------------------------------------------------------------
def setRunCfg(b, type='mpi_direct'):
    if type=='mpi_bulletin':
      b.runCfg = {'type': 'mpi_bulletin', 
                  'script': 'init.py', 
                  'skip': True}
    elif type=='mpi_direct':
        b.runCfg = {'type': 'mpi_direct',
            'nodes': 1,
            'coresPerNode': 48,
            'script': 'init.py',
            'mpiCommand': 'mpiexec',
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
            'email': 'samnemo@gmail.com',
            'folder': '/home/salvadord/m1/sim/',  # comet='/salvadord', bridges='/salvi82'
            'script': 'init.py', 
            'mpiCommand': 'ibrun', # comet='ibrun', bridges='mpirun'
            'skipCustom': '_raster.png'}
    elif type=='hpc_slurm_gcp':
        b.runCfg = {'type': 'hpc_slurm', 
            'allocation': 'default', # bridges='ib4iflp', comet m1='shs100', comet nsg='csd403', gcp='default'
            'walltime': '24:00:00', #'48:00:00',
            'nodes': 1,
            'coresPerNode': 80,  # comet=24, bridges=28, gcp=32
            'email': 'samnemo@gmail.com',
            'folder': '/home/ext_samnemo_gmail_com/A1/',  # comet,gcp='/salvadord', bridges='/salvi82'
            'script': 'init.py',
            'mpiCommand': 'mpirun', # comet='ibrun', bridges,gcp='mpirun' 
            'nrnCommand': 'nrniv -mpi -python', #'python3',
            'skipCustom': '_raster.png'}
            #'custom': '#SBATCH --exclude=compute[17-64000]'} # only use first 16 nodes (non-preemptible for long runs )
            # --nodelist=compute1
    elif type=='hpc_slurm_bridges':
        b.runCfg = {'type': 'hpc_slurm', 
            'allocation': 'ib4iflp', # bridges='ib4iflp', comet m1='shs100', comet nsg='csd403'
            'walltime': '06:00:00',
            'nodes': 2,
            'coresPerNode': 28,  # comet=24, bridges=28
            'email': 'samnemo@gmail.com',
            'folder': '/home/salvi82/m1/sim/',  # comet='/salvadord', bridges='/salvi82'
            'script': 'init.py', 
            'mpiCommand': 'mpirun', # comet='ibrun', bridges='mpirun'
            'skip': True}
    elif type=='hpc_slurm_cineca':         ## FILL THIS IN
        b.runCfg = {'type': 'hpc_slurm',
            'allocation': 'icei_H_King',
            'walltime': '1:30:00',            # g100_qos_dbg : 2 hrs           # noQOS: 24 hrs 
            'nodes': 4,                       # g100_qos_dbg : max 2 nodes     # noQOS: max 32 nodes 
            'coresPerNode': 24, #48,               # g100_qos_dbg : nodes*coresPerNode = 96 MAX       
            'partition': 'g100_usr_prod',
            'qos': None, #'noQOS',           # g100_qos_dbg  # noQOS
            'email': 'erica.griffith@downstate.edu',
            'folder': '/g100/home/userexternal/egriffit/A1/',
            'script': 'init.py',
            'mpiCommand': 'srun', 
            'skip': True}   # --cpu-bind=cores -m block:block' # mpirun   # TRY SRUN
            # 'nrnCommand': ,  # 'nrniv -mpi -python', #'python3',
            # 'skip': ,
            # 'skipCustom': ,
        #}
    elif type=='hpc_slurm_xsede':
        b.runCfg = {'type': 'hpc_slurm', 
            'allocation': 'TG-IBN160014', 
            'walltime': '12:00:00',
            'nodes': 1, # 4
            'coresPerNode': 80,#48,
            'partition': 'skx-normal', 
            'email': 'erica.griffith@downstate.edu',
            'folder': '/home1/06490/tg857900/A1/',
            'script': 'init.py', 
            'mpiCommand': 'mpirun',
            'nrnCommand': 'nrniv -mpi -python3',
            'skip': True}
            #'custom': '#SBATCH --exclude=compute[17-64000]'} # only use first 16 nodes (non-preemptible for long runs )
            # --nodelist=compute1

# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':
    from utils import getdatestr
    b = optunaERP()
    b.batchLabel = 'optunaERP_'+getdatestr()
    b.saveFolder = 'data/'+b.batchLabel
    setRunCfg(b, 'mpi_direct')
    print('running batch...')
    b.run() # run batch
    

'''
Macaque A1 model 
Local connectivity preprocessing script

- Loads experimental data from file
- Preprocesses so ready to use by NetPyNE high-level specs

'''
import numpy as np
import json
import csv

# ----------------------------------------------------------------------------------------------------------------
# Func to load data from published studies
# ----------------------------------------------------------------------------------------------------------------
def loadData():
    data = {'test': {'prob': 0.2, 'weight': 0.5}}  # oversimplified example -- will have data from different papers for differnet projections 
    
    # ----------------------------------------------------------------------------------------------------------------
    # load and pre-process Allen mouse V1 data (Billeh et al, 2019; https://www.dropbox.com/sh/xb7xasih3d8027u/AAAbKXe0Zmk86o3_y1iPVPCLa?dl=0)
    data['Allen_V1'] = {}
    
    ## load conn probs
    with open('../data/conn/Allen_V1_conn_probs.json', 'r') as f:
        data['Allen_V1']['connProb'] = json.load(f)

    ## func to calculate probConn at distance = 0um based on probConn at distance = 75 and lambda (length constant)
    ## adapted from Allen V1 code in Allen_V1_connect_cells.py

    for proj in data['Allen_V1']['connProb'].keys():

        # A_literature is different for every source-target pair and was estimated from the literature.
        A_literature = data['Allen_V1']['connProb'][proj]['A_literature']

        # R0 read from the dictionary, but setting it now at 75um for all cases but this allows us to change it
        R0 = data['Allen_V1']['connProb'][proj]['R0']

        # Sigma is measure from the literature or internally at the Allen Institute
        sigma = data['Allen_V1']['connProb'][proj]['sigma']

        # Gaussian equation was intergrated to and solved to calculate new A0 (amplitude at distance=0)
        A0 = A_literature / ((sigma / R0)** 2 * (1 - np.exp(-(R0 / sigma)** 2)))
        if A0 > 1.0: A0 = 1.0  # make max prob = 1.0
        data['Allen_V1']['connProb'][proj]['A0'] = A0


    ## load conn weights (soma PSP amp in mV)
    data['Allen_V1']['connWeight'] = {}
    with open('../data/conn/Allen_V1_conn_weights.csv', 'r') as f:
        csv_reader = csv.reader(f)
        for irow,row in enumerate(csv_reader):
            if irow == 0:
                headings = row
            else:
                for ih, h in enumerate(headings[1:]):
                    try:
                        data['Allen_V1']['connWeight'][row[0] + '-' + h] = float(row[ih + 1])
                    except:
                        data['Allen_V1']['connWeight'][row[0] + '-' + h] = 0.0
                
    
    # set correspondence between A1 pops and Allen V1 pops 
    data['Allen_V1']['pops'] = {
        'NGF1': 'i1H',                                                                              # L1
        'IT2': 'e2',                'PV2': 'i2P',   'SOM2': 'i2S',  'VIP2': 'i2H',  'NGF2': 'i2H', # L2
        'IT3': 'e2',                'PV3': 'i2P',   'SOM3': 'i2S',  'VIP3': 'i2H',  'NGF3': 'i2H',  # L3
        'ITP4': 'e4', 'ITS4': 'e4', 'PV4': 'i4P',   'SOM4': 'i4S',  'VIP4': 'i4H',  'NGF4': 'i4H',  # L4
        'IT5A': 'e5',               'PV5A': 'i5P',  'SOM5A': 'i5S', 'VIP5A': 'i5H', 'NGF5A': 'i5H', # L5A
        'IT5B': 'e5', 'PT5B': 'e5', 'PV5B': 'i5P',  'SOM5B': 'i5S', 'VIP5B': 'i5H', 'NGF5B': 'i5H', # L5B
        'IT6': 'e6',  'CT6': 'e6',  'PV6': 'i6P',   'SOM6': 'i6S',  'VIP6': 'i6H',  'NGF6': 'i6H'}  # L6


    # ----------------------------------------------------------------------------------------------------------------
    # load and pre-process BBP mouse S1 data (Markram et al, 2015; https://bbp.epfl.ch/nmc-portal/downloads
    data['BBP_S1'] = {}
    with open('../data/conn/BBP_S1_pathways_anatomy_factsheets_simplified.json', 'r') as f:
        data['BBP_S1']['connProb'] = json.load(f)

    with open('../data/conn/BBP_S1_pathways_physiology_factsheets_simplified.json', 'r') as f:
        data['BBP_S1']['connWeight'] = json.load(f)
    
    return data

# ----------------------------------------------------------------------------------------------------------------
# Params
# ----------------------------------------------------------------------------------------------------------------
Etypes = ['IT', 'ITS4', 'PT', 'CT']
Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT6', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6']  # all layers

Itypes = ['PV', 'SOM', 'VIP', 'NGF']
Ipops = ['NGF1',                            # L1
        'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
        'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
        'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
        'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
        'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
        'PV6', 'SOM6', 'VIP6', 'NGF6']  # L6 
        
TCpops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM']

layer = {'1': [0.00, 0.05], '2': [0.05, 0.08], '3': [0.08, 0.475], '4': [0.475,0.625], '5A': [0.625,0.667], '5B': [0.667,0.775], '6': [0.775,1], 'thal': [1.2, 1.4]} # 



# initialize prob and weight matrices
# format: pmat[presynaptic_pop][postsynaptic_pop] 
pmat = {}  # probability of connection matrix
lmat = {}  # length constant (lambda) for exp decaying prob conn (um) matrix
wmat = {}  # connection weight matrix = unitary conn somatic PSP (mV)
for p in Epops + Ipops + TCpops:
    pmat[p] = {}
    lmat[p] = {}
    wmat[p] = {}
    
# Load exp data
data = loadData()


# --------------------------------------------------
## E -> E 
# --------------------------------------------------

# --------------------------------------------------
## Probabilities, length constants (lambda), and weights (=unitary conn somatic PSP amplitude)

'''
Probabilities are distance dependent based on:
A0 * np.exp(- (intersomatic_distance / lambda) ** 2)

where A0 is the probability of connection and distance 0um
and lambda is the length constant
'''

# start with base data from Allen V1
for pre in Epops:
    for post in Epops:
        proj = '%s-%s' % (data['Allen_V1']['pops'][pre], data['Allen_V1']['pops'][post])
        pmat[pre][post] = data['Allen_V1']['connProb'][proj]['A0']
        lmat[pre][post] = data['Allen_V1']['connProb'][proj]['sigma']
        wmat[pre][post] = data['Allen_V1']['connWeight'][proj]

# use BBP S1 instead? (has more cell-type specificity)

# modify based on A1 exp data - TO DO


# --------------------------------------------------
## E -> I
# --------------------------------------------------

# --------------------------------------------------
## Probabilities, length constants (lambda), and weights (=unitary conn somatic PSP amplitude)

'''
Probabilities are distance dependent based on:
A0 * np.exp(- (intersomatic_distance / lambda) ** 2)

where A0 is the probability of connection and distance 0um
and lambda is the length constant
'''

# start with base data from Allen V1
for pre in Epops:
    for post in Ipops:
        proj = '%s-%s' % (data['Allen_V1']['pops'][pre], data['Allen_V1']['pops'][post])
        pmat[pre][post] = data['Allen_V1']['connProb'][proj]['A0']
        lmat[pre][post] = data['Allen_V1']['connProb'][proj]['sigma']
        wmat[pre][post] = data['Allen_V1']['connWeight'][proj]

# use BBP S1 instead? (has more cell-type specificity)

# modify based on A1 exp data - TO DO
    

# --------------------------------------------------
## I -> E
# --------------------------------------------------

bins = {}
bins['inh'] = [[0.0, 0.37], [0.37, 0.8], [0.8,1.0]]

# --------------------------------------------------
## Probabilities 

'''
- I->E/I connections (Tremblay, 2016: Sohn, 2016; Budinger, 2018; Naka et al 2016; Garcia, 2015)
- Local, intralaminar only; all-to-all but distance-based; high weights
- Although evidence for L2/3,4,6 -> L5A/B, strongest is intralaminar (Naka16)
- Consistent with Allen V1, except Allen doesn't show strong L2/3 I -> L5 E
'''

# I->E particularities:
## inh cells target apical dends (not strictly local to layer) 
## L1 + L2/3 NGF -> L2/3 E (prox apic) + L5 E (tuft) -- specify target dend in subConnParams
## upper layer SOM, VIP, NGF project strongly to deeper E cells (Kato 2017) with exp-decay distance-dep

for pre in ['SOM', 'VIP', 'NGF']:
    pmat[pre] = {}
    pmat[pre]['E'] = np.array([[1.0, 1.0, 1.0],  # from L1+L2/3+L4 to all layers 
                                 [0.25, 1.0, 1.0],  # from L5A+L5B to all layers
                                 [0.25, 0.25, 1.0]])  # from L6 to all layers

## upper layer PV project weakly to deeper E cells (mostly intralaminar) (Kato 2017) with exp-decay distance-dep
## although Naka 2016 shows L2/3 PV -> L5 Pyr
pmat['PV'] = {}
pmat['PV']['E'] = np.array([[1.0, 0.5, 0.25],  # from L1+L2/3+L4 to all layers 
                              [0.25, 1.0, 0.5],  # from L5A+L5B to all layers
                              [0.1, 0.25, 1.0]])  # from L6 to all layers

# VIP -> E (very low; 3/42; Pi et al 2013) 
pmat['VIP']['E'] *= 0.1

# --------------------------------------------------
## Weights  (=unitary conn somatic PSP amplitude)
IEweight = 1.0
for pre in Ipops:
    for post in Ipops:
        wmat[pre][post] = IEweight


# --------------------------------------------------
## I -> I
# --------------------------------------------------

# --------------------------------------------------
## Probabilities 

'''
- NGF -> all I, local/intralaminar (+ distance-dep)
- VIP -> SOM (strong; 14/18), PV (weak; 4/15), VIP (very weak -- remove?) (Pi 2013)
- SOM -> FS+VIP (strong); SOM (weak -- remove?)  (Naka et al 2016;Tremblay, 2016; Sohn, 2016)
- PV  -> PV (strong); SOM+VIP (weak -- remove?) (Naka et al 2016;Tremblay, 2016; Sohn, 2016)
- Generally consistent with the more detailed Allen V1 I->I
'''

### I->I particularities
for pre in Itypes:
    pmat[pre]
    for post in Itypes:
        pmat[pre][post] = 1.0

# NGF -> all I, local/intralaminar (+ distance-dep)
# no change required

strong = 1.0
weak = 0.35  # = ~ normalized strong/weak = (4/15) / (14/18)
veryweak = 0.1

# VIP -> SOM (strong; 14/18), PV (weak; 4/15), VIP (very weak -- remove?) (Pi 2013)
pmat['VIP']['SOM'] = strong
pmat['VIP']['PV'] = weak
pmat['VIP']['VIP'] = veryweak
pmat['VIP']['NGF'] = weak  # unknown; assume weak

# SOM -> FS+VIP (strong); SOM (weak -- remove?)  (Naka et al 2016;Tremblay, 2016; Sohn, 2016)
pmat['SOM']['PV'] = strong
pmat['SOM']['VIP'] = strong
pmat['SOM']['SOM'] = weak
pmat['VIP']['NGF'] = weak  # unknown; assume weak
 
# PV  -> PV (strong); SOM+VIP (weak -- remove?) (Naka et al 2016;Tremblay, 2016; Sohn, 2016)
pmat['PV']['PV'] = strong
pmat['PV']['SOM'] = weak
pmat['PV']['VIP'] = weak
pmat['PV']['NGF'] = weak  # unknown; assume weak


# --------------------------------------------------
## Weights  (=unitary conn somatic PSP amplitude)
IIweight = 1.0
for pre in Ipops:
    for post in Ipops:
        wmat[pre][post] = IIweight


# --------------------------------------------------
# Delays
## Make distance-dependent for now


# --------------------------------------------------
## INTRATHALAMIC (from old model; partly from Bazhenov https://www.jneurosci.org/content/32/15/5250.full and discuss with Lakatos)
# --------------------------------------------------

# --------------------------------------------------
## Probabilities 
pmat['TC']['TC'] =	    0.1
pmat['HTC']['HTC'] =	0.1
pmat['TC']['HTC'] =	    0.1
pmat['HTC']['TC'] =	    0.1
pmat['TCM']['TCM'] =	0.1
pmat['IRE']['IRE'] =	0.1
pmat['IREM']['IREM'] =	0.1
pmat['IRE']['IREM'] =	0.1
pmat['IREM']['IRE'] =	0.1
pmat['TC']['IREM'] =	0.2
pmat['HTC']['IREM'] =	0.2
pmat['IREM']['TC'] =	0.1
pmat['IREM']['HTC'] =	0.1
pmat['TCM']['IRE'] =	0.2
pmat['IRE']['TCM'] =	0.1
pmat['TC']['IRE'] =	    0.4
pmat['HTC']['IRE'] =	0.4
pmat['IRE']['TC'] =	    0.3
pmat['IRE']['HTC'] =	0.3
pmat['TCM']['IREM'] =	0.4
pmat['IREM']['TCM'] =	0.3

# --------------------------------------------------
## Weights  (=unitary conn somatic PSP amplitude)
wmat['HTC']['HTC'] =    0.1
wmat['HTC']['TC'] =     0.1
wmat['TC']['HTC'] =     0.1
wmat['TC']['TC'] =  	0.1
wmat['TCM']['TCM'] =    0.1
wmat['IRE']['IRE'] =    1.5
wmat['IREM']['IREM'] =  1.5
wmat['IRE']['IREM'] =   1.5
wmat['IREM']['IRE'] =   1.5
wmat['TC']['IREM'] =    0.23
wmat['HTC']['IREM'] =   0.123
wmat['IREM']['TC'] =    0.83
wmat['IREM']['HTC'] =   0.83
wmat['TCM']['IRE'] =    0.2
wmat['IRE']['TCM'] =    0.83
wmat['TC']['IRE'] = 	0.2
wmat['HTC']['IRE'] =    0.2
wmat['IRE']['TC'] =     0.83
wmat['IRE']['HTC'] =    0.83
wmat['TCM']['IREM'] =   0.2
wmat['IREM']['TCM'] =   0.83



# --------------------------------------------------
## CORTICOTHALAMIC (from old model; partly from Bazhenov https://www.jneurosci.org/content/32/15/5250.full and discuss with Lakatos)
# --------------------------------------------------

# --------------------------------------------------
## Probabilities 
pmat['CT6']['TC']	= 0.1
pmat['CT6']['HTC']	= 0.1
pmat['CT6']['IRE']	= 0.1
pmat['IT5B']['TCM']	= 0.1
pmat['PT5B']['TCM']	= 0.1

# --------------------------------------------------
## Weights  (=unitary conn somatic PSP amplitude)
wmat['CT6']['TC']	= 0.7
wmat['CT6']['HTC']	= 0.7
wmat['CT6']['IRE']	= 0.23
wmat['IT5B']['TCM']	= 0.7
wmat['PT5B']['TCM']	= 0.7


# --------------------------------------------------
## CORE THALAMOCORTICAL (from old model; partly from Bazhenov https://www.jneurosci.org/content/32/15/5250.full and discuss with Lakatos)
# --------------------------------------------------

# --------------------------------------------------
## Probabilities 

# note for I cells target PV, SOM and NGF for now
pmat['TC']['IT4']    = 0.25
pmat['HTC']['IT4'] = 0.25
pmat['TC']['IT4S']    = 0.25
pmat['HTC']['IT4S']   = 0.25
pmat['TC']['PT5B']   = 0.1   #*thalfctr
pmat['HTC']['PT5B']  = 0.1   #*thalfctr
pmat['TC']['IT5A']   = 0.1   #*thalfctr
pmat['HTC']['IT5A']  = 0.1   #*thalfctr
pmat['TC']['IT5B']   = 0.1   #*thalfctr
pmat['HTC']['IT5B']  = 0.1   #*thalfctr
pmat['TC']['IT6']    = 0.15  #*thalfctr
pmat['HTC']['IT6']   = 0.15  #*thalfctr
pmat['TC']['CT6']    = 0.15  #*thalfctr
pmat['HTC']['CT6']   = 0.15  #*thalfctr
pmat['TC']['PV4']    = 0.25
pmat['HTC']['PV4']   = 0.25
pmat['TC']['SOM4']    = 0.25
pmat['HTC']['SOM4']   = 0.25
pmat['TC']['PV5A']    = 0.1   #*thalfctr
pmat['HTC']['PV5A']   = 0.1   #*thalfctr
pmat['TC']['SOM5A']   = 0.1   #*thalfctr
pmat['HTC']['SOM5A']   = 0.1  #*thalfctr
pmat['TC']['PV5B']    = 0.1   #*thalfctr
pmat['HTC']['PV5B']   = 0.1   #*thalfctr
pmat['TC']['SOM5B']   = 0.1   #*thalfctr
pmat['HTC']['SOM5B']   = 0.1   #*thalfctr
pmat['TC']['PV6']     = 0.15  #*thalfctr
pmat['HTC']['PV6']   = 0.15  #*thalfctr
pmat['TC']['SOM6']     = 0.15  #*thalfctr
pmat['HTC']['SOM6']   = 0.15  #*thalfctr


# --------------------------------------------------
## Weights  (=unitary conn somatic PSP amplitude)
wmat['TC']['PT5B']   =	0.6  #* pmat[TC][E5B] / pmat[TC][E4]	
wmat['HTC']['PT5B'] = 0.6  #* pmat[TC][E5B] / pmat[TC][E4]
wmat['TC']['IT5A']   =	0.6  #* pmat[TC][E5R] / pmat[TC][E4]	
wmat['HTC']['IT5A']  =	0.6  #* pmat[TC][E5R] / pmat[TC][E4]	
wmat['TC']['IT5B']   =	0.6  #* pmat[TC][E5R] / pmat[TC][E4]	
wmat['HTC']['IT5B']  =	0.6  #* pmat[TC][E5R] / pmat[TC][E4]	
wmat['TC']['IT6']    =	0.6  #* pmat[TC][E6] / pmat[TC][E4]	
wmat['HTC']['IT6'] = 0.6  #* pmat[TC][E6] / pmat[TC][E4]	
wmat['TC']['CT6']    =	0.6  #* pmat[TC][E6] / pmat[TC][E4]
wmat['HTC']['CT6']   =	0.6  #* pmat[TC][E6] / pmat[TC][E4]	
wmat['TC']['PV4']    =	0.23 #	
wmat['HTC']['PV4']   =	0.23 #	
wmat['TC']['SOM4']    =	0.23 #	
wmat['HTC']['SOM4']   =	0.23 #	
wmat['TC']['NGF4']    =	0.23 #	
wmat['HTC']['NGF4']   =	0.23 #	
wmat['TC']['PV5A']    =	0.23 # * pmat[TC][I5] / pmat[TC][I4]	
wmat['HTC']['PV5A']   =	0.23 # * pmat[TC][I5] / pmat[TC][I4]	
wmat['TC']['SOM5A']    =	0.23 # * pmat[TC][I5] / pmat[TC][I4]	
wmat['HTC']['SOM5A'] = 0.23  # * pmat[TC][I5] / pmat[TC][I4]	
wmat['TC']['NGF5A']    =	0.23 # * pmat[TC][I5] / pmat[TC][I4]	
wmat['HTC']['NGF5A'] = 0.23  # * pmat[TC][I5] / pmat[TC][I4]	
wmat['TC']['PV5B']    =	0.23 # * pmat[TC][I5] / pmat[TC][I4]	
wmat['HTC']['PV5B']   =	0.23 # * pmat[TC][I5] / pmat[TC][I4]	
wmat['TC']['SOM5B']    =	0.23 # * pmat[TC][I5] / pmat[TC][I4]	
wmat['HTC']['SOM5B'] = 0.23  # * pmat[TC][I5] / pmat[TC][I4]	
wmat['TC']['NGF5B']    =	0.23 # * pmat[TC][I5] / pmat[TC][I4]	
wmat['HTC']['NGF5B']   =	0.23 # * pmat[TC][I5] / pmat[TC][I4]	
wmat['TC']['PV6']    =	0.23 # * pmat[TC][I6] / pmat[TC][I4]	
wmat['HTC']['PV6']   =	0.23 # * pmat[TC][I6] / pmat[TC][I4]	
wmat['TC']['SOM6']    =	0.23 # * pmat[TC][I6] / pmat[TC][I4]	
wmat['HTC']['SOM6']   =	0.23 # * pmat[TC][I6] / pmat[TC][I4]
wmat['TC']['NGF6']    =	0.23 # * pmat[TC][I6] / pmat[TC][I4]	
wmat['HTC']['NGF6']   =	0.23 # * pmat[TC][I6] / pmat[TC][I4]


# --------------------------------------------------
## MATRIX THALAMOCORTICAL (from old model; partly from Bazhenov https://www.jneurosci.org/content/32/15/5250.full and discuss with Lakatos)
# --------------------------------------------------

# --------------------------------------------------
## Probabilities 

# note for I cells target PV, SOM and NGF for now
pmat['TCM']['IT2']	= 0.25
pmat['TCM']['IT3']	= 0.25
pmat['TCM']['IT5A']	= 0.15  #* thalfctr
pmat['TCM']['IT5B']	= 0.15  #* thalfctr
pmat['TCM']['PT5B']	= 0.15  #* thalfctr
pmat['TCM']['IT6']	= 0.05  #* thalfctr
pmat['TCM']['CT6'] = 0.05  #* thalfctr

pmat['TCM']['NGF1']	= 0.25
pmat['TCM']['PV2']	= 0.25
pmat['TCM']['SOM2']	= 0.25
pmat['TCM']['NGF2']	= 0.25
pmat['TCM']['PV3']	= 0.25
pmat['TCM']['SOM3']	= 0.25
pmat['TCM']['NGF3']	= 0.25

pmat['TCM']['PV5A']	= 0.15  #* thalfctr
pmat['TCM']['SOM5A']	= 0.15  #* thalfctr
pmat['TCM']['SOM5B']	= 0.15  #* thalfctr
pmat['TCM']['PV5B']	= 0.15  #* thalfctr
pmat['TCM']['SOM5B']	= 0.15  #* thalfctr
pmat['TCM']['NGF5B']	= 0.15  #* thalfctr

pmat['TCM']['PV6']	= 0.05  #* thalfctr
pmat['TCM']['SOM6']	= 0.05  #* thalfctr
pmat['TCM']['NGF6']	= 0.05  #* thalfctr


# --------------------------------------------------
## Weights  (=unitary conn somatic PSP amplitude)

wmat['TCM']['IT2']	= 0.6
wmat['TCM']['IT3']	= 0.6
wmat['TCM']['IT5A']	= 0.6  #* thalfctr
wmat['TCM']['IT5B']	= 0.6  #* thalfctr
wmat['TCM']['PT5B']	= 0.6  #* thalfctr
wmat['TCM']['IT6']	= 0.6  #* thalfctr
wmat['TCM']['CT6'] = 0.6  #* thalfctr

wmat['TCM']['NGF1']	= 0.25
wmat['TCM']['PV2']	= 0.25
wmat['TCM']['SOM2']	= 0.25
wmat['TCM']['NGF2']	= 0.25
wmat['TCM']['PV3']	= 0.25
wmat['TCM']['SOM3']	= 0.25
wmat['TCM']['NGF3']	= 0.25

wmat['TCM']['PV5A']  = 0.25 #* thalfctr
wmat['TCM']['SOM5A'] = 0.25 #* thalfctr
wmat['TCM']['SOM5B'] = 0.25 #* thalfctr
wmat['TCM']['PV5B']	 = 0.25 #* thalfctr
wmat['TCM']['SOM5B'] = 0.25 #* thalfctr
wmat['TCM']['NGF5B'] = 0.25 #* thalfctr

wmat['TCM']['PV6']	 = 0.25  #* thalfctr
wmat['TCM']['SOM6']	 = 0.25  #* thalfctr
wmat['TCM']['NGF6']	 = 0.25  #* thalfctr

# --------------------------------------------------
# Save data to pkl file
savePickle = 1

if savePickle:
    import pickle
    with open('conn.pkl', 'wb') as f:
        pickle.dump({'pmat': pmat, 'lmat': lmat, 'wmat': wmat, 'bins': bins}, f)
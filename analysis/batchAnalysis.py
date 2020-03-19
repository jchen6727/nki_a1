"""
batchAnalysis.py 

Code to anlayse batch sim results

Contributors: salvadordura@gmail.com
"""

#import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers

#from batchAnalysisFilter import *
from batchAnalysisPlotSingle import *
#from batchAnalysisPlotCombined import *


# Main code
if __name__ == '__main__': 
    dataFolder = '../data/'
    batchLabel = 'v11_batch7' # 'v50_batch1' #
    loadAll = 0

    allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF4', 'ITP4', 'ITS4', 'VIP4', 'IT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM']

    alltypes = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'ITS4', 'PT5B', 'TC', 'HTC', 'IRE']

    # ---------------------------------------------
    # Filtering wrapper funcs
    # ---------------------------------------------
    
    #applyFilterRates(dataFolder, batchLabel, loadAll, skipDepol=0)
       
    #df = filterDepolBlock(dataFolder, batchLabel, loadAll, gids=[2931, 5149, 5180, 5234, 5523, 5709])
    
    # filterStimRates(dataFolder, batchLabel, load=loadAll)


    # ---------------------------------------------
    # Single sim plot funcs
    # ---------------------------------------------

    include = {}
    include['raster'] = allpops
    include['traces'] = [0,100, 1285, 1787,2422, 2464, 2470,30,97,98] #[(pop,0) for pop in alltypes]  # 0,100, 1285, 1787,2422, 2464, 2470,30,97,98
    sim,data = loadPlot(dataFolder, batchLabel, include=include) 
                                                            
    #plotConnFile(dataFolder, batchLabel)

    #plotConnDisynFile(dataFolder, batchLabel)


    # ---------------------------------------------
    # Combined sim (batch) funcs
    # ---------------------------------------------

    #fIAnalysis(dataFolder, batchLabel, loadAll)

    #df = popRateAnalysis(dataFolder, batchLabel, loadAll, pars=['IClamp1_amp', 'ihGbar', 'gpas', 'epas'], vals='PT5B', plotLine=False)

    # df = ihEPSPAnalysis(dataFolder, batchLabel, loadAll, pars=['groupWeight','ihGbar'], vals=['Vpeak_PTih'], zdcomp=0, plotLine=1)#, \
    # query = 'epas == 1.0 and groupWeight > 0.0003')# and axonNa==7 and gpas==0.65') #'ihLkcBasal == 0.01 and excTau2Factor==1.0') #, 'excTau2Factor', 'ihLkcBasal

    # df = ihEPSPAnalysis(dataFolder, batchLabel, loadAll, pars=['groupWeight', 'ihGbar'], vals=['Vpeak_PTih'], plotLine=True, \
    #    query = 'epas==1.0 and gpas==0.8') #, 'excTau2Factor', 'ihLkcBasal

    # extractRatePSD(dataFolder, batchLabel)

    #extractRates(dataFolder, batchLabel)

    # plotSeedsRates(dataFolder, batchLabel)

    #df = plotSimultLongRates(dataFolder, batchLabel)

    #fig, allSignal = freqAnalysis(dataFolder, batchLabel)
        

    # ---------------------------------------------
    # Comparing funcs
    # ---------------------------------------------

    #utils.compare(dataFolder+'v52_batch11/v52_batch11_0_0.json', dataFolder+'v52_batch12/v52_batch12_0_1_0.json', source_key='simConfig', target_key='simConfig')

    #utils.compare(dataFolder+'../sim/net1.json', dataFolder+'../sim/net2.json')#, source_key='net', target_key='net')

    




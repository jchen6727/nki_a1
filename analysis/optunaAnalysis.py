# analyzeEvol.py 

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import csv
import json
import seaborn as sns
import IPython as ipy
import os
import utils
from pprint import pprint
#import optuna
import pickle

def getParamLabels(dataFolder, batchSim):
    # get param labels
    with open('%s/%s/%s_batch.json' % (dataFolder, batchSim, batchSim), 'r') as f: 
        paramLabels = [str(x['label'][0])+str(x['label'][1]) if isinstance(x['label'], list) else str(x['label']) for x in json.load(f)['batch']['params']]
    return paramLabels

def loadData(dataFolder, batchSim, pops, loadStudyFromFile=False, loadDataFromFile=False):
    if loadDataFromFile:
        with open('%s/%s/%s_df.pkl' % (dataFolder, batchSim, batchSim), 'rb') as f:
            df = pickle.load(f)
    else: 
        if loadStudyFromFile:
            with open('%s/%s/%s_df_study.pkl' % (dataFolder, batchSim, batchSim), 'rb') as f:
                df = pickle.load(f)
        else:
            study = optuna.create_study(study_name=batchSim, storage='sqlite:///%s/%s/%s_storage.db' % (dataFolder, batchSim, batchSim), load_if_exists=True) 
            df = study.trials_dataframe(attrs=('number', 'value', 'params'))

            # replace column labels
            for col in df.columns:
                if col.startswith('params_'):
                    newName = col.replace('params_', '').replace('(', '').replace(')', '').replace("'", "").replace(', ','')
                    df = df.rename(columns={col: newName})

            with open('%s/%s/%s_df_study.pkl' % (dataFolder, batchSim, batchSim), 'wb') as f:
                pickle.dump(df, f)


        # load json for each trial with pop rates and add to df
        popRates = {p: [] for p in pops}

        for i in df.number:
            try:
                with open('%s/%s/trial_%d/trial_%d.json' % (dataFolder, batchSim, int(i), int(i)), 'r') as f:
                    popRatesLoad = json.load(f)['simData']['popRates']
                    
                    for p in popRatesLoad:
                        popRates[p].append(np.mean(list(popRatesLoad[p].values())))

                print('Added trial %d' % (i))
            except:
                for p in popRates:
                    popRates[p].append(0.0)
                print('Skipped trial %d' % (i))
            
        for p, rates in popRates.items():
            df.insert(len(df.columns), p, rates)

        with open('%s/%s/%s_df.pkl' % (dataFolder, batchSim, batchSim), 'wb') as f:
            pickle.dump(df, f)

    return df

def plotScatterPopVsParams(dataFolder, batchsim, df, pops):

    dfcorr=df.corr('pearson')
    
    for pop in pops:

        for param in df.columns:
            try:
                print('Plotting scatter of %s vs %s param (R=%.2f) ...' %(pop, param, dfcorr[pop][param]))
                df.plot.scatter(param, pop, s=4, c='number', colormap='viridis', alpha=0.5, figsize=(8, 8), colorbar=False)
                plt.title('%s vs %s R=%.2f' % (pop, param, dfcorr[pop][param]))
                plt.savefig('%s/%s/%s_scatter_%s_%s.png' %(dataFolder, batchSim, batchSim, pop, param), dpi=300)
            except:
                print('Error plotting %s vs %s' % (pop,param))



# -----------------------------------------------------------------------------
# Main code
# -----------------------------------------------------------------------------
if __name__ == '__main__': 
    dataFolder = '../data/'
    batchSim = 'v25_batch2' 

    allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI']#, 'IC']

    # set font size
    plt.rcParams.update({'font.size': 18})

    # get param labelsc
    paramLabels = getParamLabels(dataFolder, batchSim)

    # load evol data from files
    df = loadData(dataFolder, batchSim, pops = allpops, loadStudyFromFile=True, loadDataFromFile=True)

    plotScatterPopVsParams(dataFolder, batchSim, df, pops = ['IT3'])

    # filter results by pop rates
    #dfFilter = filterRates(dfPops, condlist=['rates'], copyFolder='best', dataFolder=dataFolder, batchLabel=batchSim, skipDepol=False) # ,, 'I>E', 'E5>E6>E2' 'PV>SOM']

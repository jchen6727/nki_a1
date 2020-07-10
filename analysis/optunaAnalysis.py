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
import optuna
import pickle

def getParamLabels(dataFolder, batchSim):
    # get param labels
    with open('%s/%s/%s_batch.json' % (dataFolder, batchSim, batchSim), 'r') as f: 
        paramLabels = [str(x['label'][0])+str(x['label'][1]) if isinstance(x['label'], list) else str(x['label']) for x in json.load(f)['batch']['params']]
    return paramLabels

def loadData(dataFolder, batchSim, pops, loadStudyFromFile=False):
    if loadStudyFromFile:
        with open('%s/%s/%s_df_study.pkl' % (dataFolder, batchSim, batchSim), 'rb') as f:
            df = pickle.load(f)
    else:
        study = optuna.create_study(study_name=batchSim, storage='sqlite:///%s/%s/%s_storage.db' % (dataFolder, batchSim, batchSim), load_if_exists=True) 
        df = study.trials_dataframe(attrs=('number', 'value', 'params'))

        # replace column labels
        for col in df.columns:
            if col.startswith('params_'):
                newName = col.replace('parms_', '').replace('(', '').replace(')', '').replace("'", "").replace(', ','')
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

    with open('%s/%s/%s_df.pkl' % (dataFolder, batchSim, batchSim), 'w') as f:
        pickle.dump(df, f)

    return df

def plotParamsVsRates(dataFolder, batchsim, dfParams, dfPops, pops, excludeAbove):

    if excludeAbove:
        dfParams = dfParams[dfParams.fit < excludeAbove]

    dfMerge = pd.merge(dfParams, dfPops, on=['gen', 'cand'])
    dfMerge = dfMerge.query('rate>=0.0')

    for pop in pops:

        popRates = list(dfMerge.query('pop==@pop').sort_values(by=['gen', 'cand']).rate)
        dfMergeParamsPop1 = dfMerge.query('pop==@pop').sort_values(by=['gen', 'cand'])
        dfMergeParamsPop2 = dfMergeParamsPop1.drop(['gen', 'cand', 'gen_cand', 'fit_x', 'fit_y', 'rate', 'avgFit', 'pop'], axis=1)
        
        plt.figure(figsize=(16,12))
        for i, (k,v) in enumerate(dfMergeParamsPop2.items()):
            y = (np.array(v)-min(v))/(max(v)-min(v)) # normalize
            x = np.random.normal(i, 0.10, size=len(y))         # Add some random "jitter" to the x-axis
            s = plt.scatter(x, y, alpha=0.1, c=popRates, cmap='jet_r')
        plt.colorbar(label = '%s rate (Hz)' % pop)
        plt.ylabel('normalized parameter value')
        plt.xlabel('parameter')
        plt.xticks(range(len(paramLabels)), paramLabels, rotation=45)
        plt.subplots_adjust(top=0.95, bottom=0.2, right=0.95)
        plt.savefig('%s/%s/%s_scatter_params_pop_%s_%s.png' % (dataFolder, batchSim, batchSim, pop,'excludeAbove-'+str(excludeAbove) if excludeAbove else ''))

        #ipy.embed()

        #plt.show()

def plotRatesVsParams(dataFolder, batchsim, dfParams, dfPops, ymax, excludeAbove, excludeBelow):

    if excludeAbove:
        dfParams = dfParams[dfParams.fit < excludeAbove]

    if excludeBelow:
        dfParams = dfParams[dfParams.fit < excludeBelow]

    dfParams = dfParams.rename(columns={'IELayerGain1-3': 'IELayerGain13', 'IILayerGain1-3': 'IILayerGain13'})
    dfMerge = pd.merge(dfParams, dfPops, on=['gen', 'cand'])
    dfMerge = dfMerge.query('rate>=0.0')
    dfMerge2 = dfMerge.drop(['gen_cand', 'cand', 'gen', 'avgFit', 'fit_x', 'fit_y'], axis=1)

    for param in [x for x in dfMerge2.columns if x not in ['pop', 'rate']]:
        print('Plotting scatter of rate vs %s param ...' %(param))
        plt.figure(figsize=(16, 12))
        dfMerge3 = dfMerge2.query('%s >= 0.0' % (param))
        dfMerge4 = dfMerge3.groupby('pop')
        pops = dfMerge4.groups.keys()
        for i, (k,v) in enumerate(dfMerge4):
            y = list(v['rate'].values)  #(np.array(v)-min(v))/(max(v)-min(v)) # normalize
            vals = list(v[param].values)
            x = np.random.normal(i, 0.04, size=len(y))         # Add some random "jitter" to the x-axis
            s = plt.scatter(x, y, alpha=0.1, c=vals, cmap='jet_r')
        plt.colorbar(label = 'Parameter value')
        plt.ylabel('firing rate')
        if ymax: plt.ylim(0,ymax)
        plt.xlabel('population')
        plt.xticks(range(len(pops)), pops, rotation=45)
        plt.subplots_adjust(top=0.95, bottom=0.2, right=0.95)
        plt.title('Parameter: %s' % (param))
        plt.savefig('%s/%s/%s_scatter_rates_%s_%s_%s_%s.png' %
            (dataFolder, batchSim, batchSim, param, str(ymax) if ymax else '', 'excludeBelow-'+str(excludeBelow) if excludeBelow else '', 'excludeAbove-'+str(excludeAbove) if excludeAbove else ''))


        #ipy.embed()

        #plt.show()

def normalize(df, exclude=[]):
    result = df.copy()
    for feature_name in [f for f in df.columns if f not in exclude]:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def filterRates(df, condlist=['rates', 'I>E', 'E5>E6>E2', 'PV>SOM'], copyFolder=None, dataFolder=None, batchLabel=None, skipDepol=False):
    from os.path import isfile, join
    from glob import glob

    df = df[['gen_cand', 'pop', 'rate']].pivot(columns='pop', index='gen_cand')
    df.columns = df.columns.droplevel(0)

    allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI']

    ranges = {}
    Erange = [0.05,100]
    Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'CT5B', 'PT5B', 'IT6','CT6', 'TC', 'TCM', 'HTC']
    for pop in Epops:
        ranges[pop] = Erange
    
    conds = []
    # check pop rate ranges
    if 'rates' in condlist:
        for k,v in ranges.items(): conds.append(str(v[0]) + '<=' + k + '<=' + str(v[1]))
    condStr = ''.join([''.join(str(cond) + ' and ') for cond in conds])[:-4]
    dfcond = df.query(condStr)

    ranges = {}
    Irange = [0.05,180]
    Ipops = ['NGF1',                        # L1
        'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
        'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
        'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
        'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
        'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',#,  # L5B
        'SOM6', 'VIP6', 'NGF6',
        'IRE', 'IREM', 'TI']      # L6 PV6

    for pop in Ipops:
        ranges[pop] = Irange

    conds = []

    # check pop rate ranges
    if 'rates' in condlist:
        for k,v in ranges.items(): conds.append(str(v[0]) + '<=' + k + '<=' + str(v[1]))
    condStr = ''.join([''.join(str(cond) + ' and ') for cond in conds])[:-4]
    dfcond = dfcond.query(condStr)


    # # check I > E in each layer
    # if 'I>E' in condlist:
    #     conds.append('PV2 > IT2 and SOM2 > IT2')
    #     conds.append('PV5A > IT5A and SOM5A > IT5A')
    #     conds.append('PV5B > IT5B and SOM5B > IT5B')
    #     conds.append('PV6 > IT6 and SOM6 > IT6')

    # # check E L5 > L6 > L2
    # if 'E5>E6>E2' in condlist:
    #     #conds.append('(IT5A+IT5B+PT5B)/3 > (IT6+CT6)/2 > IT2')
    #     conds.append('(IT5A+IT5B+PT5B)/3 > (IT6+CT6)/2')
    #     conds.append('(IT6+CT6)/2 > IT2')
    #     conds.append('(IT5A+IT5B+PT5B)/3 > IT2')
    
    # # check PV > SOM in each layer
    # if 'PV>SOM' in condlist:
    #     conds.append('PV2 > IT2')
    #     conds.append('PV5A > SOM5A')
    #     conds.append('PV5B > SOM5B')
    #     conds.append('PV6 > SOM6')


    # construct query and apply
    # condStr = ''.join([''.join(str(cond) + ' and ') for cond in conds])[:-4]
    # dfcond = df.query(condStr)

    print('\n Filtering based on: ' + str(condlist) + '\n' + condStr)
    print(dfcond)
    print(len(dfcond))

    # copy files
    if copyFolder:
        targetFolder = dataFolder+batchLabel+'/'+copyFolder
        try: 
            os.mkdir(targetFolder)
        except:
            pass
        
        for i,row in dfcond.iterrows():     
            if skipDepol:
                sourceFile1 = dataFolder+batchLabel+'/noDepol/'+batchLabel+row['simLabel']+'*.png'  
            else:
                sourceFile1 = dataFolder+batchLabel+'/gen_'+i.split('_')[0]+'/gen_'+i.split('_')[0]+'_cand_'+i.split('_')[1]+'_*raster*.png'   
            #sourceFile2 = dataFolder+batchLabel+'/'+batchLabel+row['simLabel']+'.json'
            if len(glob(sourceFile1))>0:
                cpcmd = 'cp ' + sourceFile1 + ' ' + targetFolder + '/.'
                #cpcmd = cpcmd + '; cp ' + sourceFile2 + ' ' + targetFolder + '/.'
                os.system(cpcmd) 
                print(cpcmd)


    return dfcond



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
    df = loadData(dataFolder, batchSim, pops = allpops, loadStudyFromFile=False)

    #plotParamsVsRates(dataFolder, batchSim, dfParams, dfPops, allpops, excludeAbove=400)
    #plotRatesVsParams(dataFolder, batchSim, dfParams, dfPops, ymax=50, excludeAbove=400, excludeBelow=None)


    # filter results by pop rates
    #dfFilter = filterRates(dfPops, condlist=['rates'], copyFolder='best', dataFolder=dataFolder, batchLabel=batchSim, skipDepol=False) # ,, 'I>E', 'E5>E6>E2' 'PV>SOM']

"""
paper.py 

Paper figures

Contributors: salvadordura@gmail.com
"""

import utils
import json
import numpy as np
import scipy
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sb
import os
import pickle
import batchAnalysis as ba
from netpyne.support.scalebar import add_scalebar
from matplotlib import cm
from bicolormap import bicolormap 

import IPython as ipy

#plt.ion()  # interactive

# ---------------------------------------------------------------------------------------------------------------
# Population params
allpops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3',  'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B',  'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM', 'IC']


def loadSimData(dataFolder, batchLabel, simLabel):
    ''' load sim file'''
    root = dataFolder+batchLabel+'/'
    sim,data,out = None, None, None
    if isinstance(simLabel, str): 
        filename = root+simLabel+'.pkl'
        print(filename)
        sim,data,out = utils.plotsFromFile(filename, raster=0, stats=0, rates=0, syncs=0, hist=0, psd=0, traces=0, grang=0, plotAll=0)
    
    return sim, data, out, root

def axisFontSize(ax, fontsize):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize) 



def fig_conn():
    # NOTE: data files need to be loaded using Python 2!
    # load conn matrices
    # with open('../data/v53_manualTune/v53_tune7_conn_strength_conn.pkl', 'rb') as f:
    #     data = pickle.load(f)

    simLabel = batchLabel = 'v11_manualTune'
    dataFolder = '../data/'
    root = dataFolder+batchLabel+'/'

    with open('../data/v11_manualTune/v11_sim26_conn.pkl', 'rb') as f:
        dataP = pickle.load(f)
    
    popsPre = dataP['includePre']
    popsPost = dataP['includePost']
    
    # strength
    # connMatrix = dataW['connMatrix'].T * dataP['connMatrix'].T
    # feature = 'strength'

    # prob
    connMatrix = dataP['connMatrix']
    feature = 'Probability of connection'

    connMatrix *= 0.75

    connMatrix[-5:, -5:] *= 0.75
    connMatrix[-5:, -5:] *= 0.75
    connMatrix[-5,11] = 0.3
    connMatrix[-5,12] = 0.3
    connMatrix[-3,11] = 0.3
    connMatrix[-3,12] = 0.3


    # font
    fontsiz = 14
    plt.rcParams.update({'font.size': fontsiz})


    # ----------------------- 
    # conn matrix full
    # connMatrix[:, inhPopsInds] *= -1.0

    vmin = np.nanmin(connMatrix)
    vmax = np.nanmax(connMatrix) 

    plt.figure(figsize=(14, 14))
    h = plt.axes()
    plt.imshow(connMatrix, interpolation='nearest', cmap='viridis', vmin=vmin, vmax=vmax)  #_bicolormap(gap=0)

    for ipop, pop in enumerate(popsPost):
        plt.plot(np.array([0,len(popsPre)])-0.5,np.array([ipop,ipop])-0.5,'-',c=(0.7,0.7,0.7))
    for ipop, pop in enumerate(popsPre):
        plt.plot(np.array([ipop,ipop])-0.5,np.array([0,len(popsPost)])-0.5,'-',c=(0.7,0.7,0.7))

    # Make pretty
    h.set_xticks(list(range(len(popsPre))))
    h.set_yticks(list(range(len(popsPost))))
    h.set_xticklabels(popsPre, rotation=90)
    h.set_yticklabels(popsPost)
    h.xaxis.set_ticks_position('top')
    plt.xlim(-0.5,len(popsPre)-0.5)
    plt.ylim(len(popsPost) - 0.5, -0.5)

    plt.grid(False)
    
    clim = [vmin, vmax]
    plt.clim(clim[0], clim[1])
    plt.colorbar(label=feature, shrink=0.8) #.set_label(label='Fitness',size=20,weight='bold')
    plt.xlabel('Target population')
    h.xaxis.set_label_coords(0.5, 1.12)
    plt.ylabel('Source population')
    plt.title('Connection ' + feature + ' matrix', y=1.14, fontWeight='bold')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=(0.05))

    plt.savefig('%s%s_connFull_%s.png' % (root, simLabel, feature), dpi=300)
    plt.show()


def compare_conn():

    # load Allen V1 conn
    with open('../conn/E->EI_Allen_V1_prob_0.25.pkl', 'rb') as f:
        dataAllen = pickle.load(f)
    
    # load BBP S1 conn
    with open('../conn/E->EI_BBP_S1_prob_0.25.pkl', 'rb') as f:
        dataBBP = pickle.load(f)

    # load custom A1 conn
    with open('../conn/E->EI_Allen_V1_I->EI_custom_A1_prob_0.25.pkl', 'rb') as f:
        dataCustom = pickle.load(f)
    
    popsPre = dataAllen['includePre']
    popsPost = dataAllen['includePost']
    
    # prob
    connAllen = dataAllen['connMatrix']
    connBBP = dataBBP['connMatrix']
    connCustom = dataCustom['connMatrix']
    feature = 'Probability of connection difference'

    diff_Allen_BBP_E = connAllen - connBBP

    diff_Custom_BBP_I = connCustom - connBBP

    diff_Custom_Allen_I = connCustom - connAllen


    Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'CT5B', 'PT5B', 'IT6', 'CT6']  # all layers

    Ipops = ['NGF1',                            # L1
            'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
            'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
            'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
            'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
            'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
            'PV6', 'SOM6', 'VIP6', 'NGF6']  # L6 

    allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'CT5B', 'PT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6']

    popsPost = allpops # NOTE: not sure why CT5B and PT5B order was switched

    excPopsInds = [1, 6, 11,12, 17, 18, 23, 24, 25, 30,31]
    inhPopsInds = [0,2,3,4,5,7,8,9,10,13,14,15,16,19,20,21,22,26,27,28,29,32,33,34,35]


    # font
    fontsiz = 14
    plt.rcParams.update({'font.size': fontsiz})

    cmap = bicolormap() # Default ,should work for most things
    # cmap = bicolormap(gap=0,mingreen=0,redbluemix=1,epsilon=0) # From pure red to pure blue with white in the middle
    # cmap = bicolormap(gap=0,mingreen=0,redbluemix=0,epsilon=0.1) # Red -> yellow -> gray -> turquoise -> blue
    # cmap = bicolormap(gap=0.3, mingreen=0.2, redbluemix=0, epsilon=0.01)  # Red and blue with a sharp distinction between

    connMatrices = [diff_Allen_BBP_E, diff_Custom_BBP_I, diff_Custom_Allen_I]
    diffConnFilenames = ['diff_Allen_BBP_E', 'diff_Custom_BBP_I', 'diff_Custom_Allen_I']
    diffConnTitles = ['Allen V1 (=current A1) - BBP S1 exc connectivity matrix (difference)',
                      'Custom A1 - BBP S1 inh connectivity matrix (difference)',
                      'Custom A1 - Allen V1 inh connectivity matrix (difference)']
    diffPops = [Epops, Ipops, Ipops]
    diffPopInds = [excPopsInds, inhPopsInds, inhPopsInds]
    figYsizes = [8, 12, 12]


    for connMatrix, popsPre, popInds, filename, title, figYsize in zip(connMatrices, diffPops, diffPopInds, diffConnFilenames, diffConnTitles, figYsizes):
        # ----------------------- 
        # conn matrix full
        #import IPython; IPython.embed()

        connMatrix = connMatrix[popInds,:]
        
        vmin = np.nanmin(connMatrix)
        vmax = np.nanmax(connMatrix) 
        #make symetric
        if vmax > vmin:
            vmin = -vmax
        else:
            vmax = -vmin
    
        plt.figure(figsize=(18 , figYsize))
        h = plt.axes()
        plt.imshow(connMatrix, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)  #_bicolormap(gap=0)

        for ipop, pop in enumerate(popsPre):
            plt.plot(np.array([0,len(popsPost)])-0.5,np.array([ipop,ipop])-0.5,'-',c=(0.7,0.7,0.7))
        for ipop, pop in enumerate(popsPost):
            plt.plot(np.array([ipop,ipop])-0.5,np.array([0,len(popsPre)])-0.5,'-',c=(0.7,0.7,0.7))

        # Make pretty
        h.set_yticks(list(range(len(popsPre))))
        h.set_xticks(list(range(len(popsPost))))
        h.set_yticklabels(popsPre)
        h.set_xticklabels(popsPost, rotation=90)
        h.xaxis.set_ticks_position('top')
        # plt.ylim(-0.5,len(popsPre)-0.5)
        # plt.xlim(-0.5, len(popsPost) - 0.5)

        plt.grid(False)
        
        clim = [vmin, vmax]
        plt.clim(clim[0], clim[1])
        plt.colorbar(label=feature, shrink=0.8) #.set_label(label='Fitness',size=20,weight='bold')
        plt.xlabel('Target population')
        plt.ylabel('Source population')
        if figYsize == 8:
            h.xaxis.set_label_coords(0.5, 1.20)
            plt.title(title, y=1.22, fontWeight='bold')
        else:
            h.xaxis.set_label_coords(0.5, 1.08)
            plt.title(title, y=1.10, fontWeight='bold')
        plt.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.00)

        plt.savefig('../conn/'+filename, dpi=300)

        #import IPython; IPython.embed()



def plot_empirical_conn():

    with open('../conn/conn.pkl', 'rb') as fileObj: connData = pickle.load(fileObj)
    pmat = connData['pmat']
    lmat = connData['lmat']

    allpops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC',  'TI', 'IRE', 'IREM']
        
    popsPre = allpops
    popsPost = allpops  # NOTE: not sure why CT5B and PT5B order was switched
    
    connMatrix = np.zeros((len(popsPre), len(popsPost)))

    d = 50
    for ipre, pre in enumerate(popsPre):
        for ipost, post in enumerate(popsPost):
            print(pre,post)
            try:
                if pre in lmat and post in lmat[pre]:
                    connMatrix[ipre, ipost] = pmat[pre][post] * np.exp(-d / lmat[pre][post])** 2
                else:
                    connMatrix[ipre, ipost] = pmat[pre][post]
            except:
                connMatrix[ipre, ipost] = 0.0
                #connMatrix[ipre, ipost] = pmat[pre][post]

    # font
    fontsiz = 14
    plt.rcParams.update({'font.size': fontsiz})

    # ----------------------- 
    # conn matrix full
    #import IPython; IPython.embed()

    vmin = np.nanmin(connMatrix)
    vmax = np.nanmax(connMatrix)
        
    plt.figure(figsize=(12, 12))
    h = plt.axes()
    plt.imshow(connMatrix, interpolation='nearest', cmap='viridis', vmin=vmin, vmax=vmax)  #_bicolormap(gap=0)


    ipopBoundaries = [1, 6, 11, 17, 23, 30, 40]
    for ipop, pop in enumerate(popsPre):
        if ipop in ipopBoundaries: # thicker, brighter, dotted lines for layer boundaries
            plt.plot(np.array([0,len(popsPost)])-0.5,np.array([ipop,ipop])-0.5,'-',c=(0.8,0.8,0.8), lw=3)
        else:
            plt.plot(np.array([0,len(popsPost)])-0.5,np.array([ipop,ipop])-0.5,'-',c=(0.7,0.7,0.7))
    for ipop, pop in enumerate(popsPost):
        if ipop in ipopBoundaries: # thicker, brighter, dotted lines for layer boundaries
            plt.plot(np.array([ipop,ipop])-0.5,np.array([0,len(popsPre)])-0.5,'-',c=(0.8,0.8,0.8), lw=3)
        else:
            plt.plot(np.array([ipop,ipop])-0.5,np.array([0,len(popsPre)])-0.5,'-',c=(0.7,0.7,0.7))

    ipop = 36 # thal boundary
    plt.plot(np.array([0,len(popsPost)])-0.5,np.array([ipop,ipop])-0.5,'-', c='orange', lw=3)
    plt.plot(np.array([ipop,ipop])-0.5,np.array([0,len(popsPre)])-0.5,'-', c='orange', lw=3)


    # Make pretty
    h.set_yticks(list(range(len(popsPre))))
    h.set_xticks(list(range(len(popsPost))))
    h.set_yticklabels(popsPre)
    h.set_xticklabels(popsPost, rotation=90)
    h.xaxis.set_ticks_position('top')
    # plt.ylim(-0.5,len(popsPre)-0.5)
    # plt.xlim(-0.5, len(popsPost) - 0.5)

    plt.grid(False)

    vmax = 0.5
    clim = [vmin, vmax]
    plt.clim(clim[0], clim[1])
    plt.colorbar(label='probability', shrink=0.8) #.set_label(label='Fitness',size=20,weight='bold')
    plt.xlabel('Post')
    plt.ylabel('Pre')
    title = 'Connection probability matrix' # empirical
    h.xaxis.set_label_coords(0.5, 1.11)
    plt.title(title, y=1.12, fontWeight='bold')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.00)

    #filename = 'EI->EI_Allen_custom_prob_conn_empirical_0.25.png'
    filename = 'Full_Allen_custom_prob_conn_empirical.png'
    plt.savefig('../conn/'+filename, dpi=300)

    #import IPython; IPython.embed()


def plot_net_conn():
    # load custom A1 conn
    with open('../conn/EI->EI_Allen_custom_prob_0.25_fixed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # prob
    connMatrix = data['connMatrix']    
    
    allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3',  'PV3', 'SOM3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'CT5B', 'PT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6']

    popsPre = allpops
    popsPost = allpops  # NOTE: not sure why CT5B and PT5B order was switched
    
    # font
    fontsiz = 14
    plt.rcParams.update({'font.size': fontsiz})

    # ----------------------- 
    # conn matrix full
    #import IPython; IPython.embed()

    vmin = np.nanmin(connMatrix)
    vmax = 0.5 #np.nanmax(connMatrix)
        
    plt.figure(figsize=(12, 12))
    h = plt.axes()
    plt.imshow(connMatrix, interpolation='nearest', cmap='viridis', vmin=vmin, vmax=vmax)  #_bicolormap(gap=0)

    for ipop, pop in enumerate(popsPre):
        plt.plot(np.array([0,len(popsPost)])-0.5,np.array([ipop,ipop])-0.5,'-',c=(0.7,0.7,0.7))
    for ipop, pop in enumerate(popsPost):
        plt.plot(np.array([ipop,ipop])-0.5,np.array([0,len(popsPre)])-0.5,'-',c=(0.7,0.7,0.7))

    # Make pretty
    h.set_yticks(list(range(len(popsPre))))
    h.set_xticks(list(range(len(popsPost))))
    h.set_yticklabels(popsPre)
    h.set_xticklabels(popsPost, rotation=90)
    h.xaxis.set_ticks_position('top')
    # plt.ylim(-0.5,len(popsPre)-0.5)
    # plt.xlim(-0.5, len(popsPost) - 0.5)

    plt.grid(False)

    clim = [vmin, vmax]
    plt.clim(clim[0], clim[1])
    plt.colorbar(label='probability', shrink=0.8) #.set_label(label='Fitness',size=20,weight='bold')
    plt.xlabel('Post')
    plt.ylabel('Pre')
    title = 'Network instance connection probability matrix'
    h.xaxis.set_label_coords(0.5, 1.10)
    plt.title(title, y=1.12, fontWeight='bold')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.00)

    filename = 'EI->EI_Allen_custom_prob_conn_net_0.25.png'
    plt.savefig('../conn/'+filename, dpi=300)

def plot_net_conn_cns20poster():
    # load custom A1 conn
    with open('../data/v25_batch4/conn_prob_new.pkl', 'rb') as f:   #v25_batch4_conn_prob.pkl
        data = pickle.load(f)
    
    # prob
    connMatrix = data['connMatrix']
    #includePre = data['includePre']
    #includePost = data['includePost']

    allpops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3',  'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B',  'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI']# , 'IC']

    popsPre = allpops
    popsPost = allpops  # NOTE: not sure why CT5B and PT5B order was switched


    #import IPython; IPython.embed()

    # Note: this code doesn't work to rearrange matrix rows + cols
    # Using correct pop order in recording
    #
    # connMatrix = np.zeros((len(popsPre), len(popsPost)))
    # 
    # for ipre, pre in enumerate(popsPre):
    #     for ipost, post in enumerate(popsPost):
    #         #print(pre,post)
    #         try:
    #             connMatrix[ipre, ipost] = pmat[includePre.index(pre)][includePost.index(post)]
    #         except:
    #             connMatrix[ipre, ipost] = 0.0
    #             #connMatrix[ipre, ipost] = pmat[pre][post]    

    # font
    fontsiz = 14
    plt.rcParams.update({'font.size': fontsiz})

    # ----------------------- 
    # conn matrix full
    #import IPython; IPython.embed()

    vmin = np.nanmin(connMatrix)
    vmax =  np.nanmax(connMatrix) #0.5 # 0.7 #
        
    plt.figure(figsize=(12, 12))
    h = plt.axes()
    plt.imshow(connMatrix, interpolation='nearest', cmap='viridis', vmin=vmin, vmax=vmax)  #_bicolormap(gap=0)

    for ipop, pop in enumerate(popsPre):
        plt.plot(np.array([0,len(popsPost)])-0.5,np.array([ipop,ipop])-0.5,'-',c=(0.7,0.7,0.7))
    for ipop, pop in enumerate(popsPost):
        plt.plot(np.array([ipop,ipop])-0.5,np.array([0,len(popsPre)])-0.5,'-',c=(0.7,0.7,0.7))

    # Make pretty
    h.set_yticks(list(range(len(popsPre))))
    h.set_xticks(list(range(len(popsPost))))
    h.set_yticklabels(popsPre)
    h.set_xticklabels(popsPost, rotation=90)
    h.xaxis.set_ticks_position('top')
    # plt.ylim(-0.5,len(popsPre)-0.5)
    # plt.xlim(-0.5, len(popsPost) - 0.5)

    plt.grid(False)

    clim = [vmin, vmax]
    plt.clim(clim[0], clim[1])
    plt.colorbar(label='probability', shrink=0.8) #.set_label(label='Fitness',size=20,weight='bold')
    plt.xlabel('Post')
    plt.ylabel('Pre')
    title = 'Connection probability matrix'
    h.xaxis.set_label_coords(0.5, 1.11)
    plt.title(title, y=1.12, fontWeight='bold')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.00)

    filename = 'prob_conn_net_v25_batch4.png'
    plt.savefig('../conn/'+filename, dpi=300)


def fig_raster(simLabel):
    dataFolder = '../data/'
    batchLabel = 'v34_batch27/'
    #simLabel = 'v34_batch27_0_0'

    sim, data, out, root = loadSimData(dataFolder, batchLabel, simLabel)

    timeRange = [1000, 11000] #[2000, 4000]
    
    # raste
    include = allpops
    orderBy = ['pop'] #, 'y']
    #filename = '%s%s_raster_%d_%d_%s.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
    fig1 = sim.analysis.plotRaster(include=['allCells'], timeRange=timeRange, labels='legend', 
        popRates=0, orderInverse=True, lw=0, markerSize=3.5, marker='.',  
        showFig=0, saveFig=0, figSize=(8.5, 13), orderBy=orderBy)# 
    ax = plt.gca()

    [i.set_linewidth(0.5) for i in ax.spines.values()] # make border thinner
    #plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')  #remove ticks
    plt.xticks([1000, 2000, 3000, 4000, 5000, 6000], ['1', '2', '3', '4', '5', '6'])
    plt.yticks([0, 5000, 10000], [0, 5000, 10000])
    
    plt.ylabel('Neuron ID') #Neurons (ordered by NCD within each pop)')
    plt.xlabel('Time (s)')
    
    plt.title('')
    filename='%s%s_raster_%d_%d_%s.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
    plt.savefig(filename, dpi=600)


    # stats
    fig2 = sim.analysis.plotSpikeStats(include=['eachPop'], stats=['rate'], timeRange=timeRange, includeRate0=True,
        showFig=0, saveFig=0, figSize=(8.5, 13))

    plt.xlim([0, 120])
    plt.title('')
    filename='%s%s_stats_%d_%d_%s.png'%(root, simLabel, timeRange[0], timeRange[1], orderBy)
    plt.savefig(filename, dpi=600)


def fig_stats():
    pass

def fig_traces():
    '''
    plt.figure(figsize=(18,24)) 
    time = np.linspace(0, 2000, 20001)
    plt.ylabel('V (mV)', fontsize=16)
    plt.xlabel('time (ms)', fontsize=16)
    plt.xlim(0, 2000)
    # plt.ylim(-80, -30)
    plt.ylim(-6670,20)
    plt.yticks(np.arange(-6540,60,120),popParamLabels[::-1], fontsize=14)
    plt.xticks(fontsize=14)

    paramsName = paramsNameList[3]
    number = 0
    for popName in popParamLabels:   
        for metype in popNameName[popName]:
            cellNames = cellNameName[metype]
            for cellName in cellNames:
                Vt = np.array(data[paramsName]['simData']['V_soma'][cellName])
                plt.plot(time, (Vt-number*120.0)) 
        number = number + 1

    plt.savefig('Vt_full_' + paramsName + '.png', facecolor = 'white', bbox_inches='tight' , dpi=300)
'''

def fig_optuna_fitness():
    import optunaAnalysis as oa
    import seaborn as sns

    dataFolder = '../data/'
    batchSim = 'v34_batch14'
    loadFromFile = 1
    
    allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']  #, 'IC']
    rateTimeRanges = ['1500_1750', '1750_2000', '2000_2250', '2250_2500']

    # set font size
    plt.rcParams.update({'font.size': 18})

    # get param labelsc
    paramLabels = oa.getParamLabels(dataFolder, batchSim)

    # load evol data from files
    df = oa.loadData(dataFolder, batchSim, pops=allpops, rateTimeRanges=rateTimeRanges, loadStudyFromFile=loadFromFile, loadDataFromFile=loadFromFile)

    trial_fitness_single = 1
    params_fitness_all = 1
    rates_fitness_all = 1
    rates_fitness_single = 1

    # PLOTS
    skipCols = rateTimeRanges

    # plot trial vs fitness
    if trial_fitness_single:
        excludeAbove = 1000

        if excludeAbove:
            df = df[df.value < excludeAbove]

        df.reset_index()
        df.number=range(len(df))

        dfcorr=df.corr('pearson')

        min=df.iloc[0]['value']
        minlist=[]
        for x in df.value:
            if x < min:
                minlist.append(x)
                min = x
            else:
                minlist.append(min)

        param = 'value'
        if not any([skipCol in param for skipCol in skipCols]): 
            print('Plotting scatter of %s vs %s param (R=%.2f) ...' %('fitness', param, dfcorr['value'][param]))
            #df.plot.scatter(param, 'value', s=4, c='number', colormap='viridis', alpha=0.5, figsize=(8, 8), colorbar=False)
            plt.figure(figsize=(8, 8))
            df.plot.scatter('number', param, s=6, c='blue', colormap='jet_r', alpha=0.5, figsize=(8, 8), colorbar=False, vmin=50, vmax=400)

            #import IPython; IPython.embed()

            plt.plot(list(df['value'].rolling(window=10).mean()), c='orange', label='mean')
            plt.plot(minlist, c='red', label='min')
            # f = plt.gcf()
            # cax = f.get_axes()[1]
            # cax.set_ylabel('Fitness Error')
            plt.ylabel('Fitness Error')
            plt.xlabel('Trial')
            plt.ylim([0,1000])
            plt.legend()
            #plt.title('%s vs %s R=%.2f' % ('trial', param.replace('tune', ''), dfcorr['value'][param]))
            plt.savefig('%s/%s/%s_scatter_%s_%s.png' % (dataFolder, batchSim, batchSim, 'trial', param.replace('tune', '')), dpi=300)


    # plot params vs fitness
    if params_fitness_all:
        excludeAbove = 400
        ylim = None
        if excludeAbove:
            df = df[df.value < excludeAbove]

        df2 = df.drop(['value', 'number'], axis=1)
        fits = list(df['value'])
        plt.figure(figsize=(16, 8))

        paramsData = list(df2[paramLabels].items())

        for i, (k,v) in enumerate(paramsData):
            y = v #(np.array(v)-min(v))/(max(v)-min(v)) # normalize
            x = np.random.normal(i, 0.04, size=len(y))         # Add some random "jitter" to the x-axis
            s = plt.scatter(x, y, alpha=0.3, c=[int(f-1) for f in fits], cmap='jet_r') #)
        plt.colorbar(label = 'Fitness Error')
        plt.ylabel('Parameter value')
        plt.xlabel('Parameter')
        if ylim: plt.ylim(0, ylim)
        paramLabels = [x.replace('Gain','') for x in paramLabels]
        plt.xticks(range(len(paramLabels)), paramLabels, rotation=75)
        plt.subplots_adjust(top=0.99, bottom=0.30, right=1.05, left=0.05)
        plt.savefig('%s/%s/%s_scatter_params_%s.png' % (dataFolder, batchSim, batchSim, 'excludeAbove-'+str(excludeAbove) if excludeAbove else ''))
        #plt.show()


    # plot rates vs fitness
    if rates_fitness_all:
        excludeAbove = 400
        ylim = None
        if excludeAbove:
            df = df[df.value < excludeAbove]

        df2 = df[allpops]
        fits = list(df['value'])
        plt.figure(figsize=(16, 8))

        paramsData = list(df2[allpops].items())

        for i, (k,v) in enumerate(paramsData):
            y = v #(np.array(v)-min(v))/(max(v)-min(v)) # normalize
            x = np.random.normal(i, 0.04, size=len(y))         # Add some random "jitter" to the x-axis
            s = plt.scatter(x, y, alpha=0.3, c=[int(f-1) for f in fits], cmap='jet_r') #)
        plt.colorbar(label = 'Fitness Error')
        plt.ylabel('Firing Rate (Hz)')
        plt.xlabel('Population')
        if ylim: plt.ylim(0, ylim)
        plt.xticks(range(len(allpops)), allpops, rotation=75)
        plt.subplots_adjust(top=0.99, bottom=0.30, right=1.05, left=0.05)
        plt.savefig('%s/%s/%s_scatter_rates_all%s.png' % (dataFolder, batchSim, batchSim, 'excludeAbove-'+str(excludeAbove) if excludeAbove else ''))
        #plt.show()


    # plot pop rate vs fitness
    if rates_fitness_single:
        excludeAbove = 1000
        if excludeAbove:
            df = df[df.value < excludeAbove]

        dfcorr=df.corr('pearson')

        for param in ['PV4']:
            if not any([skipCol in param for skipCol in skipCols]): 
                print('Plotting scatter of %s vs %s param (R=%.2f) ...' %('fitness', param, dfcorr['value'][param]))
                df.plot.scatter(param, 'value', s=6, c='blue', colormap='jet_r', alpha=0.5, figsize=(8, 8), colorbar=False, vmin=0, vmax=1000) #, colorbar=False)
                plt.ylabel('Fitness Error')
                plt.xlabel('PV4 Rate (Hz)')
                plt.xlim([0,30])
                #plt.title('%s vs %s R=%.2f' % ('fitness', param.replace('tune', ''), dfcorr['value'][param]))
                plt.savefig('%s/%s/%s_scatter_%s_%s.png' % (dataFolder, batchSim, batchSim, 'fitness', param.replace('tune', '')), dpi=300)


# Main
if __name__ == '__main__':
    # fig_conn()
    # compare_conn()
    plot_empirical_conn()
    # plot_net_conn_cns20poster()
    
    # for iseed in range(5):
    #     for jseed in range(5):
    #         simLabel = 'v34_batch27_%d_%d' %(iseed, jseed)
    #         fig_raster(simLabel)

    #fig_optuna_fitness()
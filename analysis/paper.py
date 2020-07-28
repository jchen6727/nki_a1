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
allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM']



def loadSimData(dataFolder, batchLabel, simLabel):
    ''' load sim file'''
    root = dataFolder+batchLabel+'/'
    sim,data,out = None, None, None
    if isinstance(simLabel, str): 
        filename = root+simLabel+'.json'
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

    allpops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI']
        
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

#### main
# fig_conn()
# compare_conn()
plot_empirical_conn()
plot_net_conn_cns20poster()
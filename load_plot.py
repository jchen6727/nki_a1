"""
script to load sim and plot
"""

from netpyne import sim
from matplotlib import pyplot as plt
import os
import IPython as ipy
import pickle as pkl


def plot_LFP_PSD_combined(dataFile, norm=False):
    NHP = dataFile[:-4]
    
    with open(dataFile, 'rb') as f:
        loadedData = pkl.load(f)
        allData = loadedData['allData']
        freqs = loadedData['allFreqs'][0]

    plt.figure(figsize=(8,12))
    fontSize = 12
    lw = 1

    elecLabels = ['All layers', 'Supragranular', 'Granular', 'Infragranular']

    for i in range(len(elecLabels)):
        plt.subplot(4, 1, i+1)
        for itime in range(len(allData)):
            signal = allData[itime][i]
            if norm:
                signal = signal / max(signal)
            plt.plot(freqs, signal, linewidth=lw) #, color=color)
        
        plt.title(elecLabels[i])
        plt.xlim([0, 50])
        if norm:
            plt.ylabel('Normalized Power', fontsize=fontSize)
        else:
            plt.ylabel('Power (mV^2/Hz)', fontsize=fontSize)
        plt.xlabel('Frequency (Hz)', fontsize=fontSize)
    
    # format plot 
    plt.tight_layout()
    plt.suptitle('LFP PSD - %s' % (NHP), fontsize=fontSize, fontweight='bold') # add yaxis in opposite side
    plt.subplots_adjust(bottom=0.08, top=0.92)

    if norm:
        plt.savefig('%s_PSD_combined_norm.png' % (NHP))
    else:
        plt.savefig('%s_PSD_combined.png' % (NHP))


###########################
######## MAIN CODE ########
###########################

if __name__ == '__main__':

    dataType = 'speech' #'speech' #'spont'

    if dataType == 'spont':
        filenames = ['data/v34_batch57/v34_batch57_%d_%d_data.pkl' % (iseed, cseed) for iseed in [0,1,2,3,4] for cseed in [0,1,2,3,4]]
        #filenames = ['data/v34_batch56/v34_batch56_%d_%d_data.pkl' % (iseed, cseed) for iseed in [0] for cseed in [0]]

        timeRange = [1500, 11500]

        #filenames = ['data/v34_batch56/v34_batch56_0_0_data.pkl']
        #timeRange = [2000, 2200]

        # find all individual sim labels whose files need to be gathered
    #filenames = [targetFolder+f for f in os.listdir(targetFolder) if f.endswith('.pkl')]

    elif dataType == 'speech':
        filenames = ['data/v34_batch58/v34_batch58_%d_%d_%d_%d_%d_data.pkl' % (i1, i2, i3, i4, i5) 
                                                                        for i1 in [0] \
                                                                        for i2 in [0,1] \
                                                                        for i3 in [0,1] \
                                                                        for i4 in [0,1] \
                                                                        for i5 in [0, 1, 2, 3, 4]]

        # good_rasters = [
        # '0_0_0_0_1',
        # '0_0_0_0_2',
        # '0_0_0_1_1',
        # '0_0_0_1_2',
        # '0_0_1_0_1',
        # '0_0_1_0_2',
        # '0_1_0_0_1',
        # '0_1_0_0_2',
        # '0_1_0_1_2',
        # '0_1_1_0_1',
        # '0_1_1_0_2',
        # '1_0_0_0_2',
        # '1_1_0_1_5',
        # '1_1_1_0_5',
        # '1_1_1_1_4',
        # '1_1_1_1_5',
        # ]

        # filenames = ['data/v34_batch55/v34_batch55_%s_data.pkl' % (s1) for s1 in good_rasters] 
        # timeRange = [2500, 4000]

    layer_bounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550}#, 'L6': 2000}
    layer_bounds= {'S': 950, 'G': 1250, 'I': 1900}#, 'L6': 2000}


    allData = []

    for filename in filenames:
        sim.load(filename, instantiate=False)

        # standardd plots
        #sim.analysis.plotRaster(**{'include': ['allCells'], 'saveFig': True, 'showFig': False, 'popRates': 'minimal', 'orderInverse': True, 'timeRange': [1500,5500], 'figSize': (7,12), 'lw': 0.3, 'markerSize': 3, 'marker': '.', 'dpi': 300})
        #sim.analysis.plotSpikeStats(stats=['rate'],figSize = (6,12), timeRange=[1500, 11500], dpi=300, showFig=0, saveFig=filename[:-4]+'_stats_10sec')
        #sim.analysis.plotSpikeStats(stats=['rate'],figSize = (6,12), timeRange=[1500, 6500], dpi=300, showFig=0, saveFig=filename[:-4]+'_stats_5sec')
        #sim.analysis.plotLFP(**{'plots': ['spectrogram'], 'electrodes': ['avg', [0], [1], [2,3,4,5,6,7,8,9], [10, 11, 12], [13], [14, 15], [16,17,18,19]], 'timeRange': timeRange, 'maxFreq': 50, 'figSize': (8,24), 'saveData': False, 'saveFig': filename[:-4]+'_LFP_spec_7s_all_elecs', 'showFig': False})
        # out = sim.analysis.plotLFP(**{'plots': ['timeSeries'], 
        #         'electrodes': 
        #         #['avg', [0], [1], [2,3,4,5,6,7,8,9], [10, 11, 12], [13], [14, 15], [16,17,18,19]], 
        #         ['avg', [0, 1,2,3,4,5,6,7,8,9], [10, 11, 12], [13,14, 15,16,17,18,19]],
        #         'timeRange': timeRange, 
        #         #'maxFreq': 50, 
        #         'figSize': (24,12), 
        #         'saveData': False, 
        #         'saveFig': filename[:-4]+'_LFP_timeSeries_10s_avg', 'showFig': False})

        # out = sim.analysis.plotLFP(**{'plots': ['spectrogram'], 
        #         'electrodes': 
        #         #['avg', [0], [1], [2,3,4,5,6,7,8,9], [10, 11, 12], [13], [14, 15], [16,17,18,19]], 
        #         ['avg', [0, 1,2,3,4,5,6,7,8,9], [10, 11, 12], [13,14, 15,16,17,18,19]],
        #         'timeRange': timeRange, 
        #         'maxFreq': 50, 
        #         'figSize': (16,12), 
        #         'saveData': False, 
        #         'saveFig': filename[:-4]+'_LFP_spect_11s_3layers', 'showFig': False})

        # out = sim.analysis.plotLFP(**{'plots': ['PSD'], 
        #         'electrodes': 
        #         #['avg', [0], [1], [2,3,4,5,6,7,8,9], [10, 11, 12], [13], [14, 15], [16,17,18,19]], 
        #         ['avg', [0, 1,2,3,4,5,6,7,8,9], [10, 11, 12], [13,14, 15,16,17,18,19]],
        #         'timeRange': timeRange, 
        #         'maxFreq': 50, 
        #         'figSize': (8,12), 
        #         'saveData': False, 
        #         'saveFig': filename[:-4]+'_LFP_psd_10s_3layers', 'showFig': False})


        # required for combined PSD plot
    #     allData.append(out[1]['allSignal'])
            
    #     with open('data/v34_batch57/v34_batch57_10sec_allData.pkl', 'wb') as f:
    #         pkl.dump({'allData': allData, 'allFreqs': out[1]['allFreqs']}, f)  
        
    # plot_LFP_PSD_combined('data/v34_batch57/v34_batch57_10sec_allData.pkl', norm=False)

    
        # # for elec in [8,9,10,11,12]:
        # #     sim.analysis.plotLFP(**{'plots': ['timeSeries'], 'electrodes': [elec], 'timeRange': [1500, 11500], 'maxFreq':50, 'figSize': (8,4), 'saveData': False, 'saveFig': filename[:-4]+'_LFP_signal_10s_elec_'+str(elec), 'showFig': False})
        # #     sim.analysis.plotLFP(**{'plots': ['PSD'], 'electrodes': [elec], 'timeRange': [1500, 11500], 'maxFreq':50, 'figSize': (8,4), 'saveData': False, 'saveFig': filename[:-4]+'_LFP_PSD_10s_elec_'+str(elec), 'showFig': False})
        # #     sim.analysis.plotLFP(**{'plots': ['spectrogram'], 'electrodes': [elec], 'timeRange': [1500, 11500], 'maxFreq':50, 'figSize': (8,4), 'saveData': False, 'saveFig': filename[:-4]+'_LFP_spec_10s_elec_'+str(elec), 'showFig': False})

        # # #plt.ion()
        # tranges = [[6000, 6200],
        #             [6000, 6300]]
        #     #         [1980, 2100],
        #     #         [2080, 2200],
        #     #         [2030, 2150],
        #     #         [2100, 2200]] 
    
        tranges = [[x, x+200] for x in range(2400, 3500, 100)]
        smooth = 30
        #tranges = [[500,11500]]
        for t in tranges:# (2100, 2200,100):    
            sim.analysis.plotCSD(**{
                'spacing_um': 100, 
                'layer_lines': 1, 
                'layer_bounds': layer_bounds, 
                'overlay': 'LFP',
                'timeRange': [t[0], t[1]], 
                'smooth': smooth,
                'saveFig': filename[:-4]+'_CSD_LFP_smooth%d_%d-%d' % (smooth, t[0], t[1]), 
                'vaknin': False,
                'figSize': (4.1,8.2), 
                'dpi': 300, 
                'showFig': 0})
        


    # ipy.embed()
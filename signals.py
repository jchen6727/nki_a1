# $Id: filter.py,v 1.1 2010/12/02 16:34:54 samn Exp $

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: filter.py
#  Purpose: Various Seismogram Filtering Functions
#   Author: Tobias Megies, Moritz Beyreuther, Yannik Behr
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2009 Tobias Megies, Moritz Beyreuther, Yannik Behr
# --------------------------------------------------------------------
"""
Various Seismogram Filtering Functions

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
"""

import numpy as np
from scipy.signal import (sosfiltfilt, butter)
from collections import namedtuple
def bandpass(data, fmin, fmax, fs, order):
    """
    Butterworth-Bandpass Filter assuming order > 2 (and therefore applying sos filtering) with zero-phase filtering.
    """
    nyquist = 0.5 * fs
    if fmax > nyquist or fmin > nyquist:
        raise ValueError(f"Selected corner frequencies ({fmin}, {fmax}) conflict with Nyquist frequency ({nyquist}).")

    bounds = (fmin / nyquist, fmax / nyquist)
    sos = butter(order, bounds, btype='band', output='sos')
    return sosfiltfilt(sos, data)


def lfp2bandpass(lfps, fs, fmin=0.05, fmax=300):
    # convert array of LFPs to bandpass filtered LFPs
    return np.array([bandpass(lfps[:, i], fmin, fmax, fs) for i in range(len(lfps[0]))])

# get CSD - first do a lowpass filter. lfps is a list or numpy array of LFPs arranged spatially by column
# spacing_um is electrode's contact spacing in units of micron
# returns CSD in units of mV/mm**2 (assuming lfps are in mV)
def lfp2csd (lfps,fs,spacing_um=100.0,fmin=0.05,fmax=300):
    band = lfp2bandpass(lfps,fs,fmin,fmax)
    ax = band.shape[0] > band.shape[1] and 1 or 0
    # can change to run Vaknin on bandpass filtered LFPs before calculating CSD, that
    # way would have same number of channels in CSD and LFP (but not critical, and would take more RAM);
    # also might want to subtract mean of each channel before calculating the diff(diff) ?
    band -= np.mean(band, axis=ax, keepdims=True)
    spacing_mm = spacing_um/1000 # spacing in mm
    # now each column (or row) is an electrode -- CSD along electrodes
    return -np.diff(band, n=2, axis=ax)/spacing_mm**2

def index2ms (i, fs): return 1e3*i/fs
def ms2index (t, fs):
    match t:
        case np.ndarray():
            return (t*fs/1e3).astype(int)
        case list():
            return [int(x*fs/1e3) for x in t]
        case _:
            return int(t*fs/1e3)


def raw2avgerp (data, fs, event_times, response_duration):
    """
    raw2avgerp (data, fs, event_times, response_duration)
    :param data: numpy.ndarray, shape (nchannels, electrophysiology data)
    :param fs: sampling rate (frequency sampling in Hz)
    :param event_times: numpy.ndarray, shape (1, event times in ms)
    :param response_duration: window of ERP, in ms
    :return:
        erp.window -> timestamped ERP window
        erp.avg -> average ERP, numpy.ndarray, shape (nchannels, ERP window)

    """
    erp = namedtuple('erp', ['window', 'avg'])
    event_indices = ms2index(event_times, fs)
    nchannels = data.shape[0]
    len_erp_window = ms2index(response_duration, fs)
    erp_window = np.linspace(0, response_duration, len_erp_window)
    erp_avg = np.zeros((nchannels, len_erp_window))
    for i in range(nchannels): # go through channels
        for j in event_indices: # go through stimuli
            erp_avg[i, :] += data[i, j: j+len_erp_window]
        erp_avg[i, :] /= len(event_times)
    return erp(erp_window, erp_avg)

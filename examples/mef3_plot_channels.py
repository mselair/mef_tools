import os
import sys

from best.files import get_files
from best.signal import PSD, buffer
from best.modules.feature import PCAModule

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from mef_tools.io import MefReader
from tqdm import tqdm


from datetime import datetime


path_output = '/path_to_output_folder'

time_limit = 24*3600 # sec
time_step = 10 # sec
bands = [
    [0.5, 4],
    [4, 8],
    [8, 12],
]

files = [] # list of paths to mefd files

for fid in tqdm(files):
    Rdr = MefReader(fid)
    start = Rdr.get_property('start_time')[0]
    end = Rdr.get_property('end_time')[0]

    if ((end - start)/1e6) > time_limit:
        end = start + (time_limit*1e6)

    for ch in Rdr.channels:
        name = f"{ch}_{fid.split(os.path.sep)[-1]}"

        fs = Rdr.get_property('fsamp', ch)
        x = Rdr.get_data(ch, start, end)

        nans = np.isnan(x)
        x[nans] = np.nanmean(x)

        b_low, a_low = signal.butter(7, np.max(bands), 'low', fs=fs)
        b_high, a_high = signal.firwin(501, 0.5, fs=fs), [1]

        x = signal.filtfilt(b_low, a_low, x)
        x = x - signal.filtfilt(b_high, a_high, x)

        x[nans] = np.nan

        t = np.arange(x.shape[0]) / fs
        t = buffer(t, fs=fs, segm_size=time_step)[:, 0]
        t = t / 3600
        t = t + datetime.fromtimestamp(start / 1e6).hour + datetime.fromtimestamp(start / 1e6).minute / 60

        f, sxx = PSD(buffer(x, fs=fs, segm_size=time_step), fs=fs)
        sxx = sxx.T

        sxx_plot = sxx[f < 30][1:]
        f_plot = f[f < 30][1:]

        plt.plot(f_plot, np.nanmean(sxx_plot, 1))
        plt.savefig(os.path.join(path_output, f'{name}_spectrum.png'))
        plt.show()

        ftrs = []
        fig = plt.figure(figsize=(24, 12))
        ax = plt.subplot(bands.__len__()+2,1,1)
        plt.pcolormesh(t, f_plot, sxx_plot, shading='gouraud', vmin=0, vmax=np.nanquantile(sxx_plot, 0.95))
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=16)
        plt.ylabel('f [Hz]', fontsize=18)
        plt.title(f"{name}", fontsize=18)

        ax = plt.subplot(bands.__len__()+2,1,2, sharex=ax)
        x_pib = np.nanmean(sxx[(f >= 0.5) & (f <= 4), :], 0) / np.nanmean(sxx[(f >= 8) & (f <= 12), :], 0)
        x_pib = np.log10(x_pib)
        nans = np.isnan(x_pib)
        x_pib[nans] = np.nanmean(nans)
        x_pib = signal.medfilt(x_pib, 5)
        x_pib[nans] = np.nan
        ftrs += [x_pib]

        plt.plot(t, x_pib)
        plt.grid(True)
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=16)
        plt.ylabel(f"D/B", fontsize=18)

        for idx, pib in list(enumerate(bands)):
            x_pib = np.nanmean(sxx[(f >= pib[0]) & (f <= pib[1]), :], 0)
            nans = np.isnan(x_pib)
            x_pib[nans] = np.nanmean(nans)
            x_pib = signal.medfilt(x_pib, 3)
            # x_pib = signal.savgol_filter(x_pib, 51, 3)
            x_pib[nans] = np.nan
            x_pib = np.log10(x_pib)
            ftrs += [x_pib]

            plt.subplot(bands.__len__()+2,1,idx+2+1, sharex=ax)
            plt.plot(t, x_pib)
            plt.grid(True)
            plt.xticks(fontsize=0)
            plt.yticks(fontsize=16)
            plt.ylabel(f"{pib[0]}-{pib[1]} Hz", fontsize=18)

        plt.xlabel('Time [hrs]', fontsize=24)
        plt.savefig(os.path.join(path_output, f'{name}_spectrum.png'))
        plt.show()

        ftrs = np.array(ftrs).T
        ftrs = ftrs[np.isnan(ftrs).sum(1) == 0]

        xb = PCAModule().fit_transform(ftrs)
        plt.plot(xb[:, 0], xb[:, 1], '.', markersize=0.5, alpha=1)
        plt.savefig(os.path.join(path_output, f'{name}_PCA.png'))
        plt.show()





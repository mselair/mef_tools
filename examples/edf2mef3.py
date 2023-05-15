import os

import numpy as np
import pandas as pd

from datetime import datetime
from tqdm import tqdm

import mne
from mef_tools.io import MefWriter



DELIMITER = os.path.sep



paths = [] # list of EDF files to convert

# export to
path_export = f"/path_to_export_folder"

# not sure if reading all types of EDF files is supported by mne
for path_edf in tqdm(paths):
    filename = path_edf.split(DELIMITER)[-1][:-4]
    print(filename)
    path_write = os.path.join(path_export, filename + '.mefd')
    path_hyp = os.path.join(path_export, filename + '.mefd', 'hypnogram.csv')

    Wrt = MefWriter(path_write, overwrite=True)

    data = mne.io.read_raw_edf(path_edf)
    info = data.info
    annotations = data.annotations

    channels = data.info.ch_names
    fsamp = data.info['sfreq']
    start = data.annotations.orig_time.timestamp()

    Wrt.mef_block_len = int(fsamp)
    Wrt.max_nans_written = 0

    if start == 0:
        start = datetime(year=2000, month=1, day=1, hour=0).timestamp()

    for ch in channels:
        Wrt.data_units = data._orig_units[ch]
        x = data.get_data(ch).squeeze() * 1e6

        if '/' in ch: ch = ch.replace('/', '-')
        Wrt.write_data(x, ch, start_uutc=start * 1e6, sampling_freq=fsamp, reload_metadata=False,)


    hyp = pd.DataFrame([{'duration': dur, 'start': start+onset, 'end': start+onset+dur, 'annotation':annot} for dur, onset, annot in zip(data.annotations.duration,
    data.annotations.onset,
    data.annotations.description) if annot in ['Sleep stage N1',
           'Sleep stage N2', 'Sleep stage N3', 'Sleep stage R',
           'Sleep stage W']]
    )

    hyp.to_csv(path_hyp, index=False)


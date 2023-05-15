import os
import pandas as pd
import mne

import numpy as np

from datetime import datetime

from mef_tools.io import MefWriter, MefReader

from tqdm import tqdm


path_input = '/path_to_the_original_mef_file.mefd'
path_new = '/path_to_new_mef_file.mefd'

time_step = 4*60*60 # process in 4 hour chunks


Rdr = MefReader(path_input)
fs = Rdr.get_property('fsamp')
print('#########################')
print(f'Session {path_input}')
print(f'Sample rate {max(fs)}')
print('#########################')

if max(fs) > 256:
    print('#########################')
    print(f'Processing {path_input}')
    print('#########################')

    start = min(Rdr.get_property('start_time'))
    end = max(Rdr.get_property('end_time'))
    ts_steps = np.arange(start, end, time_step * 1e6)

    Wrt = MefWriter(path_new, overwrite=True)
    Wrt.max_nans_written = 0
    Wrt.record_offset = 0
    Wrt.section3_dict['GMT_offset'] = 0
    for idx, ts in tqdm(list(enumerate(ts_steps))):
        s_ = ts
        e_ = s_ + (time_step * 1e6)

        for ch in Rdr.channels:
            fs = Rdr.get_property('fsamp', ch)
            unit = Rdr.get_property('unit', ch)
            mef_block = int(np.round(60*fs))
            precision = Rdr.get_property('ufact', ch)

            Wrt.mef_block_len = mef_block
            Wrt.data_units = unit

            x = Rdr.get_data(ch, s_, e_)

            if np.isnan(x).sum() < x.shape[0]:
                Wrt.write_data(x, ch, s_, sampling_freq=fs, precision=3, reload_metadata=False)







import os

import numpy as np
import pandas as pd

from datetime import datetime
from tqdm import tqdm


from mef_tools.io import MefWriter, MefReader


DELIMITER = os.path.join(' ', ' ')[1]

paths_to_merge = [
    "/path_to_file1.mefd",
    "/path_to_file2.mefd",
    "/path_to_file3.mefd",
] # must be sorted by time

output_mef = "/path_to_output_file.mefd"

Wrt = MefWriter(output_mef, overwrite=True)
Wrt.max_nans_written = 0
Wrt.record_offset = 0
Wrt.section3_dict['GMT_offset'] = 0

for fidx, fid in enumerate(paths_to_merge):
    print(f"Processing file {fidx+1} - {fid}")
    Rdr = MefReader(fid)

    start = min(Rdr.get_property('start_time'))
    end = max(Rdr.get_property('end_time'))

    for ch in tqdm(Rdr.channels):
        fs = Rdr.get_property('fsamp', ch)
        unit = Rdr.get_property('unit', ch)
        mef_block = int(10000)
        precision = 1
        x = Rdr.get_data(ch, start, end)

        Wrt.mef_block_len = mef_block
        Wrt.data_units = unit

        if np.isnan(x).sum() < x.shape[0]:
            Wrt.write_data(x, ch, start, sampling_freq=fs, precision=precision, reload_metadata=False)




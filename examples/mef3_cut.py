import os

import numpy as np
import pandas as pd

from datetime import datetime
from tqdm import tqdm

from mef_tools.io import MefWriter, MefReader


input_mef = "/path_to_input_file.mefd"
output_mef = "/path_to_output_file.mefd"

from_time = 1658277000000000 # uUTC timestamp to extract from
to_time = 1658336400000000 # uUTC timestamp to extract to

Wrt = MefWriter(output_mef, overwrite=True)
Wrt.max_nans_written = 0
Wrt.record_offset = 0
Wrt.section3_dict['GMT_offset'] = 0

Rdr = MefReader(input_mef)

start = from_time
end = to_time
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



from mef_tools import MefReader, MefWriter
import os
import numpy as np


from mef_tools.segment import *

# from datetime import datetime

path = '/Users/mivalt.filip/mef_tools_new/mef_tools_new/test.mefd'

wrt = MefWriter(path, overwrite=True, password1='kokot', password2='kokot.')
wrt.mef_block_len = 100
wrt.data_units = 'µV'
wrt.max_nans_written = 0

start = datetime.now().timestamp() * 1e6
x = np.random.randn(1005)
wrt.write_data(x, 'ch1', 250, start)

rdr = MefReader(path, 'kokot.')

self = UniversalHeader(path = os.path.join(path, 'ch1.timd', 'ch1-000000.segd', 'ch1-000000.tmet'))

self.check_password('kokot')
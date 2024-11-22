import os
import queue
import tempfile
import shutil
import pytest
import sys
import logging
import socket
import time
import threading

import numpy as np

from datetime import datetime
from mef_tools import MefWriter

@pytest.fixture()
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture()
def mef_session_no_enc(temp_dir: str):
    session = os.path.join(temp_dir, 'test_no_enc.mefd')
    start = datetime.now().timestamp()
    wrt = MefWriter(
        session, overwrite=True
    )
    wrt.mef_block_len = 1000
    wrt.max_nans_written = 0

    fs = 100
    dur_sec = 60

    freq1 = 1
    x0 = np.sin(freq1 * 2 * np.pi * np.arange(fs*dur_sec)/fs)

    freq2 = 10
    x1 = np.sin(freq2 * 2 * np.pi * np.arange(fs*dur_sec)/fs)

    print(wrt.section2_ts_dict)
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(wrt.section3_dict)

    wrt.write_data(x0, 'ch1', start*1e6, fs)
    wrt.write_data(x1, 'ch2', start*1e6, fs)
    wrt.session.close()
    yield (session, wrt.section2_ts_dict, wrt.section3_dict)


@pytest.fixture()
def mef_session_enc(temp_dir: str):
    session = os.path.join(temp_dir, 'test_no_enc.mefd')
    start = datetime.now().timestamp()
    wrt = MefWriter(
        session, overwrite=True, password1='kokot', password2='kokot.'
    )
    wrt.mef_block_len = 1000
    wrt.max_nans_written = 0

    fs = 100
    dur_sec = 60

    freq1 = 1
    x0 = np.sin(freq1 * 2 * np.pi * np.arange(fs*dur_sec)/fs)

    freq2 = 10
    x1 = np.sin(freq2 * 2 * np.pi * np.arange(fs*dur_sec)/fs)

    print(wrt.section2_ts_dict)
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(wrt.section3_dict)

    wrt.write_data(x0, 'ch1', start*1e6, fs)
    wrt.write_data(x1, 'ch2', start*1e6, fs)
    wrt.session.close()
    yield (session, wrt.section2_ts_dict, wrt.section3_dict)

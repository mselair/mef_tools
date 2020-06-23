# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import numpy as np
from shutil import rmtree
from pymef import mef_session
from pymef.mef_session import MefSession

from AISC.utils.types import ObjDict


class MefReader:
    __version__ = '1.0.0'

    def __init__(self, mef_path, password=''):
        self.session = mef_session.MefSession(mef_path, password, True)
        self.bi = self.session.read_ts_channel_basic_info()
        self.channels = [channel['name'] for channel in self.bi]

    def __del__(self):
        self.session.close()

    def get_data(self, channels, t_stamp1, t_stamp2):
        channels_to_pick = []

        if isinstance(channels, int):
            if channels < self.channels.__len__():
                channels_to_pick = [self.channels[channels]]
            else:
                raise ValueError('Number of channels in MEF file: ' + str(self.channels.__len__()) + '. However index ' + str(channels) + ' pasted')

        if isinstance(channels, str):
            if channels in self.channels:
                channels_to_pick = [channels]
            else:
                raise ValueError('Channel name is not present in MEF file.')


        if isinstance(channels, (list, np.ndarray)):
            for channel in channels:
                if isinstance(channel, int):
                    if not self.channels[channel] in channels_to_pick:
                        channels_to_pick.append(self.channels[channel])

                if isinstance(channel, str):
                    if (not channel in channels_to_pick) and channel in self.channels:
                        channels_to_pick.append(channel)

        return self.session.read_ts_channels_uutc(channels_to_pick, [t_stamp1, t_stamp2])


class MefWritter:

    __version__ = '1.0.0'

    def __init__(self, session_path=None, overwrite=False, password=''):
        self.bi = None
        self.recording_offset = None
        self.channels = None
        self.samps_mef_block = 300


        self.section3_dict = ObjDict(
            {
                  'recording_time_offset': np.nan,
                  'DST_start_time': 0,
                  'DST_end_time': 0,
                  'GMT_offset': -6*3600,
                  'subject_name_1': b'test',
                  'subject_name_2': b'test',
                  'subject_ID': b'None',
                  'recording_location': b'P'
            })


        self.section2_ts_dict = ObjDict(
            {
                 'channel_description': b'Test_channel',
                 'session_description': b'Test_session',
                 'recording_duration': np.nan,  # TODO:test 0 / None
                 'reference_description': b'None',
                 'acquisition_channel_number': 1,
                 'sampling_frequency': np.nan,
                 'notch_filter_frequency_setting': 0,
                 'low_frequency_filter_setting': 1,
                 'high_frequency_filter_setting': 10,
                 'AC_line_frequency': 0,
                 'units_conversion_factor': 0.0001,
                 'units_description': b'uV',
                 'maximum_native_sample_value': 0.0,
                 'minimum_native_sample_value': 0.0,
                 'start_sample': 0,  # Different for segments
                 'number_of_blocks': 0,
                 'maximum_block_bytes': 0,
                 'maximum_block_samples': 0,
                 'maximum_difference_bytes': 0,
                 'block_interval': 0,
                 'number_of_discontinuities': 0,
                 'maximum_contiguous_blocks': 0,
                 'maximum_contiguous_block_bytes': 0,
                 'maximum_contiguous_samples': 0,
                 'number_of_samples': 0
            })

        if overwrite is True:
            if os.path.exists(session_path):
                rmtree(session_path)
                time.sleep(2) # wait till all files are gone. Problems when many files, especially on a network drive
            self.session = MefSession(session_path, password, False, True)

        else:
            self.session = MefSession(session_path, password)
            self.bi = self.session.read_ts_channel_basic_info()
            self.channels = [channel['name'] for channel in self.bi]

    def __del__(self):
        self.session.close()

    def create_segment(self, data=None, channel=None, start_stamp=None, end_stamp=None, sampling_frequency=None, pwd1=None, pwd2=None):
        if data.dtype != np.int32:
            raise AssertionError('[TYPE ERROR] - MEF file writer accepts only int32 signal datatype!')
        data = np.append(0, data)
        self.section3_dict['recording_time_offset'] = int(start_stamp - 1e6)
        self.section2_ts_dict['sampling_frequency'] = sampling_frequency
        self.section2_ts_dict['recording_duration'] = int((end_stamp - start_stamp) / 1e6)
        self.section2_ts_dict['start_sample'] = 0

        self.session.write_mef_ts_segment_metadata(channel,
                                                   0,
                                                   pwd1,
                                                   pwd2,
                                                   start_stamp,
                                                   end_stamp,
                                                   dict(self.section2_ts_dict),
                                                   dict(self.section3_dict))

        self.session.write_mef_ts_segment_data(channel,
                                               0,
                                               pwd1,
                                               pwd2,
                                               self.samps_mef_block,
                                               data)


    def append_block(self, data=None, channel=None, start_stamp=None, end_stamp=None, pwd1=None, pwd2=None):
        self.session.append_mef_ts_segment_data(channel,
                                      int(0),
                                      pwd1,
                                      pwd2,
                                      start_stamp,
                                      end_stamp,
                                      self.samps_mef_block,
                                      data)


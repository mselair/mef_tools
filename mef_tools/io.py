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
import pandas as pd
#from AISC.utils.types import ObjDict
from copy import deepcopy

SECTION2_TS_DICT = {
    'channel_description': b'ts_channel',
    'session_description': b'ts_session',
    'recording_duration': np.nan,  # TODO:test 0 / None
    'reference_description': b'None',
    'acquisition_channel_number': 1,
    'sampling_frequency': np.nan,
    'notch_filter_frequency_setting': 0,
    'low_frequency_filter_setting': 1,
    'high_frequency_filter_setting': 10,
    'AC_line_frequency': 0,
    'units_conversion_factor': 1.0,
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
}

SECTION3_TS_DICT = {
                  'recording_time_offset': np.nan,
                  'DST_start_time': 0,
                  'DST_end_time': 0,
                 # default Chicago timezone
                  'GMT_offset': -6*3600,
                  'subject_name_1': b'none',
                  'subject_name_2': b'none',
                  'subject_ID': b'None',
                  'recording_location': b'P'
            }



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

    def __init__(self, session_path, overwrite=False, password1=None, password2=None):
        # TODO overwrite - > new_session / appending to new session -> reading segments and info from written data -> this will limit
        #  scale factor / write new channels
        # TODO handling annotations (records)
        self.pwd1 = password1
        self.pwd2 = password2
        self.bi = None
        self.channel_info = {}
        self._max_nans_written = 'fs'
#        self.recording_offset = None
#        self.channels = None
        # TODO mef_bock infered/setup
       # self.samps_mef_block = 300


        self.section3_dict = {
                  'recording_time_offset': np.nan,
                  'DST_start_time': 0,
                  'DST_end_time': 0,
                  'GMT_offset': -6*3600,
                  'subject_name_1': b'none',
                  'subject_name_2': b'none',
                  'subject_ID': b'None',
                  'recording_location': b'P'
            }


        self.section2_ts_dict =   {
                 'channel_description': b'ts_channel',
                 'session_description': b'ts_session',
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
            }

        if overwrite is True:
            if os.path.exists(session_path):
                rmtree(session_path)
                time.sleep(3) # wait till all files are gone. Problems when many files, especially on a network drive
            self.session = MefSession(session_path, password2, False, True)
        else:
            if os.path.exists(session_path):
                self.session = MefSession(session_path, password2, False, False)
                self._reload_session_info()
            else:
                self.session = MefSession(session_path, password2, False, True)

    def __del__(self):
        self.session.close()

    def _reload_session_info(self):
        self.session.reload()
        self.bi = self.session.read_ts_channel_basic_info()
        self.channel_info = {info['name']: deepcopy(info) for info in self.bi}
        for ch in self.channel_info.keys():
            self.channel_info[ch]['n_segments'] = len(self.session.session_md['time_series_channels'][ch]['segments'])
            self.channel_info[ch]['mef_block_len'] = int(self.get_mefblock_len(self.channel_info[ch]['fsamp'][0]))
            # TODO only one write method (should handle fragmenting and append/create)
    # TODO write method handling the fragmenting of data and writing
    # TODO write method of check data integrity upon write
    # TODO write progress bar
    # TODO write method which checks/reload actual metadata
    def write_data(self, data_write, channel, start_uutc, end_uutc, sampling_freq, precision=None, new_segment=False,
                   discont_handler=True):
        # check times are correct
        if end_uutc < start_uutc:
            print(f"WARNING: incorrect End uutc time {end_uutc} is before beginning: {start_uutc}")
            return None

        # check if any data exists
        if channel in self.channel_info.keys():
            # check if it is possible to write with configuration provided
            if start_uutc < self.channel_info[channel]['end_time'][0]:
                print(' Start time provided is before end time of data already written to the session. Returning None')
                return None
            # NOTE fs can be different in the new segment but we dont work with different fs in the same channel
            if sampling_freq != self.channel_info[channel]['fsamp'][0]:
                print(' Sampling frequency of provided data does not match fs of already written data')
                return None
            # read precision from metadata - scale factor / can be different in new segment but not implemented
            precision = int(-1 * np.log10(self.channel_info[channel]['ufact'][0]))

            # convert data to int32
            data_converted = convert_data_to_int32(data_write, precision=precision)

            # discont handler writes fragmented intervals ( skip nans greater than specified)
            if discont_handler:
                if self.max_nans_written == 'fs':
                    max_nans = sampling_freq
                else:
                    max_nans = self.max_nans_written

                input_bin_vector = ~np.isnan(data_write)
                df_intervals = find_intervals_binary_vector(input_bin_vector, sampling_freq, start_uutc, samples_of_nans_allowed=max_nans)

            # check new segment flag
            segment = self.channel_info[channel]['n_segments']
            if new_segment:
                self._create_segment(data=data_converted, channel=channel, start_uutc=start_uutc, end_uutc=end_uutc,
                                     sampling_frequency=sampling_freq, segment=segment)
            # append to last segment
            else:
                segment -= 1
                self._append_block(data=data_converted, channel=channel, start_uutc=start_uutc, end_uutc=end_uutc, segment=segment)

        # new channel data with no previous data
        else:
            segment = 0
            if precision is None:
                precision = infer_conversion_factor(data_write)

            ufact = 0.1**precision
            # convert data to int32
            self.channel_info[channel] = {'mef_block_len': self.get_mefblock_len(sampling_freq), 'ufact': [ufact]}
            data_converted = convert_data_to_int32(data_write, precision=precision)
            self._create_segment(data=data_converted, channel=channel, start_uutc=start_uutc, end_uutc=end_uutc,
                                 sampling_frequency=sampling_freq, segment=0)

        self._reload_session_info()
        return True

    def _create_segment(self, data=None, channel=None, start_uutc=None, end_uutc=None, sampling_frequency=None, segment=0, ):
        if data.dtype != np.int32:
            raise AssertionError('[TYPE ERROR] - MEF file writer accepts only int32 signal datatype!')

        if end_uutc < start_uutc:
            raise ValueError('End uutc timestamp lower than the start_uutc')

        self.section3_dict['recording_time_offset'] = int(start_uutc - 1e6)
        self.section2_ts_dict['sampling_frequency'] = sampling_frequency
        # recording time offset is fixed ?
        if segment == 0:
            self.section3_dict['recording_time_offset'] = int(start_uutc - 1e6)
            self.section2_ts_dict['start_sample'] = 0
        else:
            self.section3_dict['recording_time_offset'] = int(self.channel_info[channel]['start_time'][0] - 1e6)
            self.section2_ts_dict['start_sample'] = int(self.channel_info[channel]['nsamp'][0])

        self.section2_ts_dict['recording_duration'] = int((end_uutc - start_uutc) / 1e6)
        self.section2_ts_dict['units_conversion_factor'] = self.channel_info[channel]['ufact'][0]

        print(f"INFO: writing data for channel-> {channel}, segment-> {segment}, fs: {sampling_frequency}, ufac:"
              f" {self.channel_info[channel]['ufact'][0]}, start: {start_uutc}, stop {end_uutc} ")
        self.session.write_mef_ts_segment_metadata(channel,
                                                   int(segment),
                                                   self.pwd1,
                                                   self.pwd2,
                                                   start_uutc,
                                                   end_uutc,
                                                   dict(self.section2_ts_dict),
                                                   dict(self.section3_dict))

        self.session.write_mef_ts_segment_data(channel,
                                               int(segment),
                                               self.pwd1,
                                               self.pwd2,
                                               self.channel_info[channel]['mef_block_len'],
                                               data)

    def _append_block(self, data=None, channel=None, start_uutc=None, end_uutc=None, segment=0):

        if end_uutc < start_uutc:
            raise ValueError('End uutc timestamp lower than the start_uutc')
        print(f"INFO: appending data for channel-> {channel}, segment-> {segment}, ufac:"
              f" {self.channel_info[channel]['ufact'][0]}, start: {start_uutc}, stop {end_uutc} ")
        self.session.append_mef_ts_segment_data(channel,
                                                  int(segment),
                                                  self.pwd1,
                                                  self.pwd2,
                                                  start_uutc,
                                                  end_uutc,
                                                  self.channel_info[channel]['mef_block_len'],
                                                  data)

    @staticmethod
    def get_mefblock_len(fs):
        if fs >= 5000:
            return fs
        else:
            return fs * 10

    @property
    def max_nans_written(self):
        return self._max_nans_written

    @max_nans_written.setter
    def max_nans_written(self, max_samples):
        self._max_nans_written = max_samples


# Functions
def voss(nrows, ncols=32):
    """Generates pink noise using the Voss-McCartney algorithm.

    nrows: number of values to generate
    rcols: number of random sources to add

    returns: NumPy array
    """
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)

    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return total.values


def create_pink_noise(fs, seg_len, low_bound, up_bound):
    n = fs * seg_len
    if n > 20 * 1e6:
        raise ValueError('too many samples to generate')
    # if
    data = voss(n)
    norm_data = scale_signal(data, low_bound, up_bound)
    return norm_data


def scale_signal(data, a, b):
    min_x = np.min(data)
    data_range = np.max(data) - min_x
    temp_arr = (data - min_x) / data_range
    new_range = b - a
    return temp_arr * new_range + a


def check_int32_dynamic_range(x_min, x_max, alpha):
    min_value = np.iinfo(np.int32).min
    if (x_min * alpha < min_value) & (x_max * alpha > np.iinfo(np.int32).max):
        return False
    else:
        return True


def infer_conversion_factor(data):
    mean_digg_abs = np.nanmean(np.abs(np.diff(data)))
    precision = 1
    # this works for smal z-scored data, for high dynamic range input needs to be decreased again (saturation)
    while mean_digg_abs < 100:
        precision += 1
        mean_digg_abs *= 10

    data_max = np.nanmax(data)
    data_min = np.nanmin(data)
    alpha = 10 ** precision
    while (not check_int32_dynamic_range(data_min, data_max, alpha)) & (precision != 0):
        precision -= 1
        print(f" WARNING: dynamic range saturated, precision decreased to {precision}")
        alpha = 10 ** precision
    return precision


def convert_data_to_int32(data, precision=None):
    if precision is None:
        print(f"Info: convert data to int32:  precision is not given, infering...")
        precision = infer_conversion_factor(data)
        print(f"Info: precision set to {precision}")

    if (precision < 0) | (not (isinstance(precision, int))):
        print(f"WARNING: precision set to incorrect value, it is set to default (3)")
        precision = 3

    deciround = np.round(data, decimals=precision)
    data_int32 = np.empty(shape=deciround.shape, dtype=np.int32)
    data_int32[:] = 10 ** precision * (deciround)
    return data_int32


def find_intervals_binary_vector(input_bin_vector, fs, start_uutc, samples_of_nans_allowed=None):
    if samples_of_nans_allowed is None:
        samples_of_nans_allowed = fs

    vector = np.concatenate((np.array([0]), input_bin_vector, np.array([0])))
    diff_vector = np.diff(vector)
    # find start and stop position of intervals with continuous ones
    t0 = np.where(diff_vector == 1)[0]
    t1 = np.where(diff_vector == -1)[0]

    # merge neighbors with gap les than samples_of_nans_allowed
    segments = pd.DataFrame()
    segments['start_samples'] = t0
    segments['stop_samples'] = t1

    # merge neighbors ( find overlaps and get the rest (noverlaps))
    tmp_vec = np.array(segments.iloc[:-1, 1] + samples_of_nans_allowed) > np.array(segments.iloc[1:, 0])
    diff_vector = np.concatenate((np.array([0]), tmp_vec, np.array([0])))
    bin_det = diff_vector[1:]
    diff = np.diff(diff_vector)
    # get overlap intervals
    t0 = np.where(diff == 1)[0]
    t1 = set(np.where(diff == -1)[0])
    # get noverlaps segments
    t3 = set(np.where(bin_det == 0)[0])
    t_noverlap = np.sort(list(t3 - (t3 & t1)))
    t1 = np.sort(list(t1))

    # overlap segments (nans inside this interval will be stored)
    overlap_starts = np.array(segments.loc[t0, 'start_samples'])
    overlap_ends = np.array(segments.loc[t1, 'stop_samples'])

    # lonely segments
    lonely_segments = segments.loc[t_noverlap, :]

    # final fragment segments
    connected_detected_intervals = pd.DataFrame(columns=['start_samples', 'stop_samples', ])
    connected_detected_intervals['start_samples'] = overlap_starts.astype(int)
    connected_detected_intervals['stop_samples'] = overlap_ends.astype(int)
    connected_detected_intervals = connected_detected_intervals.append(lonely_segments, ignore_index=True)
    connected_detected_intervals = connected_detected_intervals.sort_values(by='start_samples').reset_index(drop=True)

    # calculate uutc time of intervals
    connected_detected_intervals['start_uutc'] = (connected_detected_intervals['start_samples'] / fs * 1e6 + start_uutc).astype(int)
    connected_detected_intervals['stop_uutc'] = (connected_detected_intervals['stop_samples'] / fs * 1e6 + start_uutc).astype(int)
    return connected_detected_intervals


def check_data_integrity(original_data, converted_data, precision):

    coverted_float = 0.1**precision*(converted_data)
    idx_numbers = ~np.isnan(original_data)
    result_bin = np.allclose(coverted_float[idx_numbers], original_data[idx_numbers], atol=0.1**(precision-1))
    return result_bin
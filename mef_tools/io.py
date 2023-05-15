# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from shutil import rmtree
from copy import deepcopy, copy

import pandas as pd
import numpy as np
from numpy import int64

from pymef import mef_session
from pymef.mef_session import MefSession


class MefReader:
    """
    Class to handle reading of MEF files.

    Attributes
    ----------
    __version__ : str
        Version of the MefReader class.
    """

    __version__ = '2.0.0'

    def __init__(self, session_path, password2=None):
        """
                Initializes the MefReader object.

                Parameters
                ----------
                session_path : str
                    Path to the MEF session.
                password2 : str, optional
                    Secondary password for the session. Default is None.
        """
        self.session = mef_session.MefSession(session_path, password2, True)
        self.bi = self.session.read_ts_channel_basic_info()

        for ch_info in self.bi:
            if ch_info['fsamp'].__len__() > 1:
                raise NotImplementedError('[ERROR]: File contains more sampling frequencies '
                                          'for a single channels. This feature is not implemented.')

    def __del__(self):
        """
        Destructor for the MefReader object, ensures the session is closed.
        """
        self.close()

    @property
    def channels(self):
        """
        Returns a list of all channels present in the session.

        Returns
        -------
        list
            List of channels.
        """
        return [ch_info['name'] for ch_info in self.bi]

    @property
    def properties(self):
        """
        Returns a list of all unique properties across all channels in the session.

        Returns
        -------
        list
            List of unique properties.
        """
        properties = []
        for ch_info in self.bi:
            properties += list(ch_info.keys())
        return list(np.unique(properties))

    def get_property(self, property_name, channel=None):
        """
        Returns the specified property for a given channel. If no channel is specified,
        returns the property for all channels.

        Parameters
        ----------
        property_name : str
            Name of the property.
        channel : str, optional
            Name of the channel. If not provided, method returns property for all channels.

        Returns
        -------
        list or str
            Property or list of properties.
        """
        if isinstance(channel, type(None)):
            props = []
            for ch_info in self.bi:
                if ch_info[property_name].__len__() == 1:
                    props.append(ch_info[property_name][0])
                else:
                    props.append(ch_info[property_name])
            return props

        for ch_info in self.bi:
            if ch_info['name'] == channel:
                if ch_info[property_name].__len__() == 1:
                    return ch_info[property_name][0]
                return ch_info[property_name]
        return None

    def get_channel_info(self, channel=None):
        """
        Returns information for a given channel. If no channel is specified,
        returns information for all channels.

        Parameters
        ----------
        channel : str, optional
            Name of the channel. If not provided, method returns info for all channels.

        Returns
        -------
        dict or list
            Channel info or list of channel info.
        """
        if isinstance(channel, type(None)):
            return self.bi

        for ch_info in self.bi:
            if ch_info['name'] == channel:
                return ch_info
        return None

    def close(self):
        """
        Closes the MEF session.
        """
        self.session.close()

    def get_raw_data(self, channels, t_stamp1=None, t_stamp2=None):
        """
        Returns raw data for specified channels and time stamps.

        Parameters
        ----------
        channels : int64, str, list, or numpy.ndarray
            Channels for which to return data.
        t_stamp1 : int64, optional
            Start time stamp. If not provided, method uses the earliest time stamp.
        t_stamp2 : int64, optional
            End time stamp. If not provided, method uses the latest time stamp.

        Returns
        -------
        numpy.ndarray
            Array of raw data.
        """
        channels_to_pick = []

        if isinstance(channels, int64):
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
                if isinstance(channel, int64):
                    if not self.channels[channel] in channels_to_pick:
                        channels_to_pick.append(self.channels[channel])

                if isinstance(channel, str):
                    if (not channel in channels_to_pick) and channel in self.channels:
                        channels_to_pick.append(channel)

        if isinstance(t_stamp1, type(None)):
            t_stamp1 = min([self.get_property('start_time', channel) for channel in self.channels if channel in channels_to_pick])

        if isinstance(t_stamp2, type(None)):
            t_stamp2 = min([self.get_property('end_time', channel) for channel in self.channels if channel in channels_to_pick])

        return self.session.read_ts_channels_uutc(channels_to_pick, [t_stamp1, t_stamp2])

    def get_data(self, channels, t_stamp1=None, t_stamp2=None):
        """
        Returns processed data for specified channels and time stamps.

        Parameters
        ----------
        channels : int64, str, list, or numpy.ndarray
        Channels for which to return data.
        t_stamp1 : int64, optional
            Start time stamp. If not provided, method uses the earliest time stamp.
        t_stamp2 : int64, optional
            End time stamp. If not provided, method uses the latest time stamp.

        Returns
        -------
        numpy.ndarray
            Array of processed data.

        """
        data = self.get_raw_data(channels, t_stamp1, t_stamp2)
        if isinstance(channels, list):
            for idx, ch_name in enumerate(channels):
                data[idx] = data[idx].astype(np.float64) * self.get_channel_info(ch_name)['ufact'][0]
        else:
            data = data[0].astype(np.float64) * self.get_channel_info(channels)['ufact'][0]
        return data

    def get_annotations(self, channel=None):
        """
        Returns annotations for a specified channel. If no channel is specified,
        returns annotations for all channels.

        Parameters
        ----------
        channel : str, optional
            Name of the channel. If not provided, method returns annotations for all channels.

        Returns
        -------
        list
            List of annotations.
        """
        annot_list = None
        try:
            if channel is None:
                annot_list = self.session.read_records()
            else:
                annot_list = self.session.read_records(channel=channel)
        except TypeError as exc:
            print('WARNING: read of annotations record failed, no annotations returned')
        return annot_list

class MefWriter:
    """
    MefWriter is a utility class for writing data in the MEF3 format. The class allows easy writing and appending of data to existing MEF3 files.

    Attributes:
        session_path: The path of the MEF3 session to be written.
        overwrite: A boolean flag that if set to True, allows overwriting of existing files. Default is False.
        password1: The password for level 1 encryption. Default is None. This password is needed only for while creating the session.
        password2: The password for level 2 encryption. Default is None. This password is required for any read/write operation of an existing session.
        verbose: A boolean flag that if set to True, enables verbose mode. Default is False.
    """
    __version__ = '2.0.0'

    def __init__(self, session_path, overwrite=False, password1=None, password2=None, verbose=False):
        self.pwd1 = password1
        self.pwd2 = password2
        self.bi = None
        self.channel_info = {}

        # ------- properties ------
        self._mef_block_len = None
        self._record_offset = 0
        self.verbose = verbose
        # maximal nans in continuous block to be stored in data and not indexed
        self._max_nans_written = 'fs'
        # units of data stored
        self._data_units = b'uV'
        # from pymef library
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

        self.section2_ts_dict = {
                 'channel_description': b'ts_channel',
                 'session_description': b'ts_session',
                 'recording_duration': np.nan,
                 'reference_description': b'None',
                 'acquisition_channel_number': 1,
                 'sampling_frequency': np.nan,
                 'notch_filter_frequency_setting': 0,
                 'low_frequency_filter_setting': 1,
                 'high_frequency_filter_setting': 10,
                 'AC_line_frequency': 0,
                 'units_conversion_factor': 1.0,
                 'units_description': copy(self._data_units),
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
        """Closes the session on object deletion."""
        self.session.close()

    def _reload_session_info(self):
        """Reloads the session information from the MEF3 file."""
        self.session.reload()
        self.bi = self.session.read_ts_channel_basic_info()
        self.channel_info = {info['name']: deepcopy(info) for info in self.bi}
        for ch in self.channel_info.keys():
            self.channel_info[ch]['n_segments'] = len(self.session.session_md['time_series_channels'][ch]['segments'])
            self.channel_info[ch]['mef_block_len'] = int64(self.get_mefblock_len(self.channel_info[ch]['fsamp'][0]))

    def write_data(self, data_write, channel, start_uutc, sampling_freq, end_uutc=None, precision=None, new_segment=False,
                   discont_handler=True, reload_metadata=True):
        """
            General method for writing any data to the session. Method handles new channel data or appending to existing channel data
            automatically. Discont handler
            flag
            can be used for
            fragmentation to smaller intervals which
            are written in sequence with nans intervals skipped.

            Parameters
            ----------
            data_write : np.ndarray
                data to be written, data will be scaled a translated to int32 automatically if precision parameter is not given
            channel : str
                name of the stored channel
            start_uutc : int64
                uutc timestamp of the first sample
            sampling_freq : float
                only 0.1 Hz resolution is tested
            end_uutc : int, optional
                end of the data uutc timestamp, if less data is provided than end_uutc - start_uutc nans gap will be inserted to the data
            precision : int, optional
                Number of floating point to be scaled above zero. Data are multiplied by 10**precision before writing and scale factor is
                stored in metadata. used for transforming data to
                int32, can be positive or 0 = no change
                 in scale, only loss of decimals.
            new_segment : bool, optional
                if new mef3 segment should be created
            discont_handler: bool, optional
                disconnected segments will be stored in intervals if the gap in data is higher than max_nans_written property
            reload_metadata: bool, optional
                default: true. Parameter Controls reloading of metadata after writing new data - frequent call of write method on short
                signals can
                slow down
                writing. When false appending is not protected for correct endtime check, but data write is faster. Metadata are always
                reloaded with new segment.
            Returns
            -------
            out : bool
                True on success
        """

        # infer end_uutc from data
        if end_uutc is None:
            end_uutc = int64(start_uutc + (len(data_write)/sampling_freq * 1e6))

        # check times are correct
        if end_uutc < start_uutc:
            print(f"WARNING: incorrect End uutc time {end_uutc} is before beginning: {start_uutc}")
            return None

        # check if any data exists -> apend or create new segment
        if channel in self.channel_info.keys():
            # check if it is possible to write with configuration provided
            if start_uutc < self.channel_info[channel]['end_time'][0]:
                print(' Given start time is before end time of data already written to the session. Returning None')
                return None
            # NOTE fs can be different in the new segment but we dont work with different fs in the same channel
            if sampling_freq != self.channel_info[channel]['fsamp'][0]:
                print(' Sampling frequency of provided data does not match fs of already written data')
                return None
            # read precision from metadata - scale factor / can be different in new segment but not implemented
            precision = int(-1 * np.log10(self.channel_info[channel]['ufact'][0]))

            # convert data to int32
            data_converted = convert_data_to_int32(data_write, precision=precision)

            # check new segment flag
            segment = self.channel_info[channel]['n_segments']

        # new channel data with no previous data
        else:
            segment = 0
            new_segment = True
            if precision is None:
                print('WARNING: precision is not specified, infering...')
                precision = infer_conversion_factor(data_write)
                print(f'INFO: precision set to {precision}')

            ufact = np.round(0.1**precision, precision)
            # convert data to int32
            self.channel_info[channel] = {'mef_block_len': self.get_mefblock_len(sampling_freq), 'ufact': [ufact]}
            data_converted = convert_data_to_int32(data_write, precision=precision)

        # discont handler writes fragmented intervals ( skip nans greater than specified)
        if discont_handler:
            if self.max_nans_written == 'fs':
                max_nans = int(sampling_freq)
            else:
                max_nans = self.max_nans_written

            input_bin_vector = ~np.isnan(data_write)
            df_intervals = find_intervals_binary_vector(input_bin_vector, sampling_freq, start_uutc, samples_of_nans_allowed=max_nans)
        else:
            df_intervals = pd.DataFrame(data={'start_samples': 0, 'stop_samples': len(data_converted), 'start_uutc': start_uutc,
                                              'stop_uutc': end_uutc}, index=[0])

        if self.verbose:
            print(f'INFO: total number of intervals to be written: {len(df_intervals)}')
            print(f'Running...')
        if new_segment:
            for i, row in df_intervals.iterrows():
                data_part = data_converted[row['start_samples']:row['stop_samples']]
                if i == 0:
                    self._create_segment(data=data_part, channel=channel, start_uutc=row['start_uutc'], end_uutc=row['stop_uutc'],
                                         sampling_frequency=sampling_freq, segment=segment)
                else:
                    self._append_block(data=data_part, channel=channel, start_uutc=row['start_uutc'], end_uutc=row['stop_uutc'],
                                       segment=segment)
            reload_metadata = True
        # append to a last segment
        else:
            segment -= 1
            for i, row in df_intervals.iterrows():
                data_part = data_converted[row['start_samples']:row['stop_samples']]
                self._append_block(data=data_part, channel=channel, start_uutc=row['start_uutc'], end_uutc=row['stop_uutc'],
                                   segment=segment)

        if reload_metadata:
            self._reload_session_info()
        if self.verbose:
            print('INFO: data write method finished.')
        return True

    def write_annotations(self, annotations, channel=None):
        """
            Method writes annotations to a session/channel. Method handles new annotations or appending to existing annotations. Input
            data has to have required structure.

            Parameters
            ----------
            annotations : pandas.DataFrame
                DataFrame has to have a proper structure with columns - time column [uutctimestamp], type ['str specified in pymef' -
                Note or EDFA],
                text ['str'],
                optional duration [usec]
            channel : str, optional
                annotations are written at the channel level
        """

        # check int of time column
        if not np.issubdtype(annotations['time'].dtype, np.int64):
            annotations['time'] = annotations['time'].astype(np.int64)

        # check duration for int
        if 'duration' in annotations.columns:
            if not np.issubdtype(annotations['duration'].dtype, np.int64):
                annotations['duration'] = annotations['duration'].astype(np.int64)

        start_time = annotations['time'].min()
        end_time = annotations['time'].max()
        record_list = annotations.to_dict('records')

        # read old annotations
        print(' Reading previously stored annotations...')
        previous_list = self._read_annotation_record(channel=channel)
        if previous_list is not None:
            read_annotations = pd.DataFrame(previous_list)
            read_start = read_annotations['time'].min()
            read_end = read_annotations['time'].max()
            if read_start < start_time:
                start_time = read_start
            if read_end > end_time:
                end_time = read_end

            record_list.extend(previous_list)

        self._write_annotation_record(start_time, end_time, record_list, channel=channel)
        print(f'Annotations written, total {len(record_list)}, channel: {channel}')
        return

    def _write_annotation_record(self, start_time, end_time, record_list, channel=None):
        """
        Write annotation records into MEF file.

        Parameters
        ----------
        start_time : int
            Start time of the annotation record in microseconds.
        end_time : int
            End time of the annotation record in microseconds.
        record_list : list
            List of annotation records to be written.
        channel : str, optional
            Name of the channel for the annotation records.

        Returns
        -------
        None
        """
        record_offset = self.record_offset
        if channel is None:
            self.session.write_mef_records(self.pwd1, self.pwd2,  start_time,
                                 end_time, record_offset, record_list)
        else:
            self.session.write_mef_records(self.pwd1, self.pwd2, start_time,
                                           end_time, record_offset, record_list, channel=channel)
        self.session.reload()

    def _read_annotation_record(self, channel=None):
        """
        Read annotation records from MEF file.

        Parameters
        ----------
        channel : str, optional
            Name of the channel for the annotation records.

        Returns
        -------
        list
            List of annotation records.
        """
        try:
            annot_list = None
            if channel is None:
                annot_list = self.session.read_records()
            else:
                annot_list = self.session.read_records(channel=channel)
        except TypeError as exc:
            print('WARNING: read of annotations record failed, no annotations returned')
        except KeyError as exc:
            print('WARNING: read of annotations record failed, no annotations returned')
        return annot_list

    def _create_segment(self, data=None, channel=None, start_uutc=None, end_uutc=None, sampling_frequency=None, segment=0,):
        """
        Create a new segment in the MEF file.

        Parameters
        ----------
        data : np.ndarray
            Data to be written in the segment.
        channel : str
            Name of the channel for the segment.
        start_uutc : int
            Start timestamp of the segment in microseconds.
        end_uutc : int
            End timestamp of the segment in microseconds.
        sampling_frequency : float
            Sampling frequency of the data.
        segment : int, optional
            Segment index.

        Returns
        -------
        None
        """
        if data.dtype != np.int32:
            raise AssertionError('[TYPE ERROR] - MEF file writer accepts only int32 signal datatype!')

        if end_uutc < start_uutc:
            raise ValueError('End uutc timestamp lower than the start_uutc')

        self.section2_ts_dict['sampling_frequency'] = sampling_frequency

        # DEFAULT VALS FOR Segment 0
        if segment == 0:
            self.section3_dict['recording_time_offset'] = self.record_offset # int(start_uutc)
            self.section2_ts_dict['start_sample'] = 0
        else:
            self.section3_dict['recording_time_offset'] = self.record_offset # int(self.channel_info[channel]['start_time'][0])
            self.section2_ts_dict['start_sample'] = int64(self.channel_info[channel]['nsamp'][0])

        self.section2_ts_dict['recording_duration'] = int64((end_uutc - start_uutc) / 1e6)
        self.section2_ts_dict['units_conversion_factor'] = self.channel_info[channel]['ufact'][0]

        if self.verbose:
            print(f"INFO: creating new segment data for channel: {channel}, segment: {segment}, fs: {sampling_frequency}, ufac:"
                  f" {self.channel_info[channel]['ufact'][0]}, start: {start_uutc}, stop {end_uutc} ")
        self.session.write_mef_ts_segment_metadata(channel,
                                                   segment,
                                                   self.pwd1,
                                                   self.pwd2,
                                                   start_uutc,
                                                   end_uutc,
                                                   dict(self.section2_ts_dict),
                                                   dict(self.section3_dict))

        self.session.write_mef_ts_segment_data(channel,
                                               segment,
                                               self.pwd1,
                                               self.pwd2,
                                               self.channel_info[channel]['mef_block_len'],
                                               data)

    def _append_block(self, data=None, channel=None, start_uutc=None, end_uutc=None, segment=0):
        """
        Append a new block of data to a segment in the MEF file.

        Parameters
        ----------
        data : np.ndarray
            Data to be appended.
        channel : str
            Name of the channel for the block.
        start_uutc : int
            Start timestamp of the block in microseconds.
        end_uutc : int
            End timestamp of the block in microseconds.
        segment : int, optional
            Segment index.

        Returns
        -------
        None
        """
        if end_uutc < start_uutc:
            raise ValueError('End uutc timestamp lower than the start_uutc')
        if self.verbose:
            print(f"INFO: appending new data for channel: {channel}, segment: {segment}, ufac:"
                  f" {self.channel_info[channel]['ufact'][0]}, start: {start_uutc}, stop {end_uutc} ")

        self.session.append_mef_ts_segment_data(channel,
                                                  int64(segment),
                                                  self.pwd1,
                                                  self.pwd2,
                                                  start_uutc,
                                                  end_uutc,
                                                  self.channel_info[channel]['mef_block_len'],
                                                  data)

    def get_mefblock_len(self, fs):
        """
        Get the length of a MEF block.

        Parameters
        ----------
        fs : float
            Sampling frequency of the data.

        Returns
        -------
        int
            Length of the MEF block.
        """
        if self.mef_block_len is not None:
            return self.mef_block_len
        if fs >= 5000:
            return int(fs)
        else:
            if fs < 0:
                return int(fs * 100)
            else:
                return int(fs * 10)

    @property
    def max_nans_written(self):
        """
        Getter for the maximum number of NaN values allowed to be written. NaNs that are written as values will be written as the maximum value of the data type.
        Recommended value is 0, which will not allow any NaN values to be written. The signal will be split into data blocks based on the NaN values. This might cause poor data compression if a lot of NaN segments are present in the data.

        Returns
        -------
        int
            The maximum number of NaN values allowed to be written.
        """
        return self._max_nans_written

    @max_nans_written.setter
    def max_nans_written(self, max_samples):
        """
        Getter for the maximum number of NaN values allowed to be written. NaNs that are written as values will be written as the maximum value of the data type.
        Recommended value is 0, which will not allow any NaN values to be written. The signal will be split into data blocks based on the NaN values. This might cause poor data compression if a lot of NaN segments are present in the data.

        Returns
        -------
        None
        """
        if (max_samples < 0) | (not (isinstance(max_samples, int))):
            print("incorrect value, please provide positive int")
            return
        self._max_nans_written = max_samples

    @property
    def data_units(self):
        """
        Getter for the units of the data.

        Returns
        -------
        str
            The units of the data.
        """
        return self._data_units

    @data_units.setter
    def data_units(self, units_str):
        """
        Setter for the units of the data.

        Parameters
        ----------
        units_str : str
            The units for the data.

        Returns
        -------
        None
        """
        if (len(units_str) < 0) | (not (isinstance(units_str, str))):
            print("incorrect value, please provide str with less than 20 chars")
            return
        self._data_units = str.encode(units_str, 'utf-8')
        self.section2_ts_dict['units_description'] = copy(self._data_units)

    @property
    def record_offset(self):
        """
        Getter for the offset of the record.

        Returns
        -------
        int
            The offset of the record.
        """

        return self._record_offset

    @record_offset.setter
    def record_offset(self, new_offset):
        """
        Setter for the offset of the record.

        Parameters
        ----------
        new_offset : int
            The new offset for the record.

        Returns
        -------
        None
        """
        self._record_offset = new_offset

    @property
    def mef_block_len(self):
        """
        Getter for the MEF block length. Higher the mef_block length, better the compression, but higher the memory usage.

        Returns
        -------
        int
            The MEF block length.
        """
        return self._mef_block_len

    @mef_block_len.setter
    def mef_block_len(self, new_mefblock_len):
        """
        Getter for the MEF block length. Higher the mef_block length, better the compression, but higher the memory usage.

        Returns
        -------
        None
        """
        self._mef_block_len = new_mefblock_len

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
    """
    Creates a pink noise signal.

    Parameters
    ----------
    fs : int
        Sampling frequency of the signal.
    seg_len : int
        Length of the segment for which pink noise is to be generated.
    low_bound : float
        Lower bound for the amplitude of the generated noise.
    up_bound : float
        Upper bound for the amplitude of the generated noise.

    Returns
    -------
    numpy.ndarray
        The generated pink noise signal.

    Raises
    ------
    ValueError
        If the requested segment length results in too many samples.
    """
    n = int(fs * seg_len)
    if n > 20 * 1e6:
        raise ValueError('too many samples to generate')
    # if
    data = voss(n)
    norm_data = scale_signal(data, low_bound, up_bound)
    return norm_data


def scale_signal(data, a, b):
    """
    Scales a signal to a specified range.

    Parameters
    ----------
    data : numpy.ndarray
        The input signal to scale.
    a : float
        The lower bound of the desired range.
    b : float
        The upper bound of the desired range.

    Returns
    -------
    numpy.ndarray
        The input signal, rescaled to the range [a, b].

    Notes
    -----
    This function performs a linear transformation of the input data such that
    the minimum value becomes `a` and the maximum value becomes `b`.
    """
    min_x = np.min(data)
    data_range = np.max(data) - min_x
    temp_arr = (data - min_x) / data_range
    new_range = b - a
    return temp_arr * new_range + a


def check_int32_dynamic_range(x_min, x_max, alpha):
    """
        Checks whether the scaled range of the input values falls within the dynamic range of int32.

        Parameters
        ----------
        x_min : float or int
            The minimum value of the input.
        x_max : float or int
            The maximum value of the input.
        alpha : float or int
            The scaling factor applied to the input range.

        Returns
        -------
        bool
            Returns True if the scaled range falls within the dynamic range of int32. Otherwise, returns False.

        Notes
        -----
        This function checks whether the input range, when scaled by a factor of alpha,
        falls within the dynamic range of the int32 datatype. If the scaled range exceeds
        the dynamic range of int32, the function returns False. If the scaled range falls
        within the dynamic range of int32, the function returns True.
    """
    min_value = np.iinfo(np.int32).min
    if (x_min * alpha < min_value) & (x_max * alpha > np.iinfo(np.int32).max):
        return False
    else:
        return True


def infer_conversion_factor(data):
    """
    Infers the optimal conversion factor to scale the input data.

    Parameters
    ----------
    data : array-like
        The input data.

    Returns
    -------
    precision : int
        The optimal conversion factor for scaling the input data.

    Notes
    -----
    This function infers the optimal conversion factor for scaling the input data
    to bring it within the dynamic range of int32. It initially calculates the mean
    of the absolute differences of the data and scales it up until the mean reaches
    a threshold value. Then it checks if the range of the scaled data falls within
    the dynamic range of int32, and if not, it reduces the scaling factor until the
    scaled data is within the dynamic range of int32.

    If the input data has high dynamic range, this function might decrease the scaling
    factor to avoid saturation. In this case, a warning message will be printed indicating
    the decreased precision.
    """
    mean_digg_abs = np.nanmean(np.abs(np.diff(data)))
    precision = 0
    # this works for small z-scored data, for high dynamic range input needs to be decreased again (saturation)
    while (mean_digg_abs < 1000) & (mean_digg_abs != 0):
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
    """
        Converts the input data to int32 type, optionally scaling it by a specified factor.

        Parameters
        ----------
        data : array-like
            The input data.
        precision : int, optional
            The scaling factor (expressed as a power of 10) to apply to the data. If not provided,
            it will be inferred using the `infer_conversion_factor` function.

        Returns
        -------
        data_int32 : ndarray
            The input data converted to int32 type and scaled by the specified factor.

        Notes
        -----
        This function converts the input data to int32 type. If a scaling factor (precision) is
        provided, it is used to scale the data before conversion. If no scaling factor is
        provided, the function infers an optimal factor using the `infer_conversion_factor` function.

        The data is first rounded to the specified number of decimal places, then multiplied by
        10 to the power of the precision factor, and finally cast to int32 type.

        If the specified precision is less than 0 or not an integer, a warning is printed and
        the precision is set to 0, meaning no scaling is applied.
    """
    if precision is None:
        print(f"Info: convert data to int32:  precision is not given, inferring...")
        precision = infer_conversion_factor(data)
        print(f"Info: precision set to {precision}")

    if (precision < 0) | (not (isinstance(precision, int))):
        print(f"WARNING: precision set to incorrect value, it is set to default (0) = conversion without scaling (scaling=1)")
        precision = 0

    deciround = np.round(data, decimals=precision)
    data_int32 = np.empty(shape=deciround.shape, dtype=np.int32)
    data_int32[:] = 10 ** precision * (deciround)
    return data_int32


def find_intervals_binary_vector(input_bin_vector, fs, start_uutc, samples_of_nans_allowed=None):
    """
        Detects continuous intervals of ones in a binary vector and returns their start and stop times.

        Parameters
        ----------
        input_bin_vector : array-like
            The input binary vector.
        fs : int
            The sampling frequency of the data.
        start_uutc : int
            The start time of the data in microseconds since Unix Epoch.
        samples_of_nans_allowed : int, optional
            The maximum number of consecutive zeros (NaNs) that are considered part of an interval.
            If not provided, it defaults to the sampling frequency.

        Returns
        -------
        connected_detected_intervals : DataFrame
            A DataFrame containing the start and stop times (in samples and microseconds) of the
            continuous intervals of ones in the input binary vector.

        Notes
        -----
        This function processes a binary vector and detects continuous intervals of ones. It
        considers an interval to continue over a stretch of zeros (NaNs) if their number does not
        exceed a specified limit (samples_of_nans_allowed).

        The function returns a DataFrame containing the start and stop times of each detected
        interval, both in number of samples and in microseconds since Unix Epoch.

        The function first extends the input vector with a zero at both ends, then calculates the
        difference between consecutive elements. The positions where this difference equals 1
        correspond to the starts of intervals of ones, while the positions where it equals -1
        correspond to their ends. The function then merges intervals that are closer to each other
        than samples_of_nans_allowed and calculates the corresponding start and stop times.
        """
    if samples_of_nans_allowed is None:
        samples_of_nans_allowed = int(fs)

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
    connected_detected_intervals['start_samples'] = overlap_starts.astype(int64)
    connected_detected_intervals['stop_samples'] = overlap_ends.astype(int64)

    # connected_detected_intervals = connected_detected_intervals.append(lonely_segments, ignore_index=True)
    connected_detected_intervals = pd.concat([connected_detected_intervals, lonely_segments], ignore_index=True)

    connected_detected_intervals = connected_detected_intervals.sort_values(by='start_samples').reset_index(drop=True)

    # calculate uutc time of intervals
    connected_detected_intervals['start_uutc'] = (connected_detected_intervals['start_samples'] / fs * 1e6 + start_uutc).astype(int64)
    connected_detected_intervals['stop_uutc'] = (connected_detected_intervals['stop_samples'] / fs * 1e6 + start_uutc).astype(int64)
    return connected_detected_intervals


def check_data_integrity(original_data, converted_data, precision):
    """
        Check the integrity of the original data against the converted data.

        Parameters
        ----------
        original_data : array-like
            The original data before conversion.
        converted_data : array-like
            The data after conversion.
        precision : int
            The precision used during the conversion process.

        Returns
        -------
        result_bin : bool
            True if all close, else False.

        Notes
        -----
        This function checks the integrity of the original data against the converted data.
        It converts the converted data back to the original scale, excludes NaNs, and checks
        if the original and reconverted data are close to each other within a specified tolerance.
        The check is performed using numpy's allclose function with a tolerance of 0.1^(precision-1).
    """
    coverted_float = 0.1**precision*(converted_data)
    idx_numbers = ~np.isnan(original_data)
    result_bin = np.allclose(coverted_float[idx_numbers], original_data[idx_numbers], atol=0.1**(precision-1))
    return result_bin






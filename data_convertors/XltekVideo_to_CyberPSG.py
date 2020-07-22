# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Converting Videos from XLTEK patient record into CyberPSG format.
    - Check all libraries or use Lambda2-Anaconda Environment "filip_dogs"
    - Call from bash :
        XltekVideo_to_CyberPSG.py -path_from /path/to/folder/with/original/data -path_to /where/you/want/new/folder

    Reads *.vtc header file and copies all videos into a new folder with filename in CyberPSG compatible format.
"""


import os
import datetime
import shutil
from tqdm import tqdm
import argparse
import numpy as np
from datetime import datetime, timedelta, tzinfo
from calendar import timegm

# # How To Convert a UNIX time_t to a Win32 FILETIME or SYSTEMTIME - http://support.microsoft.com/kb/167296
EPOCH_AS_FILETIME = 116444736000000000  # January 1, 1970 as MS file time
HUNDREDS_OF_NANOSECONDS = 10000000

ZERO = timedelta(0)
HOUR = timedelta(hours=1)

class UTC(tzinfo):
    """UTC"""

    def utcoffset(self, dt):
        return ZERO

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return ZERO
utc = UTC()


def dt_to_filetime(dt):
    if (dt.tzinfo is None) or (dt.tzinfo.utcoffset(dt) is None):
        dt = dt.replace(tzinfo=utc)
    ft = EPOCH_AS_FILETIME + (timegm(dt.timetuple()) * HUNDREDS_OF_NANOSECONDS)
    return ft + (dt.microsecond * 10)

def filetime_to_dt(ft):
    # Get seconds and remainder in terms of Unix epoch
    (s, ns100) = divmod(ft - EPOCH_AS_FILETIME, HUNDREDS_OF_NANOSECONDS)
    # Convert to datetime object
    dt = datetime.fromtimestamp(s)
    # Add remainder in as microseconds. Python 3.2 requires an integer
    dt = dt.replace(microsecond=(ns100 // 10))
    return dt

def get_folders(path):
    [os.path.join(root, dirs)
     for root, dirs, files in os.walk(path)]

def get_files(path, endings_tuple=None):
    if endings_tuple:
        return [os.path.join(root, name)
                for root, dirs, files in os.walk(path)
                for name in files
                if name.endswith(endings_tuple) and '._' not in name ]
    else:
        return [os.path.join(root, name)
                for root, dirs, files in os.walk(path)
                for name in files if '._' not in name]

class VTC_Reader:
    FILETIME_null_date = datetime(1601, 1, 1, 0, 0, 0)
    bl_vtc = 16
    bl_schema = 4

    bl_FileName = 261
    bl_mess = 16
    bl_StartTime = 8
    bl_EndTime = 8

    def __init__(self, path):
        self.FileName = []
        #self.mess = []
        self.StartTime = []
        self.EndTime = []
        self.N = 0

        with open(path, 'rb') as fstream:
            self.fstream = fstream
            self._read_vtc_header()

            while self._read_vtc_record() is True:
                self.N += 1

    def _read_vtc_header(self):
        self.vtc = self.fstream.read(self.bl_vtc)
        self.schema = self.fstream.read(self.bl_schema)

    def _read_vtc_record(self):
        fname = self.fstream.read(self.bl_FileName)
        mess = self.fstream.read(self.bl_mess)
        StartTime = int.from_bytes(self.fstream.read(self.bl_StartTime), byteorder='little', signed=True)
        EndTime = int.from_bytes(self.fstream.read(self.bl_EndTime), byteorder='little', signed=True)

        if fname.__len__() > 0:
            self.FileName.append(fname)
            self.StartTime.append(StartTime)
            self.EndTime.append(EndTime)
            return True
        return False

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        string = self.FileName[item]
        string = str(np.array(string).astype(np.str))

        ftime = self.StartTime[item]

        a = filetime_to_dt(ftime)

        #ftime = ftime / 1e7
        #ftime = ftime + self.FILETIME_null_date.timestamp()
        #strftime = datetime.utcfromtimestamp(a).strftime("%Y%m%d%H%M%S") + '.' + string.split('.')[-1]
        strftime = a.strftime("%Y%m%d%H%M%S") + '.' + string.split('.')[-1]
        return string, strftime

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Converting video files from XLTEK format to CyberPSG compatible files.')
    parser.add_argument(
        '--path_from', dest='PATH_FROM', required=True,
        help='Path to the dir containing video files which are used to unique frame extraction')
    parser.add_argument(
        '--path_to', dest='PATH_TO', required=True,
        help='Dir path where unique frames are extracted to')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Check windows or linux and set separator
    if os.name == 'nt': DELIMITER = '\\' # Windows
    else: DELIMITER = '/' # posix

    args = parse_args()
    PATH_FROM = args.PATH_FROM
    PATH_TO = args.PATH_TO

    if not os.path.exists(PATH_TO):
        raise AssertionError('Path does not exist ' + PATH_TO)

    if not os.path.exists(PATH_FROM):
        raise AssertionError('Path does not exist ' + PATH_FROM)

    vtc_file = get_files(PATH_FROM, ('vtc'))

    if vtc_file.__len__() != 1:
        raise AssertionError('Found ' + str(vtc_file.__len__()) + ' files in the folder. Exactly 1 VTC file has to be present')

    VTCRdr = VTC_Reader(vtc_file[0])

    outp_dir = os.path.join(PATH_TO, PATH_FROM.split(DELIMITER)[-1], 'Videos')
    #if os.path.exists(outp_dir):
        #shutil.rmtree(outp_dir)
        #time.sleep(0.1)
    if not os.path.exists(outp_dir):
        os.makedirs(outp_dir)

    print('Copying files')
    for file_idx in tqdm(range(VTCRdr.__len__())):
        old_name, new_name = VTCRdr[file_idx]

        old_file_path = os.path.join(PATH_FROM, old_name)
        new_file_path = os.path.join(outp_dir, new_name)

        if os.path.isfile(old_file_path):
            shutil.copyfile(old_file_path, new_file_path)



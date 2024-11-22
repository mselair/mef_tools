import os
import struct
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import numpy as np
import copy

# Encryption Levels
LEVEL_0_ENCRYPTION = 0
LEVEL_1_ENCRYPTION = 1
LEVEL_2_ENCRYPTION = 2

# Metadata Section Offsets and Sizes
METADATA_SECTION_1_SIZE = 1536
METADATA_SECTION_2_SIZE = 10752
METADATA_SECTION_3_SIZE = 3072

# AES Constants
AES_BLOCK_SIZE = 16
PASSWORD_VALIDATION_FIELD_SIZE = 16

# Path Constants
METADATA_FILE_EXTENSION = 'tmet'
DATA_FILE_EXTENSION = 'tdat'

# Encryption Fields in Metadata Section 1
SECTION_2_ENCRYPTION_OFFSET = 1024
SECTION_3_ENCRYPTION_OFFSET = 1025

# Password Validation Field Offsets in Universal Header
LEVEL_1_PASSWORD_VALIDATION_FIELD_OFFSET = 868
LEVEL_2_PASSWORD_VALIDATION_FIELD_OFFSET = 884

class MEFSegment:
    def __init__(self, segment_path, password=None):
        self.segment_path = segment_path
        self.password = password
        self.metadata_section_1 = None
        self.metadata_section_2 = None
        self.metadata_section_3 = None
        self.data = None
        self.section3_dict = {}
        self.section2_ts_dict = {}

    def read_metadata(self):
        tmet_dir = os.path.basename(self.segment_path).split('.segd')[0] + '.tmet'
        tmet_path = os.path.join(self.segment_path, tmet_dir)
        with open(tmet_path, 'rb') as f:
            section_1 = f.read(METADATA_SECTION_1_SIZE)  # Metadata Section 1
            self.metadata_section_1 = section_1

            # Check encryption levels
            section_2_encryption = struct.unpack('b', section_1[SECTION_2_ENCRYPTION_OFFSET:SECTION_2_ENCRYPTION_OFFSET + 1])[0]
            section_3_encryption = struct.unpack('b', section_1[SECTION_3_ENCRYPTION_OFFSET:SECTION_3_ENCRYPTION_OFFSET + 1])[0]

            section_2 = f.read(METADATA_SECTION_2_SIZE)  # Metadata Section 2
            if self.password and section_2_encryption in [LEVEL_1_ENCRYPTION, LEVEL_2_ENCRYPTION]:
                section_2 = self._decrypt_data(section_2)
            self.metadata_section_2 = section_2

            section_3 = f.read(METADATA_SECTION_3_SIZE)  # Metadata Section 3
            if self.password and section_3_encryption == LEVEL_2_ENCRYPTION:
                section_3 = self._decrypt_data(section_3)
            self.metadata_section_3 = section_3

            self._parse_metadata()

    def read_data(self):
        tdat_path = os.path.join(self.segment_path, 'tdat')
        with open(tdat_path, 'rb') as f:
            self.data = f.read()
            # Decrypt data if needed
            if self.password:
                self.data = self._decrypt_data(self.data)

    def _decrypt_data(self, data):
        # Implement AES decryption using the password
        key = hashlib.sha256(self.password.encode()).digest()
        cipher = AES.new(key, AES.MODE_CBC)
        decrypted_data = unpad(cipher.decrypt(data), AES.block_size)
        return decrypted_data

    def _safe_decode(self, byte_data):
        try:
            return byte_data.decode('utf-8').strip('\x00')
        except UnicodeDecodeError:
            return byte_data.decode('utf-8', errors='replace').strip('\x00')

    def _parse_metadata(self):
        # Parse Section 3
        self.section3_dict = {
            'recording_time_offset': struct.unpack('q', self.metadata_section_3[0:8])[0],
            'DST_start_time': struct.unpack('q', self.metadata_section_3[8:16])[0],
            'DST_end_time': struct.unpack('q', self.metadata_section_3[16:24])[0],
            'GMT_offset': struct.unpack('i', self.metadata_section_3[24:28])[0],
            'subject_name_1': self._safe_decode(self.metadata_section_3[28:59]),
            'subject_name_2': self._safe_decode(self.metadata_section_3[59:90]),
            'subject_ID': self._safe_decode(self.metadata_section_3[90:121]),
            'recording_location': self._safe_decode(self.metadata_section_3[121:252]),
            'protected_region': self.metadata_section_3[252:1376],
            'discretionary_region': self.metadata_section_3[1376:2400],
        }

        # Parse Section 2 (Time Series Metadata)
        self.section2_ts_dict = {
            'channel_description': self._safe_decode(self.metadata_section_2[0:2048]),
            'session_description': self._safe_decode(self.metadata_section_2[2048:4096]),
            'recording_duration': struct.unpack('q', self.metadata_section_2[4096:4104])[0],
            'reference_description': self._safe_decode(self.metadata_section_2[4104:6144]),
            'acquisition_channel_number': struct.unpack('q', self.metadata_section_2[6144:6152])[0],
            'sampling_frequency': struct.unpack('d', self.metadata_section_2[6152:6160])[0],
            'low_frequency_filter_setting': struct.unpack('d', self.metadata_section_2[6160:6168])[0],
            'high_frequency_filter_setting': struct.unpack('d', self.metadata_section_2[6168:6176])[0],
            'notch_filter_frequency_setting': struct.unpack('d', self.metadata_section_2[6176:6184])[0],
            'AC_line_frequency': struct.unpack('d', self.metadata_section_2[6184:6192])[0],
            'units_conversion_factor': struct.unpack('d', self.metadata_section_2[6192:6200])[0],
            'units_description': self._safe_decode(self.metadata_section_2[6200:6336]),
            'maximum_native_sample_value': struct.unpack('d', self.metadata_section_2[6336:6344])[0],
            'minimum_native_sample_value': struct.unpack('d', self.metadata_section_2[6344:6352])[0],
            'start_sample': struct.unpack('q', self.metadata_section_2[6352:6360])[0],
            'number_of_samples': struct.unpack('q', self.metadata_section_2[6360:6368])[0],
            'number_of_blocks': struct.unpack('q', self.metadata_section_2[6368:6376])[0],
            'maximum_block_bytes': struct.unpack('q', self.metadata_section_2[6376:6384])[0],
            'maximum_block_samples': struct.unpack('I', self.metadata_section_2[6384:6388])[0],
            'maximum_difference_bytes': struct.unpack('I', self.metadata_section_2[6388:6392])[0],
            'block_interval': struct.unpack('q', self.metadata_section_2[6392:6400])[0],
            'number_of_discontinuities': struct.unpack('I', self.metadata_section_2[6400:6404])[0],
            'maximum_contiguous_blocks': struct.unpack('q', self.metadata_section_2[6404:6412])[0],
            'maximum_contiguous_block_bytes': struct.unpack('q', self.metadata_section_2[6412:6420])[0],
            'maximum_contiguous_samples': struct.unpack('q', self.metadata_section_2[6420:6428])[0],
            'protected_region': self.metadata_section_2[6428:10692],
            'discretionary_region': self.metadata_section_2[10692:],
        }

# Example usage
path = '/Users/mivalt.filip/Data/BIDS_BCI2000_CortecInterchange_v2.0.0/BIDS_BCI2000_CortecInterchange_v2.0.0/sub-c02/ses-intraop01/ieeg/sub-c02_ses-intraop01_acq-ref1_ieeg.mefd/Ch0.timd/Ch0-000000.segd'

segment = MEFSegment(path)
segment.read_metadata()

print(segment.section3_dict)
print(segment.section2_ts_dict)

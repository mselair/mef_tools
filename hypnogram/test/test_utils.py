import unittest
from unittest import TestCase
from datetime import datetime, timedelta
from pandas._libs.tslibs.timestamps import Timestamp
from dateutil import tz
import pandas as pd
from hypnogram.utils import (
    _convert_to_timestamp, _convert_to_utc, _convert_to_datetime_utc, _convert_to_local,
    _convert_to_pandas_timestamp_utc, _convert_to_timezone,
    time_to_local, time_to_timestamp, time_to_timezone, time_to_utc,
    create_duration, create_day_indexes, merge_annotations, tile_annotations
     )



class TestMefWriter(TestCase):
    def test__convert_to_timestamp(self):
        inputs = [datetime.now(tz.tzlocal()), Timestamp.now(tz.tzlocal()), datetime.now().timestamp()]
        for inp in inputs:
            x = _convert_to_timestamp(inp)
            self.assertIsInstance(x, (float, int))

    def test__convert_to_datetime_utc(self):
        inputs = [datetime.now(tz.tzlocal()), Timestamp.now(tz.tzlocal()), datetime.now().timestamp()]
        for inp in inputs:
            x = _convert_to_datetime_utc(inp)
            self.assertIsInstance(x, datetime)

    def test__convert_to_pandas_timestamp_utc(self):
        inputs = [datetime.now(tz.tzlocal()), Timestamp.now(tz.tzlocal()), datetime.now().timestamp()]
        for inp in inputs:
            x = _convert_to_pandas_timestamp_utc(inp)
            self.assertIsInstance(x, Timestamp)

    def test__convert_to_utc(self):
        now = datetime.now(tz.tzlocal())
        nowutc = now.astimezone(tz.tzutc())
        self.assertEqual(_convert_to_utc(now), nowutc)

    def test__convert_to_local(self):
        nowutc = datetime.now(tz.tzutc())
        nowlocal = nowutc.astimezone(tz.tzlocal())
        self.assertEqual(_convert_to_local(nowutc), nowlocal)

    def test__convert_to_timezone(self):
        now = datetime.now(tz.gettz('America/Chicago'))
        nowutc = now.astimezone(tz.tzutc())
        self.assertEqual(_convert_to_timezone(now, tz.tzutc()), nowutc)

    def test_time_to_utc(self):
        now = datetime.now().timestamp()-1000
        df = pd.DataFrame(
            [
                {'start': _convert_to_local(now), 'end':_convert_to_local(now+30), 'duration':30, 'annotation':' WAKE'},
                {'start': _convert_to_local(now+30), 'end':_convert_to_local(now+60), 'duration':30, 'annotation': 'WAKE'},
            ]
        )
        df = time_to_utc(df)
        for row in df.iterrows():
            self.assertEqual(row[1]['start'].tzinfo, tz.tzutc())
            self.assertEqual(row[1]['end'].tzinfo, tz.tzutc())

    def test_time_to_local(self):
        now = datetime.now().timestamp()-1000
        df = pd.DataFrame(
            [
                {'start': _convert_to_utc(now), 'end':_convert_to_utc(now+30), 'duration':30, 'annotation':' WAKE'},
                {'start': _convert_to_utc(now+30), 'end':_convert_to_utc(now+60), 'duration':30, 'annotation': 'WAKE'},
            ]
        )
        df = time_to_local(df)
        for row in df.iterrows():
            self.assertEqual(row[1]['start'].tzinfo, tz.tzlocal())
            self.assertEqual(row[1]['end'].tzinfo, tz.tzlocal())

    def test_time_to_timezone(self):
        now = datetime.now().timestamp()-1000
        df = pd.DataFrame(
            [
                {'start': _convert_to_utc(now), 'end':_convert_to_utc(now+30), 'duration':30, 'annotation':' WAKE'},
                {'start': _convert_to_utc(now+30), 'end':_convert_to_utc(now+60), 'duration':30, 'annotation': 'WAKE'},
            ]
        )
        df = time_to_timezone(df, tz.gettz('America/Chicago'))
        for row in df.iterrows():
            self.assertEqual(row[1]['start'].tzinfo, tz.gettz('America/Chicago'))
            self.assertEqual(row[1]['end'].tzinfo, tz.gettz('America/Chicago'))

    def test_time_to_timestamp(self):
        now = datetime.now().timestamp()-1000
        df = pd.DataFrame(
            [
                {'start': _convert_to_utc(now), 'end':_convert_to_utc(now+30), 'duration':30, 'annotation':' WAKE'},
                {'start': _convert_to_utc(now+30), 'end':_convert_to_utc(now+60), 'duration':30, 'annotation': 'WAKE'},
            ]
        )
        df = time_to_timestamp(df)
        for row in df.iterrows():
            self.assertIsInstance(row[1]['start'], (float, int))
            self.assertIsInstance(row[1]['end'], (float, int))

    def test_create_duration(self):
        now = datetime.now().timestamp()-1000
        df = pd.DataFrame(
            [
                {'start': _convert_to_utc(now), 'end':_convert_to_utc(now+30), 'annotation':' WAKE'},
                {'start': _convert_to_utc(now+30), 'end':_convert_to_utc(now+60), 'annotation': 'WAKE'},
            ]
        )
        df = create_duration(df)
        self.assertEqual(df['duration'][0], 30)

    def test_create_merge_annotations(self):
        now = datetime.now().timestamp()-1000
        df = pd.DataFrame(
            [
                {'start': _convert_to_utc(now), 'end':_convert_to_utc(now+30), 'duration':30, 'annotation':'WAKE'},
                {'start': _convert_to_utc(now+30), 'end':_convert_to_utc(now+60), 'duration':30, 'annotation': 'WAKE'},
            ]
        )
        df = merge_annotations(df)
        self.assertEqual(df.__len__(), 1)

    def test_create_tile_annotations(self):
        now = datetime.now().timestamp()-1000
        df = pd.DataFrame(
            [
                {'start': _convert_to_utc(now), 'end':_convert_to_utc(now+30), 'duration':30, 'annotation':'WAKE'},
                {'start': _convert_to_utc(now+30), 'end':_convert_to_utc(now+60), 'duration':30, 'annotation': 'WAKE'},
            ]
        )
        #df = merge_annotations(df)
        df = tile_annotations(df, 10)
        self.assertEqual(df.__len__(), 6)


    def test_create_day_indexes(self):
        t = datetime(year=2010, month=1, day=1, hour=12, minute=30, second=00, tzinfo=tz.tzutc())
        df = pd.DataFrame(
            [
                {'start':t+timedelta(minutes=k), 'end':t+timedelta(minutes=k+10)} for k in range(0, 120, 10)
            ]
        )
        df = create_day_indexes(df, 14)
        self.assertEqual(df['day'].sum(), 3)




        t = datetime(year=2010, month=1, day=1, hour=23, minute=30, second=00, tzinfo=tz.tzutc())
        df = pd.DataFrame(
            [
                {'start':t+timedelta(minutes=k), 'end':t+timedelta(minutes=k+10)} for k in range(0, 120, 10)
            ]
        )
        df = create_day_indexes(df, 0)
        self.assertEqual(df['day'].sum(), 9)


if __name__ == '__main__':
    unittest.main()
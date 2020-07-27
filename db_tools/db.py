# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import pandas as pd
import sqlalchemy as sqla
from sqlalchemy.pool import NullPool
from sshtunnel import SSHTunnelForwarder


class DatabaseHandler:
    __version__ = '0.0.1'
    def __init__(self, sql_host=None, sql_db_name=None, sql_user=None, sql_pwd=None, sql_port='3360', ssh_host=None, ssh_user=None, ssh_pwd=None, ssh_port='22'):
        self._ssh_host = ssh_host
        self._ssh_user = ssh_user
        self._ssh_pwd = ssh_pwd
        self._ssh_port = ssh_port

        self._sql_host = sql_host
        self._sql_port = sql_port
        self._sql_user = sql_user
        self._sql_pwd = sql_pwd
        self._sql_db_name = sql_db_name

        self._sql_connection = None
        self._ssh_tunnel = None
        self._engine = None


        self.open()
        self._init_sql_engine()

    def _init_sql_engine(self):
        if self.check_ssh_connection():
            self._engine = sqla.create_engine(
                'mysql+pymysql://{}:{}@{}:{}/{}'.format(self._sql_user, self._sql_pwd, 'localhost', self._ssh_tunnel.local_bind_port, self._sql_db_name), poolclass=NullPool)
        else:
            self._engine = sqla.create_engine(
                'mysql+pymysql://{}:{}@{}:{}/{}'.format(self._sql_user, self._sql_pwd, self._sql_host, self._sql_port, self._sql_db_name), poolclass=NullPool)


    def _open_sql(self):
        self._sql_connection = self._engine.connect()

    def _close_sql(self):
        self._sql_connection.close()

    def check_sql_connection(self):
        self._open_sql()
        self._close_sql()
        return True

    def _open_ssh(self):
        self._ssh_tunnel = SSHTunnelForwarder(
            (self._ssh_host, int(self._ssh_port)),
            ssh_username=self._ssh_user,
            ssh_password=self._ssh_pwd,
            remote_bind_address=(self._sql_host, int(self._sql_port)))
        self._ssh_tunnel.start()

    def check_ssh_connection(self):
        if self._ssh_tunnel:
            return self._ssh_tunnel.is_active
        return False

    def _close_ssh(self):
        if self.check_ssh_connection():
            self._ssh_tunnel.close()

    def open(self):
        if self._ssh_host:
            self._open_ssh()
            self.check_ssh_connection()

        self._init_sql_engine()
        self.check_sql_connection()

    def close(self):
        self._close_sql()
        self._close_ssh()

    def check_connection(self):
        self.check_ssh_connection()
        return self.check_sql_connection()

    def __del__(self):
        self.close()

class SessionFinder(DatabaseHandler):
    __version__ = '0.0.1'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_mef_session(self, id, uutc_start, uutc_stop):
        self._sql_connection = self._engine.connect()
        query = f"SELECT uutc_start, uutc_stop, session, fsamp, channels FROM {self._sql_db_name}.Sessions where id='{id}' and uutc_start<='{uutc_start}' and uutc_stop>='{uutc_stop}' order by uutc_start desc"
        df_data = pd.read_sql(query, self._sql_connection)
        self._sql_connection.close()
        return df_data

    @property
    def patient_ids(self):
        self._open_sql()
        query = f"SELECT DISTINCT id FROM {self._sql_db_name}.Sessions"
        unique_ids = pd.read_sql(query, self._sql_connection)
        self._close_sql()
        return unique_ids['id'].to_list()



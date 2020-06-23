# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import zmq
import sqlalchemy as sqla
from tqdm import tqdm
from sqlalchemy.pool import NullPool
from sshtunnel import SSHTunnelForwarder
import pandas as pd
import numpy as np

class MefClient:
    """
    MEF Database client - asks for data
    """

    RESPONSE_WAIT = 30
    def __init__(self, ports, server_ip):
        context = zmq.Context(1)
        self.client = context.socket(zmq.REQ)
        self.poll = zmq.Poller()
        self.ports = ports
        self.server_ip = server_ip

    def request_data(self, path, channel, passwd='', start=None, stop=None, sample=0):
        # sample arg must be 1 if data are read by sample
        client = self.client
        for p in self.ports:
            client.connect(f"tcp://{self.server_ip}:{p}")

        self.poll.register(client, zmq.POLLIN)
        try:
            client.send_json({'path': path, 'channel': channel, 'passwd': passwd, 'start': start, 'stop': stop, 'sample': sample})
            socks = dict(self.poll.poll(self.RESPONSE_WAIT * 1000))
            if socks.get(client) == zmq.POLLIN:
                md = client.recv_json(flags=0, )
                msg = client.recv(flags=0, copy=True, track=False)
                buf = memoryview(msg)
                res_numpy = np.frombuffer(buf, dtype=md['dtype'])
                return res_numpy.reshape(md['shape']), md['fsamp']
            else:
                client.setsockopt(zmq.LINGER, 0)
                client.close()
                self.poll.unregister(client)
                return False, f'Server response time elapsed {self.RESPONSE_WAIT} s'

        except Exception as exc:
            exce = f'{exc}'
            return False, exce


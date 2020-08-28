.. image:: https://travis-ci.org/xmival00/MEF_Tools.svg?branch=master
    :target: https://pypi.org/project/mselair-aisc/

.. image:: https://readthedocs.org/projects/mef-tools/badge/?version=latest
    :target: https://mef-tools.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/pypi-1.0.6-orange
    :target: https://pypi.org/project/mselair-aisc/

.. image:: https://img.shields.io/pypi/pyversions/mselair-aisc.svg?longCache=True
    :target: https://pypi.org/project/mselair-aisc/


MEF_Tools
----------------

This package provides tools for easier `Multiscale Electrophysiology Format (MEF) <https://doi.org/10.1016%2Fj.jneumeth.2009.03.022>`_ data file saving and reading. See an example below. 

`Multiscale Electrophysiology Format (MEF) <https://doi.org/10.1016%2Fj.jneumeth.2009.03.022>`_ is a data file format designed for storing electro-physiological signals. MEF was developed to handle the large amounts of data produced by large-scale electro-physiology in human and animal subjects. See original `GitHub repository <https://github.com/msel-source/meflib>`_.

`pymef <https://github.com/msel-source/meflib>`_ is a Python wrapper for meflib. See `documentation <https://pymef.readthedocs.io/en/latest/>`_.


--------------------------------------------------------------------------------------------------

Source
----------------

* Brinkmann BH, Bower MR, Stengel KA, Worrell GA, Stead M. Large-scale electrophysiology: acquisition, compression, encryption, and storage of big data. J Neurosci Methods. 2009;180(1):185‐192. doi:10.1016/j.jneumeth.2009.03.022

* `Repository <https://github.com/msel-source/meflib>`_

--------------------------------------------------------------------------------------------------


Example
----------------


.. code-block:: python

    # Requirements
    # Python 3.6
    # pymef - pip install pymef
    # numpy - if anaconda conda install -c anaconda numpy; else pip install numpy
    # pandas - same as numpy
    
    # imports
    import numpy as np
    from mef_tools.io import MefWriter, MefReader, create_pink_noise
    import os

    # path to data
    session_name = 'session'
    session_path = os.getcwd() + f'/{session_name}.mefd'
    mef_session_path = session_path

    # define how much is written
    secs_to_write = 30

    # define start of data uutc in uUTC time
    start_time = 1578715810000000
    # define end of data in uUTC time (optional) if None it is inferred from start_time and number samples + fs
    end_time = int(start_time + 1e6*secs_to_write)

    # passwords
    pass1 = 'pass1'
    pass2 = 'pass2'

    # overwrite flag - delete all session with the same path
    writer = MefWriter(session_path, overwrite=True, password1=pass1, password2=pass2)

    # property max nans in continuous block set (default is equal to fs)
    writer.max_nans_written = 100
    # property units of data - default uV
    writer.data_units = 'mV'

    # create test data with fs 500 Hz and dynamic range <-10; 10>
    fs = 500
    low_b = -10
    up_b = 10

    data_to_write = create_pink_noise(fs, secs_to_write, low_b, up_b)

    # channel name
    channel = 'channel_1'

    # precision - how many floating points will be scaled up to int. e.g. sample with float = 0.001 with precision 3 -> will be stored as int
    # = 1 and scaling factor will be 0.001. This will be automatically set if not specified and no data exist with the same channel name)
    precision = 3
    writer.write_data(data_to_write, channel, start_time, fs, precision=precision)

    # append new data - precision is now fixed by the first data written
    secs_to_append = 5
    discont_length = 3
    append_time = end_time + int(discont_length*1e6)
    append_end = int(append_time + 1e6*secs_to_append)
    data = create_pink_noise(fs, secs_to_append, low_b, up_b)
    del writer

    # create a new writer and append to previous data
    writer2 = MefWriter(session_path, overwrite=False, password1=pass1, password2=pass2)
    # new data are appended to the previous data
    writer2.write_data(data, channel, append_time, fs)

    # new segment - same call just flag is changed
    secs_to_write_seg2 = 10
    gap_time = 3.36*1e6
    newseg_time = append_end + int(gap_time)
    newseg_end = int(newseg_time + 1e6*secs_to_write_seg2)
    data = create_pink_noise(fs, secs_to_write_seg2, low_b, up_b)
    data[30:540] = np.nan
    data[660:780] = np.nan
    writer2.write_data(data, channel, newseg_time, fs, new_segment=True, )

    # inferred precision
    channel = 'channel_2'
    writer2.write_data(data, channel, newseg_time, fs, new_segment=True, )

    # ----------- write annotations ---------
    # define start of data uutc in uUTC time
    start_time = 1578715810000000 - 1000000
    # define end of data in uUTC time
    end_time = int(start_time + 1e6 * 300)
    # offset time - if not data written
    offset = int(start_time - 1e6)
    # create note annotation ( no duration)
    starts = np.arange(start_time, end_time, 2e6)
    text = ['test'] * len(starts)
    types = ['Note'] * len(starts)
    note_annotations = pd.DataFrame(data={'time': starts, 'text': text, 'type': types})
    # write annotations to session level
    writer2.write_annotations(note_annotations,)

    # create annotation with duration and store them to a channel
    starts = np.arange(start_time, end_time, 1e5)
    text = ['test'] * len(starts)
    types = ['EDFA'] * len(starts)
    duration = [10025462] * len(starts)
    note_annotations = pd.DataFrame(data={'time': starts, 'text': text, 'type': types, 'duration':duration})
    # write annotations to the channel level
    writer2.write_annotations(note_annotations, channel=channel )

    # -------- reader example -----------

    Reader = MefReader(path_file_to, password=pass2)
    signals = []
    
    for channel in Reader.channels:
        x = Reader.get_data(key)
        x = Reader.get_data(key, Reader.get_property('start_time', key), Reader.get_property('end_time', key))
        print('Overall Difference in signal ', key, ' ', (df[key][:-1] - x).sum())
        signals.append(x)
    

-------------------------------------------------------------------------------------------------------------

Installation
----------------

See installation instructions `INSTALL.md <https://github.com/xmival00/MEF_Tools/blob/master/INSTALL.md>`_.

------------------------------------------------------------------------------------------------------------

License
----------------

This software is licensed under the Apache-2.0 License. See `LICENSE <https://github.com/xmival00/MEF_Tools/blob/master/LICENSE>`_ file in the root directory of this project.


`MEF <https://doi.org/10.1016%2Fj.jneumeth.2009.03.022>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package provides two objects for easier `Multiscale Electrophysiology Format (MEF) <https://doi.org/10.1016%2Fj.jneumeth.2009.03.022>`_ data file saving and reading. See an example below.

`Multiscale Electrophysiology Format (MEF) <https://doi.org/10.1016%2Fj.jneumeth.2009.03.022>`_ is a data file format designed for storing electro-physiological signals. MEF was developed to handle the large amounts of data produced by large-scale electro-physiology in human and animal subjects. See original `GitHub repository <https://github.com/msel-source/meflib>`_.

`pymef <https://github.com/msel-source/meflib>`_ is a Python wrapper for meflib. See `documentation <https://pymef.readthedocs.io/en/latest/>`_.


Source
^^^^^^^^^^^

* Brinkmann BH, Bower MR, Stengel KA, Worrell GA, Stead M. Large-scale electrophysiology: acquisition, compression, encryption, and storage of big data. J Neurosci Methods. 2009;180(1):185‐192. doi:10.1016/j.jneumeth.2009.03.022

* `Repository <https://github.com/msel-source/meflib>`_

Example
^^^^^^^^^^


.. code-block:: python

    # Requirements
    # Python 3.6
    # pymef - pip install pymef
    # numpy - if anaconda conda install -c anaconda numpy; else pip install numpy
    # pandas - same as numpy
    
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from AISC.MEF.io import MEF_READER, MEF_WRITTER
    
    path_file_from = '/Volumes/Hydrogen/filip/Shared/239_monitor_20200520_1639.csv'
    path_file_to = '/Users/m220339/Desktop/239_monitor_20200520_1639.mefd'
    
    
    # read data
    df = pd.read_csv(path_file_from, header=2)
    
    # read start measuring stamp
    date_str_format = '%Y/%m/%d %H:%M'
    with open(path_file_from, 'r') as fid:
        txt = fid.read(100).splitlines()
        date_str = txt[1]
        start_datetime = datetime.strptime(date_str, date_str_format)
    
    # add ms offset to the stamp
    start_datetime += timedelta(milliseconds=float(df['Time(ms)'][0]))
    
    # get samplimng frequency
    fs = int(1 / (np.diff(np.array(df['Time(ms)'])).mean() / 1e3)) # must be integer - otherwise resample
    
    # end timestamp
    end_datetime = start_datetime + timedelta(milliseconds=float(df.iloc[-1]['Time(ms)']) - float(df['Time(ms)'][0]))
    
    # passwords
    pass1 = 'pass1'
    pass2 = 'pass2'
    
    Writter = MEF_WRITTER(path_file_to, password=pass2, overwrite=True)
    
    units_conversion_factor = 1e-5
    Writter.section2_ts_dict['units_conversion_factor'] = units_conversion_factor
    channels = list(df.keys())[1:]
    for key in channels:
        data = np.array(df[key])
        data = np.round(data / units_conversion_factor).astype(np.int32)
    
        Writter.create_segment(data=data,
                               channel=key,
                               start_stamp=int(start_datetime.timestamp()*1e6),
                               end_stamp=int(end_datetime.timestamp()*1e6),
                               sampling_frequency=fs,
                               pwd1=pass1,
                               pwd2=pass2
                               )
    
    Writter.session.close()
    
    
    Reader = MEF_READER(path_file_to, password=pass2)
    signals = []
    
    for idx in range(Reader.bi.__len__()):
        key = Reader.bi[idx]['name']
        x = np.round(Reader.get_data(key, Reader.bi[idx]['start_time'][0], Reader.bi[idx]['end_time'][0])[0]).astype(np.int32)
        x = x * Reader.bi[idx]['ufact'][0]
        print('Overall Difference in signal ', key, ' ', (df[key][:-1] - x).sum())
        signals.append(x)
    

 
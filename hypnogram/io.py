import datetime
from io import StringIO
import xml.etree.ElementTree as ET
import os
import pandas as pd

from io import StringIO
import xml.etree.ElementTree as ET
import os
from hypnogram.CyberPSG import CyberPSGFile, CyberPSG_XML_Writter
from hypnogram.utils import time_to_utc
import pandas as pd

_hypnogram_colors = {
    'AWAKE': '#e7b233',
    'WAKE': '#e7b233',
    'Arrousal': '#d44b05',
    'REM': '#3500d3',
    'N1': '#2bc7c4',  # 2b7cc7
    'N2': '#2b5dc7',
    'N3': '#000000',
    'UNKNOWN': '#eaeded'
}

def load_CyberPSG(path):
    if not os.path.isfile(path):
        raise FileNotFoundError('[FILE ERROR]: File not found ' + path)
    fid = CyberPSGFile(path)
    annotations = fid.get_hypnogram()
    df = pd.DataFrame(annotations)
    return df

def save_CyberPSG(path, df):
    #TODO: Do Tests
    #TODO: Implement annotation groups etc

    fid = CyberPSG_XML_Writter(path)
    annotation_group = 'Import'
    annotation_types = list(df['annotation'].unique())
    df = time_to_utc(df)

    fid.add_AnnotationGroup(annotation_group)
    for atype in annotation_types:
        fid.add_AnnotationType(atype, groupAssociationId=annotation_group, color=_hypnogram_colors[atype])

    for row in df.iterrows():
        row = row[1]
        fid.add_Annotation(row['start'], row['end'], AnnotationTypeId=row['annotation'])
    fid.dump()






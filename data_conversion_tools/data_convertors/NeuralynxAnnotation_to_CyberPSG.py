# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Converting annotations from Neuralynx to CyberPSG-XML file. See example in readme file for this AISC sub-package.
"""


import os
import re
from datetime import datetime
import pytz
from copy import deepcopy
import argparse
import xml.etree.ElementTree as ET
from annotation_tools.XML_CyberPSG import *

xsi =  "http://www.w3.org/2001/XMLSchema-instance"
xlmns =  "http://tempuri.org/CyberPSG.xsd"
ns = {"xmlns:xsi": xsi}

class IDGenerator:
    def __init__(self):
        self.id_part0 = '00000000'
        self.id_part1 = '0000'
        self.id_part2 = '0000'
        self.id_part3 = '0000'
        self.id_part4 = '000000000000'

        self.len_idp0 = 8
        self.len_idp1 = 4
        self.len_idp2 = 4
        self.len_idp3 = 4
        self.len_idp4 = 12

    def check_id(self):
        error_str = "Length of id parts must be: \n" \
                    "id_part0 {0}\n" \
                    "id_part1 {1}\n" \
                    "id_part2 {2}\n" \
                    "id_part3 {3}\n" \
                    "id_part4 {4}\n" \
                    "".format(self.len_idp0, self.len_idp1, self.id_part2, self.id_part3, self.id_part4)

        if not isinstance(self.id_part0, str) or self.id_part0.__len__() != self.len_idp0:
            raise AssertionError("Id_part0 is wrong datatype or string of a wrong length. \n{0}".format(error_str))

        if not not isinstance(self.id_part1, str) and not self.id_part1.__len__() == self.len_idp1:
            raise AssertionError("Id_part1 is wrong datatype or string of a wrong length. \n{0}".format(error_str))

        if not isinstance(self.id_part2, str) and not self.id_part2.__len__() == self.len_idp2:
            raise AssertionError("Id_part2 is wrong datatype or string of a wrong length. \n{0}".format(error_str))

        if not isinstance(self.id_part3, str) and not self.id_part3.__len__() == self.len_idp3:
            raise AssertionError("Id_part3 is wrong datatype or string of a wrong length. \n{0}".format(error_str))

        if not isinstance(self.id_part4, str) and not self.id_part4.__len__() == self.len_idp4:
            raise AssertionError("Id_part4 is wrong datatype or string of a wrong length. \n{0}".format(error_str))

    @property
    def next_id(self):
        current_id = '-'.join((self.id_part0, self.id_part1, self.id_part2, self.id_part3, self.id_part4))
        current_id_part4 = self.id_part4
        current_id_part4_value = int(current_id_part4)
        self.id_part4 = ("{:0" + str(self.len_idp4) + "}" ).format(current_id_part4_value+1)
        return current_id

class CyberPSG_XML_Writter:
    _xsi =  "http://www.w3.org/2001/XMLSchema-instance"
    _xlmns =  "http://tempuri.org/CyberPSG.xsd"
    _ns = {"xmlns:xsi": xsi}

    def __init__(self, path):
        for attr, uri in self._ns.items():
            ET.register_namespace(attr.split(":")[1], uri)

        self.AnnotationGroup_IDGenerator = IDGenerator()
        self.AnnotationGroup_IDGenerator.id_part0 ='c0000000'

        self.AnnotationType_IDGenerator = IDGenerator()
        self.AnnotationType_IDGenerator.id_part0 ='b0000000'

        self.Annotation_IDGenerator = IDGenerator()
        self.Annotation_IDGenerator.id_part0 ='a0000000'

        self.AnnotationData = myXML_AnnotationData.init_EmptyInstance(file_path=path)
        self.AnnotationData.attributes['xmlns'] = self._xlmns

        self.AnnotationGroups = self.AnnotationData.add_child(myXML_AnnotationGroups)
        self.AnnotationTypes = self.AnnotationData.add_child(myXML_AnnotationTypes)
        self.Annotations = self.AnnotationData.add_child(myXML_Annotations)



    @property
    def nAnnotationGroups(self):
        return self.AnnotationGroupKeys.__len__()

    @property
    def AnnotationGroupKeys(self):
        return [AnnotationGroup['name'][0].param for AnnotationGroup in self.AnnotationGroups['AnnotationGroup']]

    @property
    def AnnotationGroupIDs(self):
        return [AnnotationGroup['id'][0].param for AnnotationGroup in self.AnnotationGroups['AnnotationGroup']]

    def add_AnnotationGroup(self, name_string=''):
        if name_string.__len__() == 0:
            name_string = 'AnnotationGroup{0}'.format(self.AnnotationGroups['AnnotationGroup'].__len__())

        for AnnotationGroup in self.AnnotationGroups['AnnotationGroup']:
            if AnnotationGroup['name'] == name_string:
                raise KeyError("AnnotationGroup \"" + name_string + "\" already exists!")

        AnnotationGroup = self.AnnotationGroups.add_child(myXML_AnnotationGroup)
        ID = AnnotationGroup.add_child(myXML_Id)
        ID.param = self.AnnotationGroup_IDGenerator.next_id
        name = AnnotationGroup.add_child(myXML_Name)
        name.param = name_string

    def remove_AnnotationGroup(self, item):
        if not (isinstance(item, int) and item >= 0 and item < self.AnnotationGroups['AnnotationGroup'].__len__()) and not isinstance(item, str):
            raise KeyError('Annotation group identifier must be an integer indexing Annotation groups or string representing a category name.')

        for idx, AnnotationGroup in enumerate(self.AnnotationGroups['AnnotationGroup']):
            if AnnotationGroup['name'][0].param == item:
                item = idx

        self.AnnotationGroups['AnnotationGroup'][item].Element.clear()
        del self.AnnotationGroups.AnnotationGroup[item]

    @property
    def nAnnotationTypes(self):
        return self.AnnotationTypeKeys.__len__()

    @property
    def AnnotationTypeKeys(self):
        return [AnnotationType['name'][0].param for AnnotationType in self.AnnotationTypes.AnnotationType]

    @property
    def AnnotationTypeIDs(self):
        return [AnnotationType['id'][0].param for AnnotationType in self.AnnotationTypes.AnnotationType]

    def add_AnnotationType(self, name_string="", groupAssociationId=None, color='#FF6C1FBA', note="", startsWithEpoch=False, stdDurationInSec=0):
        if name_string.__len__() == 0:
            name_string = 'AnnotationType{0}'.format(self.AnnotationTypes.__len__())

        for AnnotationType in self.AnnotationTypes.AnnotationType:
            if AnnotationType.name[0].param == name_string:
                raise KeyError("AnnotationType \"" + name_string + "\" already exists!")

        AnnotationType = self.AnnotationTypes.add_child(myXML_AnnotationType)
        ID = AnnotationType.add_child(myXML_Id, param=self.AnnotationType_IDGenerator.next_id)
        name = AnnotationType.add_child(myXML_Name, param=name_string)
        #stdDurationInSec = AnnotationType.add_child(myXML_stdDurationInSec, param=str(stdDurationInSec))
        #startsWithEpoch = AnnotationType.add_child(myXML_startsWithEpoch, param=str(startsWithEpoch))
        color = AnnotationType.add_child(myXML_Color, param=color)
        if note.__len__() > 0:
            note = AnnotationType.add_child(myXML_Note, param=note)

        if isinstance(groupAssociationId, str):
            is_AGroup_set = False
            for idx, AGroupName in enumerate(self.AnnotationGroupKeys):
                if AGroupName == groupAssociationId:
                    groupAssociationId = idx
                    is_AGroup_set = True

            if is_AGroup_set is False:
                raise KeyError('AnnotationType cannot be assigned to an AnnotationGroup \"' + groupAssociationId + '\". Available AnnotationGroups \"' + str(self.AnnotationGroupKeys ))

        if isinstance(groupAssociationId, int):
            group_id = self.AnnotationGroups.AnnotationGroup[groupAssociationId].id[0].param
            groupAssociation = AnnotationType.add_child(myXML_GroupAsociations)
            gAID = groupAssociation.add_child(myXML_Id, param=group_id)

    def remove_AnnotationType(self, item):
        if not (isinstance(item, int) and item >= 0 and item < self.AnnotationTypes.AnnotationType.__len__()) and not isinstance(item, str):
            raise KeyError('Annotation type identifier must be an integer indexing Annotation types or string representing a category name.')

        for idx, AnnotationTypeKey in enumerate(self.AnnotationTypesKeys):
            if AnnotationTypeKey == item:
                item = idx

        self.AnnotationTypes.AnnotationType[item].Element.clear()
        del self.AnnotationTypes.AnnotationType[item]

    @property
    def nAnnotations(self):
        return self.Annotations.Annotation.__len__()

    @property
    def AnnotationIDs(self):
        return [Annotation['id'][0].param for Annotation in self.Annotations.Annotation]

    def add_Annotation(self, start_datetime:datetime, end_datetime:datetime, AnnotationTypeId=0, note='', type='GlobalAnnotation'):
        start_s = start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")
        end_s = end_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")

        Annotation = self.Annotations.add_child(myXML_Annotation)
        Annotation.attributes[ET.QName(xsi, "type")] = type
        ID = Annotation.add_child(myXML_Id, param=self.Annotation_IDGenerator.next_id)

        StartTimeUtc = Annotation.add_child(myXML_StartTimeUtc, param=start_s)
        endTimeUtc = Annotation.add_child(myXML_endTimeUtc, param=end_s)



        if isinstance(AnnotationTypeId, str):
            is_AType_set = False
            for idx, ATypepName in enumerate(self.AnnotationTypeKeys):
                if ATypepName == AnnotationTypeId:
                    AnnotationTypeId = idx
                    is_AType_set = True

            if is_AType_set is False:
                raise KeyError('AnnotationType cannot be assigned to an AnnotationGroup \"' + AnnotationTypeId + '\". Available AnnotationGroups \"' + str(self.AnnotationGroupKeys ))

        ATypeId_s = self.AnnotationTypeIDs[AnnotationTypeId]
        AnnotationTypeId = Annotation.add_child(myXML_AnnotationTypeId, param=ATypeId_s)

        if note.__len__() > 0:
            note = Annotation.add_child(myXML_Note, param=note)

    def remove_Annotation(self, item):
        if not (isinstance(item, int) and item >= 0 and item < self.Annotations.Annotation.__len__()) and not isinstance(item, str):
            raise KeyError('Annotation identifier must be an integer indexing Annotation groups or string representing a category name.')

        for idx, AnnotationTypeKey in enumerate(self.AnnotationTypesKeys):
            if AnnotationTypeKey == item:
                item = idx

        self.Annotations.Annotation[item].Element.clear()
        del self.Annotations.Annotation[item]

    def dump(self, path=None):
        def indent(elem, level=0): # creates new_line and indents in the xml file
            i = "\r\n" + level*"  "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "  "
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for elem in elem:
                    indent(elem, level+1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i

        AnnotationDataElement = deepcopy(self.AnnotationData.Element) # deep copy - indent not in the original file
        for idx, Element in enumerate(AnnotationDataElement):
            if Element.__len__() == 0:
                AnnotationDataElement[idx].clear()
                del AnnotationDataElement[idx]

        indent(AnnotationDataElement) # indents
        xml = ET.ElementTree(AnnotationDataElement) # Tree to write
        if isinstance(path, type(None)):
            path = self.AnnotationData._file_path

        xml.write(path, encoding='utf-8', xml_declaration=True) # dump file


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Converting annotations from Neuralynx to CyberPSG-XML file')
    parser.add_argument(
        '--path_from', dest='PATH_FROM', required=True,
        help='Path to the dir txt file with annotations.')
    parser.add_argument(
        '--path_to', dest='PATH_TO', required=True,
        help='Dir path to output xml file, where the file will be generated,')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Check windows or linux and set separator
    if os.name == 'nt': DELIMITER = '\\' # Windows
    else: DELIMITER = '/' # posix

    args = parse_args()
    path_txt = args.PATH_FROM
    path = args.PATH_TO

    #path = '/mnt/Hydrogen/filip/2020/MSEL/pokus.xml'
    #path_txt = '/mnt/Hydrogen/filip/2020/MSEL/events.txt'

    # read and parse annotation file
    with open(path_txt, 'r') as fid:
        txt = fid.read()
    txt = [sample for sample in re.split(r'[,][\s]|[\n]', txt) if sample.__len__() > 0]


    # Create AnnotationType Note and paste annotations
    CAnnots = CyberPSG_XML_Writter(path)
    CAnnots.add_AnnotationType('Note')
    for k in range(0, txt.__len__(), 2):
        sstamp = datetime.fromtimestamp(int(txt[k]) / 1e6).astimezone(pytz.utc)
        estamp = datetime.fromtimestamp(0.001 + int(txt[k]) / 1e6).astimezone(pytz.utc)
        note = txt[k+1]
        CAnnots.add_Annotation(sstamp, estamp, AnnotationTypeId='Note', note=note)
    CAnnots.dump()



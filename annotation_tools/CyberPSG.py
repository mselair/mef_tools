from io import StringIO
import xml.etree.ElementTree as ET
import os
import pandas as pd

class TwoWayDict(dict):
    def __init__(self, d=None):
        if not isinstance(d, type(None)):
            if isinstance(d, dict):
                for k, v in d.items():
                    self[k] = v



    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        return dict.__len__(self) // 2

class AnnotationFile:
    def __init__(self, path=None):
        self.namespaces = TwoWayDict() # list of namespaces - key=prefix; value=uri
        self.Element = None
        self.path = path

        if path:
            if os.path.isfile(path):
                self.read_file(path_xml=path)


    def register_namespace(self, prefix, uri):
        if not prefix in self.namespaces.keys():
            self.namespaces[prefix] = uri
            ET.register_namespace(prefix, uri)

    def parse_namespaces(self, path_xml):
        schema = open(path_xml, 'r').read()
        namespaces = dict([
            node for _, node in ET.iterparse(StringIO(schema), events=['start-ns'])
        ])
        for k, v in namespaces.items(): self.register_namespace(k, v)

    def register_CyberPSG_namespace(self):
        self.register_namespace('', "http://tempuri.org/CyberPSG.xsd")
        self.register_namespace('xsi', "http://www.w3.org/2001/XMLSchema-instance")


    def read_file(self, path_xml):
        self.parse_namespaces(path_xml)
        self.Element = ET.parse(path_xml).getroot()

    def get_annotations(self):
        annotationTypes = {}
        #for


    def get_annotation_types(self):
        types = {}
        for Element in self.Element:
            tag = Element.tag
            if '}' in tag: tag = tag.split('}')[-1]
            if tag == 'AnnotationTypes':
                for AnnotationTypeElement in Element:
                    name = None
                    id = None
                    for SubElement in AnnotationTypeElement:
                        subtag = SubElement.tag
                        if '}' in subtag: subtag = subtag.split('}')[-1]
                        if subtag == 'id': id = SubElement.text
                        if subtag == 'name': name = SubElement.text
                    types[id] = name
        return types


    def get_annotations(self):
        types = self.get_annotation_types()
        types = TwoWayDict(types)
        annotations = []

        for Element in self.Element:
            tag = Element.tag
            if '}' in tag: tag = tag.split('}')[-1]
            if tag == 'Annotations':
                for AnnotationElement in Element:
                    annot = {
                        'annotation': None,
                        'startTimeUtc': None,
                        'endTimeUtc': None
                    }

                    for SubElement in AnnotationElement:
                        subtag = SubElement.tag
                        if '}' in subtag: subtag = subtag.split('}')[-1]
                        if subtag == 'annotationTypeId':  annot['annotationTypeId'] = types[types[SubElement.text]]
                        if subtag == 'startTimeUtc':  annot['startTimeUtc'] = SubElement.text
                        if subtag == 'endTimeUtc':  annot['endTimeUtc'] = SubElement.text
                    annotations += [annot]
        return annotations

def parse_CyberPSG_Annotations_xml(path):
    Annotations = AnnotationFile(path)
    annotationTypes = Annotations.get_annotation_types()
    annotations = Annotations.get_annotations()
    dfAnnotations = pd.DataFrame(annotations)
    return dfAnnotations, annotationTypes


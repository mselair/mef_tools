Neuralynx Annotation to CyberPSG convertor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash
    conda activate filip_dogs
    cd MSEL_folder
    AISC/XlterkConvertor/NeuralynxAnnotation_to_CyberPSG.py -path_from ExampleData/NeuralynxAnnotation.txt -path_to ExampleData/NeuralynxAnnotation_CBPSG.xml


CyberPSG_XML_writter
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from AISC.XltekConvertor import CyberPSG_XML_Writter
    import pytz # python timezone
    from datetime import datetime, timedelta

    path = "/tmp/cyberpsg_annotation.xml"

    # Creates XML Structure
    CAnnots = CyberPSG_XML_Writter(path)

    # Create Annotation group
    CAnnots.add_AnnotationGroup()
    CAnnots.add_AnnotationGroup('G1')
    CAnnots.add_AnnotationGroup('G2')
    CAnnots.add_AnnotationGroup('G3')

    # Remove annotation groups
    CAnnots.remove_AnnotationGroup('G3')
    CAnnots.remove_AnnotationGroup(2)

    # Create Annotation type with or without association to a given annotation group
    CAnnots.add_AnnotationType('A1', groupAssociationId=0) # indexning by id
    CAnnots.add_AnnotationType('A2', groupAssociationId='G1') # indexing by name
    CAnnots.add_AnnotationType()
    #CAnnots.add_AnnotationType('A1', groupAssociationId='G1') # rises error - name already exists

    # create annotations
    s = datetime.now() # annotation start
    e = datetime.now() + timedelta(seconds=1) # annotation end
    CAnnots.add_Annotation(s, e) # add annotation without annot type association
    CAnnots.add_Annotation(s, e, AnnotationTypeId=1) # annotation type by index
    CAnnots.add_Annotation(s, e, AnnotationTypeId='A2') # by name

    CAnnots.dump() # writes file

    print(CAnnots.AnnotationGroupKeys)
    print(CAnnots.AnnotationGroups)



Neuralynx MEF-signal + Annotation to SignalPlant convertor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda activate filip_dogs
    cd MSEL_folder

    AISC/XlterkConvertor/NeuralynxAnnotation_to_SignalPlant.py -path_mef /path/to/mef.mefd -path_txt ExampleData/NeuralynxAnnotation.txt -path_to /output_folder
    # folder must exist. New file will be named after the original mef file.



XLTEK to CyberPSG Video Convertor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda activate filip_dogs
    cd MSEL_folder

    AISC/XlterkConvertor/XltekVideo_to_CyberPSG.py -path_from /path/to/folder/with/original/data -path_to /where/you/want/new/folder

from functools import partial
from detnet.nn import register_model
from .ssd import SingleShotDetectorWithClassifier

register_model('XSD', SingleShotDetectorWithClassifier)
register_model('xsd', partial(SingleShotDetectorWithClassifier, cls_add=True))
register_model('XAD', partial(SingleShotDetectorWithClassifier, attention=True))
register_model('XOD', partial(SingleShotDetectorWithClassifier, oc_net=True))

from typing import List
from functools import partial

from .base import BaseFeaturizer
from .composite import CompositeFeaturizer
from .molecules import MorganFPFeaturizer, RDKitDesc3DFeaturizer
# from .reactions import RxnOneHotFeaturizer

MOLFEATURIZER_GENERATORS = {
    "morgan3-1024": partial(MorganFPFeaturizer, radius=3, size=1024),
    "morgan2-1024": partial(MorganFPFeaturizer, radius=2, size=1024),
    "morgan3-2048": partial(MorganFPFeaturizer, radius=3, size=2048),
    "morgan2-2048": partial(MorganFPFeaturizer, radius=2, size=2048),
    "morgan3-4096": partial(MorganFPFeaturizer, radius=3, size=4096),
}

CONFFEATURIZER_GENERATORS = {
    "rdkit-desc3d": RDKitDesc3DFeaturizer
}

def get_featurizer(name: str | List[str]) -> BaseFeaturizer:
    if isinstance(name, str):
        try:
            return MOLFEATURIZER_GENERATORS[name]()
        except Exception:
            return CONFFEATURIZER_GENERATORS[name]()
    
    featurizers = [
        get_featurizer(nam) for nam in name
    ]
    return CompositeFeaturizer(featurizers)
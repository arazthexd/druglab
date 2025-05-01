from typing import List

from .base import BaseFeaturizer
from .composite import CompositeFeaturizer
from .molecules import MorganFPFeaturizer
from .reactions import RxnOneHotFeaturizer

NAME2FEATURIZER = {
    "morgan3-1024": MorganFPFeaturizer(radius=3, size=1024),
    "morgan2-1024": MorganFPFeaturizer(radius=2, size=1024),
    "morgan3-2048": MorganFPFeaturizer(radius=3, size=2048),
    "morgan2-2048": MorganFPFeaturizer(radius=2, size=2048),
    "morgan3-4096": MorganFPFeaturizer(radius=3, size=4096),
    "rxn-onehot": RxnOneHotFeaturizer()
}

def get_featurizer(name: str | List[str]) -> BaseFeaturizer:
    if isinstance(name, str):
        return NAME2FEATURIZER[name]
    
    featurizers = [
        get_featurizer(nam) for nam in name
    ]
    return CompositeFeaturizer(featurizers)
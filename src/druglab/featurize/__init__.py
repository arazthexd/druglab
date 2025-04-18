from .base import BaseFeaturizer
from .molecules import MorganFPFeaturizer
from .reactions import RxnOneHotFeaturizer

NAME2FEATURIZER = {
    "morgan3-1024": MorganFPFeaturizer(radius=3, size=1024),
    "rxn-onehot": RxnOneHotFeaturizer()
}
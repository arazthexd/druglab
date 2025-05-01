from typing import List, Any
from dataclasses import dataclass, field

from .drawopts import DrawOptions
from .features import (
    PharmFeatures, PharmArrowFeats, PharmSphereFeats,
)

@dataclass
class PharmFeatureType:
    name: str
    compradii: float = field(default=1.7, repr=False)
    drawopts: DrawOptions = field(default_factory=DrawOptions, repr=False)

    def initiate_features(self):
        raise NotImplementedError()
    
    def __eq__(self, value):
        assert isinstance(value, PharmFeatureType)
        if self.name == value.name:
            return True
        return False

class PharmArrowType(PharmFeatureType):
    def initiate_features(self):
        return PharmArrowFeats()

class PharmSphereType(PharmFeatureType):
    def initiate_features(self):
        return PharmSphereFeats()
__all__ = [
    "get_default_pharm_featurizer",
    "get_default_pharm_fper",
    "InternalStericAdjuster",
    "PharmFeatures", "PharmArrowFeats", "PharmSphereFeats",
    "PharmFeatureType", "PharmArrowType", "PharmSphereType",
    "PharmGenerator", "BASE_DEFINITIONS_PATH",
    "PharmGroup",
    "PharmDefinitions", "PharmDefaultParser",
    "Pharmacophore",
    "PharmProfile",
    "PharmProfiler", "PharmDefaultProfiler",
    "PharmFingerprinter", "PharmCompositeFingerprinter",
    "PharmDistFingerprinter", "PharmCosineFingerprinter", 
    "PharmTypeIDFingerprinter",
    "PharmFPFeaturizer",
    "PharmStorageProfiler"
]

from .adjusters import InternalStericAdjuster
from .features import PharmFeatures, PharmArrowFeats, PharmSphereFeats
from .ftypes import PharmFeatureType, PharmArrowType, PharmSphereType
from .generator import PharmGenerator, BASE_DEFINITIONS_PATH
from .groups import PharmGroup
from .parser import PharmDefinitions, PharmDefaultParser
from .pharmacophore import Pharmacophore
from .pprofile import PharmProfile
from .profiler import PharmProfiler, PharmDefaultProfiler
from .fingerprint import (
    PharmFingerprinter, PharmCompositeFingerprinter,
    PharmDistFingerprinter, PharmCosineFingerprinter, 
    PharmTypeIDFingerprinter
)
from .featurizer import PharmFPFeaturizer
# from .storage import PharmStorageProfiler

def get_default_pharm_fper(fpsize=7000):
    return PharmCompositeFingerprinter(
        fpers=[
            PharmTypeIDFingerprinter(),
            PharmCompositeFingerprinter(
                fpers=[
                    PharmDistFingerprinter(bins=(1, 4, 7, 10, 13)),
                    PharmDistFingerprinter(bins=(2, 5, 8, 11, 14)),
                    PharmDistFingerprinter(bins=(3, 6, 9, 12, 15)),
                ],
                mode="sum"
            ),
            PharmCompositeFingerprinter(
                fpers=[
                    PharmCosineFingerprinter(bins=(-0.5, 0.0, 0.5, 1.0)),
                    PharmCosineFingerprinter(bins=(-0.75, -0.25, 0.25, 0.75, 1.0)),
                ],
                mode="sum"
            )
        ],
        mode="prod",
        fpsize=fpsize
    )

def get_default_pharm_featurizer(fpsize=7000, ngroup=4):
    pgen = PharmGenerator()
    pgen.load_file(BASE_DEFINITIONS_PATH)

    return PharmFPFeaturizer(
        generator=pgen,
        adjuster=InternalStericAdjuster(),
        profiler=PharmDefaultProfiler(pgen.ftypes, ngroup=ngroup, mindist=0.2),
        fingerprinter=get_default_pharm_fper(fpsize=fpsize)
    )
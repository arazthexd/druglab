from __future__ import annotations
from typing import Dict, Any, Optional, Literal, List, Tuple
import logging
import h5py
import dill
from tqdm import tqdm

import numpy as np

from rdkit import Chem

from ..storage import StorageFeaturizer, MolStorage
from .generator import PharmGenerator, BASE_DEFINITIONS_PATH
from .profiler import (
    PharmProfiler, PharmDefaultProfiler, PharmProfile, 
    Pharmacophore, PharmacophoreList
)
from .adjusters import PharmAdjuster

logger = logging.getLogger(__name__)
MolStorage.required_object_keys
class MolStorageWithProfiles(MolStorage):
    @property
    def required_object_keys(self) -> List[str]:
        return ['molecules', 'pharms', 'profiles']
    
    @property
    def save_dtypes(self) -> Dict[str, type]:
        return {'molecules': h5py.string_dtype(),
                'pharms': h5py.string_dtype(),
                'profiles': h5py.string_dtype()}

    def __init__(self, 
                 molecules = None, 
                 pharms = None,
                 profiles = None,
                 features = None, 
                 metadata = None, 
                 conformer_features = None):
        super().__init__(molecules, features, metadata, conformer_features)
        if pharms is None:
            pharms = [None] * len(self.molecules)
        if profiles is None:
            profiles = [None] * len(self.molecules)
        
        self._objects['pharms'] = pharms
        self._objects['profiles'] = profiles

        self._pgen: PharmGenerator = None
        self._pler: PharmProfiler = None

    @classmethod
    def from_mols(cls,
                  mols: MolStorage | List[Chem.Mol],
                  pgen: PharmGenerator = None,
                  pler: PharmProfiler = None,
                  single_conf: bool = False):
        
        storage = cls(molecules=mols.molecules,
                      features=mols.features,
                      metadata=mols.metadata,
                      conformer_features=mols.conformer_features)
        
        storage.set_generator(pgen)
        storage.set_profiler(pler)

        if pgen is not None:
            storage.generate_pharms(single_conf=single_conf)
            if pler is not None:
                storage.generate_profiles()
        
        return storage

    def set_generator(self, pgen: PharmGenerator):
        self._pgen = pgen

    def set_profiler(self, pler: PharmProfiler):
        self._pler = pler

    def generate_pharms(self, 
                        overwrite: bool = False,
                        single_conf: bool = False):
        assert self._pgen

        # TODO: multi proc
        print("Generating pharmacophores...")
        for i, pharm in enumerate(tqdm(self.pharms)):
            if not isinstance(pharm, Pharmacophore) or overwrite:
                mol = self.molecules[i]
                confid = -1 if single_conf else 'all'
                self._objects['pharms'][i] = self._pgen.generate(mol, confid)
    
    def generate_profiles(self, 
                          overwrite: bool = False):
        assert self._pler

        # TODO: multi proc
        print("Generating profiles...")
        for i, prof in enumerate(self.profiles):
            if not isinstance(prof, PharmProfile) or overwrite:
                pharm = self.pharms[i]
                self._objects['profiles'][i] = self._pler.profile(pharm)

    @property
    def pharms(self) -> List[Pharmacophore]:
        return self._objects['pharms']

    @property
    def profiles(self) -> List[PharmProfile]:
        return self._objects['profiles']
    
    def add_molecule(self, molecule):
        if not self._pgen or not self._pler:
            raise ValueError("Must set profiler before adding molecules")
        profile = self._pler.profile(self._pgen.generate(molecule))
        self._objects['profiles'].append(profile)
        return super().add_molecule(molecule)
    
    def add_molecules(self, molecules):
        if not self._pgen or not self._pler:
            raise ValueError("Must set profiler before adding molecules")
        profiles = [self._pler.profile(self._pler.profile(mol)) 
                    for mol in molecules]
        self._objects['profiles'].extend(profiles)
        return super().add_molecules(molecules)
    
    def get_conformers_as_storage(self):
        new: MolStorageWithProfiles = super().get_conformers_as_storage()
        
        stid = 0
        for i, nc in enumerate(self.num_conformers_per_molecule):
            endid = stid + nc
            for j, idx in enumerate(range(stid, endid)):
                if self.pharms[i].__class__ == Pharmacophore:
                    if j == 0:
                        new.pharms[idx] = self.pharms[i]
                    else:
                        new.pharms[idx] = \
                            self._pgen.generate(new.molecules[idx])
                elif isinstance(self.pharms[i], PharmacophoreList):
                    new.pharms[idx] = self.pharms[i][j]   
                new.profiles[idx] = self.profiles[i][j]
            stid = endid

        return new
    
    def extend(self, other: MolStorage | MolStorageWithProfiles):
        super().extend(other)
        if isinstance(other, MolStorageWithProfiles):
            self._objects['pharms'].extend(other.pharms)
            self._objects['profiles'].extend(other.profiles)
        return
    
    def subset(self, indices) -> MolStorageWithProfiles:
        new = super().subset(indices) 
        # TODO: unify API (return MolStorage or wProfs?)
        new._objects['pharms'] = [self.pharms[i] for i in indices]
        new._objects['profiles'] = [self.profiles[i] for i in indices]
        return new
    
    def get_save_ready_objects(self) -> Dict[str, List[Any]]:
        save_objs = super().get_save_ready_objects()
        profs = []
        for prof in self.profiles:
            profs.append(dill.dumps(prof, 0))
        pharms = []
        for pharm in self.pharms:
            if isinstance(pharm, PharmacophoreList):
                for p in pharm.pharms:
                    p.conformer = None
            else:
                pharm.conformer = None
            pharms.append(dill.dumps(pharm, 0))
        save_objs['pharms'] = pharms
        save_objs['profiles'] = profs
        return save_objs
    
    def get_load_ready_objects(self, 
                               db: h5py.File, 
                               indices: List[int] = None) \
                                -> Dict[str, List[Any]]:
        load_objs = super().get_load_ready_objects(db, indices)
        pharms = [dill.loads(pharm) for pharm in db['pharms']]
        profs = [dill.loads(prof) for prof in db['profiles']]
        load_objs['pharms'] = pharms
        load_objs['profiles'] = profs
        return load_objs


    
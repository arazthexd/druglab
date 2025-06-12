from typing import List, Dict, Any, Optional, Callable
import logging

from rdkit import Chem
from rdkit.Chem import (
    rdDistGeom, rdForceFieldHelpers, 
    TorsionFingerprints,
    rdMolAlign,
    SaltRemover,
    rdBase
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Cluster import Butina # type: ignore

from ..modify import BaseStorageModifier
from ..base import BaseStorage

logger = logging.getLogger(__name__)

class _CGenParamHolder:
    def __init__(self, base: str = None):
        if base is None:
            base = 'ETKDGv3'
        self.base = base
        self.maxit = None

    def get_params(self) -> rdDistGeom.EmbedParameters:
        params: rdDistGeom.EmbedParameters = getattr(rdDistGeom, self.base)()
        params.maxIterations = self.maxit
        return params

class GenericMoleculePrepper(BaseStorageModifier):
    """Generic molecule preparation class that standardizes and prepares molecules.
    
    This class handles various molecule preparation steps including:
    - Salt removal and neutralization
    - Hydrogen addition/removal
    - Conformer generation
    - Conformer optimization
    - Conformer clustering
    - Conformer alignment
    """
    
    def __init__(self,
                 # Standardization parameters
                 remove_salts: bool = True,
                 keep_largest_frag: bool = True,
                 neutralize: bool = True,
                 standardize_tautomers: bool = True,
                 # Hydrogen handling
                 addhs: bool = False,
                 removehs: bool = False,
                 # Conformer generation
                 cgen: bool = False, 
                 cgen_n: int = 1, 
                 cgen_maxatts: Optional[int] = None,
                 cgen_parambase: Optional[str] = None,
                 # Conformer optimization
                 copt: bool = False,
                 copt_nthreads: int = 1,
                 copt_maxits: int = 200,
                 # Conformer clustering
                 cclust: bool = False,
                 cclust_tol: float = 0.3,
                 cclust_afteropt: bool = True,
                 # Conformer alignment
                 calign: bool = False,
                 # Multiprocessing parameters
                 n_processes: int = 1,
                 chunk_size: Optional[int] = None):
        """Initialize the molecule prepper.
        
        Args:
            remove_salts: Remove salt counterparts from molecules
            neutralize: Neutralize charged molecules
            standardize_tautomers: Standardize tautomeric forms
            addhs: Add hydrogen atoms
            removehs: Remove hydrogen atoms
            cgen: Generate conformers
            cgen_n: Number of conformers to generate
            cgen_maxatts: Maximum attempts for conformer generation
            cgen_params: Parameters for conformer generation
            copt: Optimize conformers
            copt_nthreads: Number of threads for optimization
            copt_maxits: Maximum iterations for optimization
            cclust: Cluster conformers
            cclust_tol: Tolerance for conformer clustering
            cclust_afteropt: Whether to cluster after optimization
            calign: Align conformers
            n_processes: Number of processes for parallel processing
            chunk_size: Chunk size for batch processing
        """
        super().__init__(n_processes=n_processes, chunk_size=chunk_size)
        
        # Standardization settings
        self.remove_salts = remove_salts
        self.keep_largest_frag = keep_largest_frag
        self.neutralize = neutralize
        self.standardize_tautomers = standardize_tautomers
        
        # Molecule preparation settings
        self.addhs = addhs
        self.removehs = removehs
        self.cgen = cgen
        self.cgen_n = cgen_n
        self.cgen_paramgen: _CGenParamHolder = _CGenParamHolder(cgen_parambase)
        self.cgen_maxatts = cgen_maxatts or (cgen_n * 4)
        self.copt = copt
        self.copt_nthreads = copt_nthreads
        self.copt_maxits = copt_maxits
        self.cclust = cclust
        self.cclust_tol = cclust_tol
        self.cclust_afteropt = cclust_afteropt
        self.calign = calign
        
        # Set up conformer generation parameters
        self.cgen_paramgen.maxit = self.cgen_maxatts
        
        # # Initialize standardization tools
        # if self.remove_salts:
        #     self.salt_remover = SaltRemover.SaltRemover()
        # if self.keep_largest_frag:
        #     self.fragment_chooser = rdMolStandardize.LargestFragmentChooser(
        #         preferOrganic=True
        #     )
        # if self.neutralize:
        #     self.uncharger = rdMolStandardize.Uncharger()
        # if self.standardize_tautomers:
        #     self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
    
    def _standardize_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        """Standardize a molecule by removing salts and neutralizing charges.
        
        Args:
            mol: Input molecule
            
        Returns:
            Standardized molecule
        """
        if mol is None:
            return None
            
        # Remove salts
        if self.remove_salts:
            blocker = rdBase.BlockLogs()
            salt_remover = SaltRemover.SaltRemover()
            mol = salt_remover.StripMol(mol)
            del blocker

        # Keep largest fragment
        if self.keep_largest_frag:
            blocker = rdBase.BlockLogs()
            fragment_chooser = rdMolStandardize.LargestFragmentChooser(
                preferOrganic=True
            )
            mol = fragment_chooser.choose(mol)
            del blocker
        
        # Neutralize charges
        if self.neutralize:
            blocker = rdBase.BlockLogs()
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)
            del blocker
        
        # Standardize tautomers
        if self.standardize_tautomers:
            blocker = rdBase.BlockLogs()
            tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
            mol = tautomer_enumerator.Canonicalize(mol)
            del blocker
        
        return mol
    
    def _cluster_conformers(self, mol: Chem.Mol) -> Chem.Mol:
        """Cluster conformers and keep only representative ones.
        
        Args:
            mol: Molecule with conformers
            
        Returns:
            Molecule with clustered conformers
        """
        if mol.GetNumConformers() <= 1:
            return mol
            
        tfds = TorsionFingerprints.GetTFDMatrix(mol)
        clusts = Butina.ClusterData(tfds, 
                                    mol.GetNumConformers(), 
                                    self.cclust_tol, 
                                    isDistData=True, 
                                    reordering=True)
        
        # Get representative conformers (first in each cluster)
        clustcs = [clust[0] for clust in clusts]
        
        # Remove non-representative conformers
        conformers_to_remove = []
        for cid in range(mol.GetNumConformers()):
            if cid not in clustcs:
                conformers_to_remove.append(cid)
        
        # Remove conformers in reverse order to maintain indices
        for cid in sorted(conformers_to_remove, reverse=True):
            mol.RemoveConformer(cid)
            
        return mol

    def _prepare_molecule(self, mol: Chem.Mol) -> Chem.Mol: # TODO: Dupl???
        """Prepare a single molecule with all specified operations.
        
        Args:
            mol: Input molecule
            
        Returns:
            Prepared molecule
        """
        if mol is None:
            return None
            
        # Standardize molecule first
        mol = self._standardize_molecule(mol)
        if mol is None:
            return None
        
        # Add hydrogens if requested
        if self.addhs:
            addcs = False
            if mol.GetNumConformers() > 0:
                if mol.GetConformer().Is3D():
                    addcs = True
            mol = Chem.AddHs(mol, addCoords=addcs)
        
        # Generate conformers
        if self.cgen:
            rdDistGeom.EmbedMultipleConfs(mol, self.cgen_n, 
                                          self.cgen_paramgen.get_params())
        
        # Cluster conformers before optimization if requested
        if self.cclust and not self.cclust_afteropt:
            mol = self._cluster_conformers(mol)
        
        # Optimize conformers
        if self.copt and mol.GetNumConformers() > 0:
            rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, 
                                                            self.copt_nthreads, 
                                                            self.copt_maxits)
        
        # Cluster conformers after optimization if requested
        if self.cclust and self.cclust_afteropt:
            mol = self._cluster_conformers(mol)
        
        # Align conformers
        if self.calign and mol.GetNumConformers() > 1:
            rdMolAlign.AlignMolConformers(mol)
        
        # Remove hydrogens if requested
        if self.removehs:
            mol = Chem.RemoveHs(mol)
        
        # Ensure conformer IDs are sequential
        conformers = mol.GetConformers()
        for i in range(mol.GetNumConformers()):
            conformers[i].SetId(i)
        
        return mol

    def _prepare_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        """Prepare a single molecule with all specified operations.
        
        Args:
            mol: Input molecule
            
        Returns:
            Prepared molecule
        """
        if mol is None:
            return None
            
        # Standardize molecule first
        mol = self._standardize_molecule(mol)
        if mol is None:
            return None
        
        # Add hydrogens if requested
        if self.addhs:
            addcs = False
            if mol.GetNumConformers() > 0:
                if mol.GetConformer().Is3D():
                    addcs = True
            mol = Chem.AddHs(mol, addCoords=addcs)
        
        # Generate conformers
        if self.cgen:
            suc = rdDistGeom.EmbedMultipleConfs(mol, 
                                                self.cgen_n, 
                                                self.cgen_paramgen.get_params())
            if len(suc) == 0:
                return None
        
        # Cluster conformers before optimization if requested
        if self.cclust and not self.cclust_afteropt:
            mol = self._cluster_conformers(mol)
        
        # Optimize conformers
        if self.copt and mol.GetNumConformers() > 0:
            rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, 
                                                          self.copt_nthreads, 
                                                          self.copt_maxits)
        if self.copt and mol.GetNumConformers() == 0:
            return None
        
        # Cluster conformers after optimization if requested
        if self.cclust and self.cclust_afteropt:
            mol = self._cluster_conformers(mol)
        
        # Align conformers
        if self.calign and mol.GetNumConformers() > 1:
            rdMolAlign.AlignMolConformers(mol)
        
        # Remove hydrogens if requested
        if self.removehs:
            mol = Chem.RemoveHs(mol)
        
        # Ensure conformer IDs are sequential
        conformers = mol.GetConformers()
        for i in range(mol.GetNumConformers()):
            conformers[i].SetId(i)
        
        return mol
    
    def prepare(self, mol: Chem.Mol) -> Chem.Mol:
        return self._prepare_molecule(mol)
    
    def modify_objects(self, 
                       object_dict: Dict[str, Any],
                       context_data: Any) -> Optional[Dict[str, Any]]:
        """Modify a single object by preparing its molecule.
        
        Args:
            object_dict: Dictionary containing molecule data
            context_data: Context data (unused for molecule preparation)
            
        Returns:
            Modified object dictionary with prepared molecule, or None if failed
        """
        try:
            key = 'molecules'
            mol = object_dict[key]
            if mol is None:
                logger.warning("Molecule is None")
                return None
            
            # Prepare the molecule
            prepared_mol = self.prepare(mol)
            if prepared_mol is None:
                logger.warning("Failed to prepare molecule")
                return None
            
            # Create a copy of the object dict and update the molecule
            modified_dict = object_dict.copy()
            modified_dict[key] = prepared_mol
            
            return modified_dict
            
        except Exception as e:
            logger.error(f"Error in modify_objects: {e}")
            return None
    
    def apply_modifications(self, 
                            storage: BaseStorage,
                            modified_objects: List[Optional[Dict[str, Any]]],
                            context_data: Any,
                            remove_fails: bool = False) -> None:
        """Apply modifications to storage.
        
        This method handles the storage update and optionally removes failed 
            objects.
        """
        # Filter out None values (failed modifications)
        success_indices = [i for i, obj in enumerate(modified_objects) 
                           if obj is not None]
        
        if remove_fails:
            # Keep only successful modifications
            filtered_objects = [obj for obj in modified_objects 
                                if obj is not None]
            modified_objects = filtered_objects
            
            # Update storage with filtered objects
            from ..utils import _list_to_dict
            storage._objects = _list_to_dict(modified_objects)
            
            # Update features to match the filtered objects
            if hasattr(storage, '_features') and storage._features is not None:
                storage._features = storage._features.subset(success_indices)
        else:
            # Replace failed modifications with original objects
            from ..utils import _dict_to_list, _list_to_dict
            orig_objects = _dict_to_list(storage.objects)
            
            final_objects = [
                modified_obj if modified_obj is not None else orig_objects[i]
                for i, modified_obj in enumerate(modified_objects)
            ]
            
            storage._objects = _list_to_dict(final_objects)
        
        # Log preparation statistics
        n_success = len(success_indices)
        n_total = len(modified_objects) \
            if not remove_fails \
                else len(success_indices) + \
                    (len(modified_objects) - len(success_indices))
        n_failed = n_total - n_success
        
        logger.info(f"Molecule preparation completed: {n_success}/{n_total} "
                    f"successful, {n_failed} failed")
        if remove_fails and n_failed > 0:
            logger.info(f"Removed {n_failed} failed molecules from storage")
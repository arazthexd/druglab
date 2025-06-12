from __future__ import annotations
from typing import List, Any, Dict, Optional
import logging

import h5py

from ..storage import (
    BaseStorage, StorageFeatures, StorageMetadata,
    serialize_objects, deserialize_objects,
    CustomFuncFilter
)
from .route import SynthesisRoute, StepType

logger = logging.getLogger(__name__)

class SynRouteStorage(BaseStorage):
    """Storage for synthesis routes."""
    
    @property
    def required_object_keys(self) -> List[str]:
        return ['routes']
    
    @property
    def save_dtypes(self) -> Dict[str, type]:
        return {'routes': h5py.string_dtype()}
    
    def __init__(self,
                 routes: Optional[List[SynthesisRoute]] = None,
                 features: Optional[StorageFeatures] = None,
                 metadata: Optional[StorageMetadata] = None):
        """
        Initialize SynRouteStorage.
        
        Args:
            routes: List of SynthesisRoute objects
            features: StorageFeatures container for routes
            metadata: Storage metadata
        """
        # Convert routes to proper format for BaseStorage
        objects = None
        if routes is not None:
            objects = {'routes': routes}
            
        super().__init__(
            objects=objects,
            features=features,
            metadata=metadata
        )
    
    @property
    def routes(self) -> List[SynthesisRoute]:
        """Get routes (alias for objects['routes'])."""
        return self._objects['routes']
    
    def add_route(self, route: SynthesisRoute) -> None:
        """Add a single route to storage."""
        self._objects['routes'].append(route)
    
    def add_routes(self, routes: List[SynthesisRoute]) -> None:
        """Add multiple routes to storage."""
        self._objects['routes'].extend(routes)
    
    def _get_route_reactant_indices(self, route: SynthesisRoute) -> set:
        """Get all reactant indices used in a route."""
        if route is None:
            return set()
        return set(route.get_reactant_indices())
    
    def _get_route_intermediate_indices(self, route: SynthesisRoute) -> set:
        """Get all product indices used in a route."""
        if route is None:
            return set()
        return set(route.get_intermediate_indices())
    
    def _get_route_rxn_indices(self, route: SynthesisRoute) -> set:
        """Get all reaction indices used in a route."""
        if route is None:
            return set()
        return set(route.get_reaction_indices())
    
    @staticmethod
    def _update_route_indices(route: SynthesisRoute, 
                              reactant_index_mapping: \
                                Optional[Dict[int, int]] = None,
                              product_index_mapping: \
                                Optional[Dict[int, int]] = None,
                              rxn_index_mapping: \
                                Optional[Dict[int, int]] = None) -> SynthesisRoute:
        """Update indices in a route based on provided mappings."""
        if route is None:
            return None
        
        updated_route = route.copy()
        
        for step in updated_route.steps:
            # Update reactant indices
            if step.step_type == StepType.ADD_REACTANT \
                and reactant_index_mapping:
                if step.mol_idx is not None:
                    if step.mol_idx in reactant_index_mapping:
                        step.mol_idx = reactant_index_mapping[step.mol_idx]
                    else:
                        # Molecule was removed, mark as invalid
                        step.mol_idx = -1

            # Update reactant indices
            if step.step_type == StepType.APPLY_REACTION \
                and product_index_mapping:
                if step.mol_idx is not None:
                    if step.mol_idx in product_index_mapping:
                        step.mol_idx = product_index_mapping[step.mol_idx]
                    else:
                        # Molecule was removed, mark as invalid
                        step.mol_idx = -1
            
            # Update reaction indices
            if rxn_index_mapping and step.rxn_idx is not None:
                if step.rxn_idx in rxn_index_mapping:
                    step.rxn_idx = rxn_index_mapping[step.rxn_idx]
                else:
                    # Reaction was removed, mark as invalid
                    step.rxn_idx = -1
        
        return updated_route
    
    def subset_by_component_ids(self,
                                reactant_ids: List[int] = None,
                                intermediate_ids: List[int] = None,
                                rxn_ids: List[int] = None,
                                n_processes: int = 1) -> SynRouteStorage:
        """
        Subset routes that contain all specified reactant molecule IDs.
        
        Args:
            reactant_ids: List of reactant molecule indices that must all be 
                present in routes
            intermediate_ids: List of product molecule indices that must all be 
                present in routes
            rxn_ids: List of reaction indices that must all be 
                present in routes
            
        Returns:
            New SynRouteStorage with filtered routes and updated indices
        """
        if reactant_ids:
            reactant_ids_set = set(reactant_ids)
        if intermediate_ids:
            intermediate_ids_set = set(intermediate_ids)
        if rxn_ids:
            rxn_ids_set = set(rxn_ids)

        def filter_func(route: SynthesisRoute):
            if route is None:
                return False
            if reactant_ids:
                if not self._get_route_reactant_indices(route)\
                    .issubset(reactant_ids_set):
                    return False
            if intermediate_ids:
                if not self._get_route_intermediate_indices(route)\
                    .issubset(intermediate_ids_set):
                    return False
            if rxn_ids:
                if not self._get_route_rxn_indices(route)\
                    .issubset(rxn_ids_set):
                    return False
            return True
        
        storage_filter = CustomFuncFilter(
            filter_func,
            input_keys=['routes'],
            n_processes=n_processes,
            name='SubsetByComponentIDs'
        )
        keep_indices = storage_filter.filter(self)

        # remove_ids = set()
        
        # for i, route in enumerate(self.routes):
        #     if route is None:
        #         remove_ids.add(i)
        #         continue
            
        #     if reactant_ids:
        #         route_rids = self._get_route_reactant_indices(route)
        #         if not route_rids.issubset(reactant_ids_set):
        #             remove_ids.add(i)
        #             continue
            
        #     if intermediate_ids:
        #         route_intids = self._get_route_intermediate_indices(route)
        #         if not route_intids.issubset(intermediate_ids_set):
        #             remove_ids.add(i)
        #             continue

        #     if rxn_ids:
        #         route_rxnids = self._get_route_rxn_indices(route)
        #         if not route_rxnids.issubset(rxn_ids_set):
        #             remove_ids.add(i)
        #             continue
        
        # # Create subset using parent method
        # keep_indices = [i for i in range(len(self.routes)) 
        #                 if i not in remove_ids]
        subset_storage: SynRouteStorage = self.subset(keep_indices)
        
        # Create mappings...
        if reactant_ids:
            rmapping = {
                old_idx: new_idx 
                for new_idx, old_idx in enumerate(sorted(reactant_ids))
            }
        else:
            rmapping = None
        
        if intermediate_ids:
            pmapping = {
                old_idx: new_idx 
                for new_idx, old_idx in enumerate(sorted(intermediate_ids))
            }
        else:
            pmapping = None
        
        if rxn_ids:
            rmapping = {
                old_idx: new_idx 
                for new_idx, old_idx in enumerate(sorted(rxn_ids))
            }
        else:
            rmapping = None
        
        # Update molecule indices in all routes
        updated_routes = []
        for route in subset_storage.routes:
            updated_route = self._update_route_indices(
                route, 
                reactant_index_mapping=rmapping,
                product_index_mapping=pmapping, 
                rxn_index_mapping=rmapping
            )
            updated_routes.append(updated_route)
        
        subset_storage._objects['routes'] = updated_routes
        return subset_storage
    
    def get_routes_using_component(self, 
                                   reactant_id: int = None,
                                   intermediate_id: int = None,
                                   rxn_id: int = None) -> List[int]:
        """Get indices of routes that use a specific component."""
        route_indices = []
        for i, route in enumerate(self.routes):
            if route is None:
                continue
            if reactant_id is not None:
                if reactant_id not in self._get_route_reactant_indices(route):
                    continue
            if intermediate_id is not None:
                if intermediate_id not in self._get_route_intermediate_indices(route):
                    continue
            if rxn_id is not None:
                if rxn_id not in self._get_route_rxn_indices(route):
                    continue
            route_indices.append(i)
        return route_indices
    
    def get_save_ready_objects(self) -> Dict[str, List[Any]]:
        """Convert routes to serialized strings for saving."""
        return {'routes': serialize_objects(self.routes)}
    
    def get_load_ready_objects(self, 
                               db: h5py.File, 
                               indices: Optional[List[int]] = None) -> Dict[str, List[Any]]:
        """Load routes from serialized strings."""
        if indices is None:
            serialized_data = db['routes'][:]
        else:
            serialized_data = db['routes'][indices]
        
        # Convert bytes back to proper format for deserialization
        serialized_routes = [data.encode() if isinstance(data, str) else data 
                             for data in serialized_data]
        
        return {'routes': deserialize_objects(serialized_routes)}
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({len(self)} routes, "
                f"{self.num_features} feature sets)")
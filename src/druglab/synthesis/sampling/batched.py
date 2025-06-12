from __future__ import annotations
from typing import List, Tuple, Optional
from dataclasses import dataclass
import random
import logging
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdBase
from tqdm import tqdm

from ...storage import RxnStorage, MolStorage
from ..route import SynthesisRoute, StepType

logger = logging.getLogger(__name__)

class SamplingUtils:
    """Utility functions for sampling route templates."""
    
    @staticmethod
    def count_completions(l: int, avail: int, # noqa: E741
                          Lmin: int, Lmax: int) -> int:  
        """Count how many valid completions exist from the current state."""
        if l == Lmax:
            return 1 if avail == 1 else 0

        total = 0
        # Option 1: Terminate the sequence now if allowed.
        if l >= Lmin and avail == 1:
            total += 1

        # Option 2: Extend the sequence.
        for k in range(avail + 1):
            new_avail = avail - k + 1
            total += SamplingUtils.count_completions(l + 1, new_avail, 
                                                     Lmin, Lmax)
        return total

    @staticmethod
    def sample_state(seq: List[int], l: int, avail: int,  # noqa: E741
                     Lmin: int, Lmax: int) -> List[int]:
        """Recursively sample complete sequence starting from current state."""
        if l == Lmax:
            if avail != 1:
                raise ValueError("Invalid state reached at maximum length!")
            return seq

        options = []
        counts = []

        # Option to stop
        if l >= Lmin and avail == 1:
            options.append(("stop", None))
            counts.append(1)

        # Option to continue
        for k in range(avail + 1):
            new_avail = avail - k + 1
            cnt = SamplingUtils.count_completions(l + 1, new_avail, Lmin, Lmax)
            options.append(("continue", k))
            counts.append(cnt)

        total_options = sum(counts)
        if total_options == 0:
            raise ValueError(f"No valid completions from state: "
                             f"length={l}, avail={avail}")

        # Randomly choose an option, weighted by the number of completions
        r = random.randrange(total_options)
        for opt, cnt in zip(options, counts):
            if r < cnt:
                chosen = opt
                break
            r -= cnt

        if chosen[0] == "stop":
            return seq
        else:
            k = chosen[1]
            new_seq = seq + [k]
            new_avail = avail - k + 1
            return SamplingUtils.sample_state(new_seq, l + 1, new_avail, 
                                              Lmin, Lmax)

    @staticmethod
    def sample_sequence_variable(Lmin: int, Lmax: int) -> List[int]:
        """Sample a valid sequence with total length between Lmin and Lmax."""
        if Lmin < 1 or Lmax < Lmin:
            raise ValueError("Invalid length bounds.")
        return SamplingUtils.sample_state([0], 1, 1, Lmin, Lmax)

@dataclass
class SamplingConfig:
    """Configuration for synthesis route sampling."""
    min_steps: int = 1
    max_steps: int = 4
    n_routes_per_template: int = 100
    n_template_batches: int = 6
    allow_multi_prods: bool = True
    allow_branching: bool = True # TODO: Implement not branch version
    max_retries: int = 10 # TODO: Implement max retries
    random_seed: Optional[int] = None # TODO: This doesn't seem to work

class RouteTemplateSamplingComponent:
    """Component for sampling routes at a specific template step."""
    
    def __init__(self, n_intermediates_needed: int):
        self.n_intermediates_needed = n_intermediates_needed
    
    def sample_routes(self, 
                      rxn_storage: RxnStorage,
                      mol_storage: MolStorage,
                      generated_products: List[Chem.Mol],
                      intermediate_route_batches: List[List[SynthesisRoute]],
                      config: SamplingConfig,
                      n_routes: int = 100) -> List[SynthesisRoute]:
        """
        Sample routes that use the specified number of intermediates.
        
        Args:
            rxn_storage: Reaction storage
            mol_storage: Molecule storage (reactants)
            intermediate_route_batches: List of route batches that can provide 
                intermediates
            intermediate_molecules: List of intermediates (Chem.Mol)
            n_routes: Number of routes to sample
            config: Sampling config containing important settings
            
        Returns:
            List of newly sampled routes
        """
        assert len(intermediate_route_batches) == self.n_intermediates_needed
        
        # Make copies to avoid modifying original batches
        available_routes = [batch.copy() 
                            for batch in intermediate_route_batches]
        
        # Find reactions that can accept at least n_intermediates_needed 
        # reactants
        candidate_rxn_indices = [
            i for i in range(len(rxn_storage)) 
            if rxn_storage[i].GetNumReactantTemplates() \
                >= self.n_intermediates_needed
        ]
        
        if not candidate_rxn_indices:
            logger.warning(f"No reactions found that can accept "
                           F"{self.n_intermediates_needed} intermediates")
            return []
        
        results = []
        
        for _ in range(n_routes):
            try:
                routes = self._sample_single_route(
                    rxn_storage, 
                    mol_storage,
                    generated_products,
                    candidate_rxn_indices, 
                    available_routes,
                    config
                )
                if routes:
                    results.extend(routes)
            except Exception as e:
                logger.debug(f"Failed to sample route: {e}")
                continue
        
        return results
    
    def _sample_single_route(self, 
                             rxn_storage: RxnStorage,
                             mol_storage: MolStorage,
                             generated_products: List[Chem.Mol],
                             candidate_rxn_indices: List[int],
                             available_routes: List[List[SynthesisRoute]],
                             config: SamplingConfig) \
                                -> Optional[List[SynthesisRoute]]:
        """Sample a single route using intermediates."""
        
        # Select random reaction
        rxn_idx = random.choice(candidate_rxn_indices)
        rxn = rxn_storage.reactions[rxn_idx]
        n_reactants = rxn.GetNumReactantTemplates()
        
        # Select which reactant positions will use intermediates
        intermediate_positions = random.sample(range(n_reactants), 
                                               k=self.n_intermediates_needed)
        
        # Try to find compatible intermediate routes
        selected_routes: List[SynthesisRoute] = []
        current_reactants: List[Chem.Mol] = []
        current_rids: List[int | None] = []
        intermediate_counter = 0
        
        # Iterate over reactants that need choosing...
        for reactant_pos in range(n_reactants):

            # If the reactant is an intermediate, need to find one that matches
            if reactant_pos in intermediate_positions:
                
                # Every time an intermediate is used, the last set is considered
                batch_idx = len(available_routes) - intermediate_counter - 1
                if not available_routes[batch_idx]:
                    return None  # No available routes in this batch
                intermediate_counter += 1
                
                compatible_route = None
                route_idx = None
                
                # Get reactant template for this position
                reactant_template = rxn.GetReactants()[reactant_pos]
                
                # Iterate over available routes and products
                iterator = enumerate(available_routes[batch_idx])
                for i, candidate_route in iterator:
                    # Find candidate product object
                    candidate_product: Chem.Mol = \
                        generated_products[candidate_route.steps[-2].mol_idx]
                    
                    # If a route is compatible, save it and move on to rest
                    if candidate_product.HasSubstructMatch(reactant_template):
                        compatible_route = candidate_route
                        compatible_product = candidate_product
                        route_idx = i
                        break
                
                if compatible_route is None:
                    return None  # No compatible route found
                
                # Remove the used route and product from available routes/prods
                selected_routes.append(
                    available_routes[batch_idx].pop(route_idx)
                )
                # Use the final product index as the reactant
                current_reactants.append(compatible_product)
                current_rids.append(None)
                
            else:
                # Use a fresh reactant molecule
                reactant_group = rxn_storage.get_reactant_group(rxn_idx, 
                                                                reactant_pos)
                if not reactant_group or not reactant_group.mol_indices:
                    return None
                
                mol_idx = random.choice(reactant_group.mol_indices)
                current_reactants.append(mol_storage[mol_idx])
                current_rids.append(mol_idx)

        # Perform reaction
        assert len(current_reactants) == n_reactants
        results = rxn.RunReactants(current_reactants) 
        products = [p for ps in results for p in ps]
        
        # Checks
        if len(products) == 0:
            logger.error(f"No products found for reaction {rxn_idx} "
                         "when sampling")
            return None
        if len(products) > rxn.GetNumProductTemplates() \
            and not config.allow_multi_prods:
            return None
        
        combined_routes = []
        for i, product in enumerate(products):
            Chem.SanitizeMol(product)
            product.UpdatePropertyCache()

            # Create the combined route
            combined_route = SynthesisRoute()
            combined_route.start()
            
            # Add all previous routes in reverse
            # REASON: Every intermediate use, uses the last unused product in route
            for prev_route in reversed(selected_routes):
                combined_route.steps.extend(prev_route.steps[1:-1])  # Skip START/END
            
            # Add reactants for this step
            for reactant_pos in range(n_reactants):
                if reactant_pos in intermediate_positions:
                    combined_route.use_intermediate(
                        selected_routes.pop(0).steps[-1].mol_idx
                    )
                else:
                    combined_route.add_reactant(current_rids[reactant_pos])
            
            # Apply the reaction
            combined_route.apply_reaction(rxn_idx, 
                                          product_idx=i, 
                                          mol_idx=len(generated_products))
            generated_products.append(product)
            combined_route.end()

            combined_routes.append(combined_route)
            
        return combined_routes

class EfficientSynthesisRouteSampler:
    """Memory-efficient synthesis route sampler using template-based batching."""
    
    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed) # TODO: Doesn't seem to work
    
    def sample_routes(self, 
                      rxn_storage: RxnStorage,
                      mol_storage: MolStorage,
                      n_processes: int = 1,
                      show_progress: bool = True,
                      only_final: bool = True) \
                        -> Tuple[List[SynthesisRoute], List[Chem.Mol]]:
        """
        Sample synthesis routes using efficient template-based batching.
        
        Args:
            rxn_storage: Storage containing reactions and molecules
            n_processes: Number of parallel processes
            show_progress: Whether to show progress bar
            only_final: If True, return only final routes; 
                if False, return all intermediate routes
            
        Returns:
            List of sampled synthesis routes and list of 
                generated intermediates
        """
        if len(rxn_storage) == 0:
            logger.warning("Empty reaction storage provided")
            return []
        
        rdBase.DisableLog("rdApp.*")
        
        # Prepare tasks for parallel processing
        tasks = []
        for batch_id in range(self.config.n_template_batches):
            tasks.append((batch_id, rxn_storage, mol_storage, only_final))
        
        all_routes = []
        all_products = []
        
        if n_processes == 1:
            # Single-process execution
            iterator = tqdm(tasks, desc="Sampling route batches") \
                if show_progress else tasks
            for task in iterator:
                routes, prods = self._sample_template_batch(*task)
                for route in routes:
                    for step in route.steps:
                        if step.step_type == StepType.APPLY_REACTION:
                            step.mol_idx += len(all_products)
                all_routes.extend(routes)
                all_products.extend(prods)
        else:
            # Multi-process execution
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                futures = [executor.submit(self._sample_template_batch, *task) 
                           for task in tasks]
                
                iterator = tqdm(futures, desc="Sampling route batches") \
                    if show_progress else futures
                for future in iterator:
                    try:
                        routes, prods = future.result()
                        for route in routes:
                            for step in route.steps:
                                if step.step_type == StepType.APPLY_REACTION:
                                    step.mol_idx += len(all_products)
                        all_routes.extend(routes)
                        all_products.extend(prods)
                    except Exception as e:
                        logger.error(f"Template batch sampling failed: {e}")
        
        logger.info(f"Sampled {len(all_routes)} synthesis routes")
        return all_routes, all_products
    
    def _sample_template_batch(self, 
                               batch_id: int, 
                               rxn_storage: RxnStorage,
                               mol_storage: MolStorage,
                               only_final: bool) \
                                -> Tuple[List[SynthesisRoute], List[Chem.Mol]]:
        """Sample a batch of routes using a random template sequence."""
        
        # Sample a template sequence 
        # (number of intermediates to use at each step)
        template_sequence = SamplingUtils.sample_sequence_variable(
            Lmin=self.config.min_steps,
            Lmax=self.config.max_steps
        )
        
        # Track available route batches at each step, 
        # as well as generated products
        available_route_batches: List[List[SynthesisRoute]] = []
        all_sampled_routes = []
        generated_products: List[Chem.Mol] = []
        
        for step_idx, n_intermediates in enumerate(template_sequence):
            # Create sampling component for this step
            component = RouteTemplateSamplingComponent(n_intermediates)
            
            # Get the required number of intermediate route batches
            if n_intermediates > len(available_route_batches):
                logger.warning(f"Not enough intermediate batches "
                               f"for step {step_idx}")
                break
            
            # Pop the required batches (most recent first)
            intermediate_batches = []
            for _ in range(n_intermediates): # TODO: Reverse or not?
                intermediate_batches.append(available_route_batches.pop()) 
            
            # Sample new routes using these intermediates
            new_routes = component.sample_routes(
                rxn_storage, 
                mol_storage,
                generated_products,
                intermediate_batches,
                config=self.config,
                n_routes=self.config.n_routes_per_template
            )
            
            if not new_routes:
                logger.debug(f"No routes sampled at step {step_idx}")
                break
            
            # Add new routes to available batches for next step
            available_route_batches.append(deepcopy(new_routes))
            
            # Collect routes if needed
            if not only_final:
                all_sampled_routes.extend(deepcopy(new_routes))
        
        # Return final routes or all routes based on flag
        if only_final and available_route_batches:
            return available_route_batches[-1], generated_products
        else:
            return all_sampled_routes, generated_products
    
    def validate_routes(self, 
                        routes: List[SynthesisRoute], 
                        rxn_storage: RxnStorage) \
                            -> Tuple[List[SynthesisRoute], List[SynthesisRoute]]:
        """
        Validate synthesis routes.
        
        Returns:
            Tuple of (valid_routes, invalid_routes)
        """
        valid_routes = []
        invalid_routes = []
        
        for route in routes:
            if route.validate(rxn_storage):
                valid_routes.append(route)
            else:
                invalid_routes.append(route)
        
        logger.info(f"Validation: {len(valid_routes)} valid, "
                    f"{len(invalid_routes)} invalid routes")
        return valid_routes, invalid_routes
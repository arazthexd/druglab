from __future__ import annotations
from typing import List, Tuple, Any
from dataclasses import dataclass, field
import random, math
import mpire

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdDepictor, rdBase
from rdkit.Chem.rdChemReactions import ChemicalReaction as Rxn

from ..storage import RxnStorage
from .route import SynthesisRoute, ActionTypes
from .storage import SynRouteStorage
from .utils import SamplingUtils

class SynRouteSampler:
    def __init__(self, 
                 min_steps: int = 1, 
                 max_steps: int = 4,
                 n_template_batch: int = 6,
                 n_route_batch: int = 100,
                 templates: List[List[int]] | None = None):
        
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.n_template_batch = n_template_batch
        self.n_route_batch = n_route_batch
        self.templates = templates

    def sample(self, 
               rxns: RxnStorage, 
               only_final: bool = True, 
               num_processes: int = 2,
               return_storage: bool = True):
        
        rdBase.DisableLog("rdApp.*")
        
        def task(i):
            try:
                return self._sample_for_random_template(rxns, only_final)
            except:
                return None
        
        with mpire.WorkerPool(num_processes) as pool:
            samproutes_process = pool.map(
                task,
                range(self.n_template_batch),
                progress_bar=True
            )
            samproutes_process = [r for r in samproutes_process 
                                  if r is not None]
        
        if return_storage:
            return SynRouteStorage(sum(samproutes_process, start=[]))
        return sum(samproutes_process, start=[])

    def _sample_for_random_template(self, 
                                    rxns: RxnStorage, 
                                    only_final: bool = True):
        sampled_routes = []

        template_seq = SamplingUtils.sample_sequence_variable(
            Lmin=self.min_steps,
            Lmax=self.max_steps
        )

        available_routes: List[List[SynthesisRoute]] = []
        for n_uprods in template_seq:
            component = _SynRouteSamplingComponent(n_uprods)
            uprod_candidates = [available_routes.pop() 
                                for _ in range(n_uprods)]
            results = component.sample(rxns, 
                                       *uprod_candidates, 
                                       n_batch=self.n_route_batch)
            assert len(results) > 0
            available_routes.append(results)
        
        if only_final:
            sampled_routes.extend(available_routes[-1])
        
        else:
            [sampled_routes.extend(avs) for avs in available_routes]
    
        return sampled_routes

class _SynRouteSamplingComponent:
    def __init__(self, n_uprods: int):
        self.n_uprods = n_uprods

    def sample(self, 
               rxns: RxnStorage, 
               *uprod_candidates: List[SynthesisRoute],
               n_batch: int = 100):
        assert len(uprod_candidates) == self.n_uprods
        uprod_candidates = tuple([upcs.copy() for upcs in uprod_candidates])
        
        rxn_candidate_ids = [
            i for i in range(len(rxns)) 
            if rxns[i].GetNumReactantTemplates() >= self.n_uprods
        ]
        
        rxn_storage = rxns.subset(rxn_candidate_ids)
        if len(rxn_storage) == 0:
            return []
        rxnids = [random.randint(0, len(rxn_storage)-1) 
                  for _ in range(n_batch)]

        results: List[SynthesisRoute] = []
        for rxnid in rxnids:
            rxn: Rxn = rxn_storage[rxnid]
            n_rs = rxn.GetNumReactantTemplates()

            current_route = SynthesisRoute(check=False)
            current_route.start()
            
            prev_routes: List[SynthesisRoute] = []
            
            uprodids = random.sample(range(n_rs), k=self.n_uprods)
            current_rs: List[Chem.Mol] = []
            for rid in range(n_rs):
                reactant_template = rxn.GetReactants()[rid]
                reactant_candidates = rxn_storage.mstores[rxnid][rid]

                if rid not in uprodids:
                    r: Chem.Mol = reactant_candidates.sample()
                    current_rs.append(r)
                    continue
                
                upcrs = uprod_candidates[uprodids.index(rid)]
                upcid = None
                for i, upcr in enumerate(upcrs):
                    upcr: SynthesisRoute
                    if upcr.products[-1].HasSubstructMatch(reactant_template):
                        upcid = i
                        current_rs.append(upcr.products[-1])
                        break
                
                if upcid is None:
                    break

                prev_routes = [upcrs.pop(upcid)] + prev_routes

            if len(prev_routes) < self.n_uprods:
                continue
            elif len(prev_routes) > self.n_uprods:
                raise ValueError("should be equal or lower...")
            
            try:
                products = [p for ps in rxn.RunReactants(current_rs) for p in ps]
                [Chem.SanitizeMol(p) for p in products]
                [rdDepictor.Compute2DCoords(p) for p in products]
            except:
                continue

            final_route = SynthesisRoute()
            final_route.start()
            final_route = sum(prev_routes, start=final_route)

            for rid in range(n_rs):
                if rid in uprodids:
                    final_route.use_product()
                else:
                    final_route.add_reactant(current_rs[rid])
            
            final_route.add_reaction(rxn)
            final_route.add_product(random.choice(products))
            final_route.end()
            results.append(final_route)

        return results
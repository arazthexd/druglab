from __future__ import annotations
from typing import List, Union, Tuple, Callable, Dict

import random
from itertools import product
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions as rdRxn
from rdkit.Chem.rdChemReactions import ChemicalReaction as Rxn

from ..storage import MolStorage, RxnStorage
from .route import SynthesisRoute, ActionTypes

class RxnProductSampler:
    def __init__(self,
                 rxn: Rxn,
                 rpools: List[MolStorage | RxnProductSampler]):
        self.rpools = rpools.copy()
        self.rxn = rxn

        self._isrun = False

        self._reactants: List[Chem.Mol] = []
        self._products: List[Chem.Mol] = []
    
    def reset(self):
        self._isrun = False

    def sample(self, n_st: int = 1) -> List[Chem.Mol]:
        if self._isrun:
            return self._products
        
        rcandidates: List[List[Chem.Mol]] = []
        for rpool in self.rpools:
            if isinstance(rpool, MolStorage):
                rcandidates.append(rpool.sample(n_st))
            elif isinstance(rpool, RxnProductSampler):
                rcandidates.append(rpool.sample(n_st))
            else:
                raise TypeError("rpool must be MolStorage or RxnProductSampler")
        
        for i in range(len(rcandidates)):
            if isinstance(rcandidates[i], Chem.Mol):
                rcandidates[i] = [rcandidates[i]]
        
        rscombs = list(product(*rcandidates))
        random.shuffle(rscombs)

        for rs in rscombs:
            self._reactants = rs
            self._products = [
                prod 
                for prods in self.rxn.RunReactants(self._reactants) 
                for prod in prods
            ]
            [Chem.SanitizeMol(p) for p in self._products]
            [p.UpdatePropertyCache() for p in self._products]
            if len(self._products) > 0:
                break

        self._isrun = True
        return self._products
    
    def routes(self) -> List[SynthesisRoute]:
        prev_routes: List[SynthesisRoute] = []
        for rid in range(len(self._reactants)):
            if isinstance(self.rpools[rid], RxnProductSampler):
                routes = self.rpools[rid].routes()
                correct_route = [
                    ro for ro in routes 
                    if Chem.MolToSmiles(ro.products[-1]) == \
                        Chem.MolToSmiles(self._reactants[rid])
                ][0]
                prev_routes.append(correct_route)
        
        route = SynthesisRoute()
        route.start()
        route: SynthesisRoute = sum(reversed(prev_routes), start=route)
        
        for rid in range(len(self._reactants)):
            if isinstance(self.rpools[rid], RxnProductSampler):
                route.use_product()
            
            if isinstance(self.rpools[rid], MolStorage):
                route.add_reactant(self._reactants[rid])

        route.add_reaction(self.rxn)

        route_plus_ps: List[SynthesisRoute] = []
        for product in self._products:
            out_route: SynthesisRoute = route.copy()
            out_route.add_product(product)
            out_route.end()
            route_plus_ps.append(out_route)
        
        return route_plus_ps
    
    def has_avaiable_input(self) -> bool:
        return sum([isinstance(rpool, MolStorage) 
                    for rpool in self.rpools]) > 0
    
    def available_rids(self) -> List[int]:
        return [i for i in range(len(self.rpools))
                if isinstance(self.rpools[i], MolStorage)]
        
@dataclass
class SynRouteSamplerOpts:
    max_sample_attempts: int = 100
    max_construct_attempts: int = 10
    rpool_sample_size: int = 10
    min_steps: int = 2
    max_steps: int = 4

class SynRouteSampler:
    def __init__(self,
                 bbs: MolStorage,
                 rxns: RxnStorage,
                 bb2rxnr: Dict[int, List[Tuple[int, int]]] = None,
                 processed: bool = False,
                 options: SynRouteSamplerOpts = None):
        
        self.bbs = bbs
        self.rxns = rxns

        if not processed:
            self.bbs.clean()
            self.rxns.add_mols(self.bbs)
            self.rxns.clean()
        
        if bb2rxnr is None:
            bb2rxnr = self.rxns.match_mols(bbs)
        self.bb2rxnr = bb2rxnr

        if options is None:
            options = SynRouteSamplerOpts()
        self.options = options
    
    def sample(self, only_last: bool = True) -> List[SynthesisRoute]:
        
        for _ in range(self.options.max_construct_attempts):
            nodes = self.construct()
            for i in range(self.options.max_sample_attempts):
                [node.reset() for node in nodes]
                products = nodes[-1].sample(self.options.rpool_sample_size)
                if len(products) > 0:
                    break
            
            if only_last:
                return nodes[-1].routes()
            return [route for node in nodes for route in node.routes()]

    def construct(self) -> List[RxnProductSampler]:
        n_steps = random.randint(self.options.min_steps,
                                 self.options.max_steps)
        
        nodes: List[RxnProductSampler] = []
        for i in range(n_steps):
            rxnid = random.randint(0, len(self.rxns)-1)
            node = RxnProductSampler(
                rxn=self.rxns[rxnid],
                rpools=self.rxns.mstores[rxnid]
            )
            nodes.append(node)

        for i in range(n_steps-2, -1, -1):
            current_node = nodes[i]

            available_targets = []
            for j in range(i+1, n_steps):
                if nodes[j].has_avaiable_input():
                    available_targets.append(j)

            target_idx: int = random.choice(available_targets)
            target_node: RxnProductSampler = nodes[target_idx]
            rid = random.choice(target_node.available_rids())
            target_node.rpools[rid] = current_node

        return nodes


    # def sample_construct(self):
    #     n_branches = random.randint(self.options.min_branches, 
    #                                 self.options.max_branches)
    #     n_steps = random.randint(self.options.min_steps,
    #                              self.options.max_steps)
    #     n_steps = max(n_steps, n_branches+1)
    #     n_fellows = n_steps - n_branches
    #     print(n_branches, n_steps, n_fellows)
        
    #     depth = random.randint(self.options.min_depth,
    #                            self.options.max_depth)

    #     branch_samplers: List[RxnProductSampler] = []
    #     for _ in range(n_branches):
    #         rxnid = random.choice(range(len(self.rxns)))
    #         rxn = self.rxns[rxnid]
    #         fellow_sampler = RxnProductSampler(
    #             rxn=rxn,
    #             rpools=self.rxns.mstores[rxnid]
    #         )
    #         branch_samplers.append(fellow_sampler)
        
    #     fellow_samplers: List[RxnProductSampler] = []
    #     for _ in range(n_fellows):
    #         rxnid = random.choice(range(len(self.rxns)))
    #         rxn = self.rxns[rxnid]
    #         fellow_sampler = RxnProductSampler(
    #             rxn=rxn,
    #             rpools=self.rxns.mstores[rxnid]
    #         )
    #         fellow_samplers.append(fellow_sampler)

    #     for branch_sampler in branch_samplers:
    #         n_connects = random.randint(1, len(fellow_samplers))
    #         connect_idx = list(range(len(fellow_samplers)))
    #         random.shuffle(connect_idx)
    #         connect_idx = connect_idx[:n_connects]
    #         for idx in connect_idx:
    #             rids = [i for i in range(len(fellow_samplers[idx].rpools))
    #                     if isinstance(fellow_samplers[idx].rpools[i],
    #                                   MolStorage)]
    #             if len(rids) == 0:
    #                 continue
    #             rid = random.choice(rids)
    #             fellow_samplers[idx].rpools[rid] = branch_sampler
            
    #     for i, fellow_sampler in enumerate(fellow_samplers[:-1]):
    #         tempdepth = min(depth, len(fellow_samplers)-i-1)
    #         n_connects = random.randint(1, tempdepth) # len(fellow_samplers)-i-1
    #         connect_idx = list(range(i+1, i+1+tempdepth)) # len(fellow_samplers)
    #         random.shuffle(connect_idx)
    #         connect_idx = connect_idx[:n_connects]
    #         for idx in connect_idx:
    #             rids = [i for i in range(len(fellow_samplers[idx].rpools))
    #                     if isinstance(fellow_samplers[idx].rpools[i],
    #                                   MolStorage)]
    #             if len(rids) == 0:
    #                 continue
    #             rid = random.choice(rids)
    #             fellow_samplers[idx].rpools[rid] = fellow_sampler

    #     return branch_samplers + fellow_samplers
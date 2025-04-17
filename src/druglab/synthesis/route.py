from __future__ import annotations
from typing import List, Union, Tuple, Callable

import random
from itertools import product
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions as rdRxn
from rdkit.Chem.rdChemReactions import ChemicalReaction as Rxn

class ActionTypes:
    START = 0
    REACTANT = 1
    REACTION = 2
    PRODUCT = 3
    USEPROD = 4
    END = 5

    @classmethod
    def idx2str(cls, idx: int):
        return {
            0: "START",
            1: "REACTANT",
            2: "REACTION",
            3: "PRODUCT",
            4: "USEPROD",
            5: "END",
        }[idx]

@dataclass(repr=False)
class _SequenceMember:
    type: int
    idx: int = None

    def __repr__(self):
        if self.type in [ActionTypes.START, ActionTypes.END]:
            return f"SeqMember({ActionTypes.idx2str(self.type)})"
        return f"SeqMember({ActionTypes.idx2str(self.type)}, {self.idx})"
    
    def copy(self):
        return _SequenceMember(self.type, self.idx)
    
class SynthesisRoute:
    def __init__(self, 
                    seq: List[_SequenceMember] = None,
                    reactants: List[Chem.Mol] = None,
                    reactions: List[Rxn] = None,
                    products: List[Chem.Mol] = None):
        self.seq = seq or list()
        self.reactants = reactants or list()
        self.reactions = reactions or list()
        self.products = products or list()
    
    def __repr__(self):
        return f"SynthesisRoute(" + \
            f"seq={len(self.seq)}, " + \
            f"reactants={len(self.reactants)}, " + \
            f"reactions={len(self.reactions)}, " + \
            f"products={len(self.products)})"
    
    def start(self):
        assert len(self.seq) == 0
        self.seq.append(_SequenceMember(ActionTypes.START))

    def add_reactant(self, reactant: Chem.Mol):
        assert self.seq[-1].type in [ActionTypes.START, 
                                     ActionTypes.PRODUCT,
                                     ActionTypes.REACTANT,
                                     ActionTypes.USEPROD]
        member = _SequenceMember(ActionTypes.REACTANT, len(self.reactants))
        self.seq.append(member)
        self.reactants.append(reactant)

    def add_reaction(self, rxn: Rxn):
        assert self.seq[-1].type in [ActionTypes.REACTANT, ActionTypes.USEPROD]

        member = _SequenceMember(ActionTypes.REACTION, len(self.reactions))
        self.seq.append(member)
        self.reactions.append(rxn)

    @property
    def last_unused_pseqid(self) -> int:
        helper_idx = 0
        for i, mem in enumerate(reversed(self.seq)):
            if mem.type == ActionTypes.USEPROD:
                helper_idx += 1
            if mem.type == ActionTypes.PRODUCT:
                if helper_idx == 0:
                    return len(self.seq)-i-1
                else:
                    helper_idx -= 1
    
    def add_product(self, product: Chem.Mol):
        member = _SequenceMember(ActionTypes.PRODUCT, len(self.products))
        self.seq.append(member)
        self.products.append(product)

    def use_product(self):
        last_free_pidx = self.seq[self.last_unused_pseqid].idx
        member = _SequenceMember(ActionTypes.USEPROD, last_free_pidx)
        self.seq.append(member)

    def end(self):
        self.seq.append(_SequenceMember(ActionTypes.END))

    def __add__(self, other: SynthesisRoute):
        selfseq: List[_SequenceMember] = [
            mem.copy() 
            for mem in self.seq 
            if mem.type not in [ActionTypes.START, ActionTypes.END]
        ]
        otherseq: List[_SequenceMember] = [
            mem.copy() 
            for mem in other.seq 
            if mem.type not in [ActionTypes.START, ActionTypes.END]
        ]
        for othermem in otherseq:
            if othermem.type in [ActionTypes.PRODUCT, ActionTypes.USEPROD]:
                othermem.idx += len(self.products)
            elif othermem.type in [ActionTypes.REACTANT]:
                othermem.idx += len(self.reactants)
            elif othermem.type in [ActionTypes.REACTION]:
                othermem.idx += len(self.reactions)

        return SynthesisRoute(
            seq=[_SequenceMember(ActionTypes.START)] + selfseq + otherseq,
            reactants=self.reactants + other.reactants,
            reactions=self.reactions + other.reactions,
            products=self.products + other.products
        )
    
    def copy(self):
        return SynthesisRoute(
            seq=self.seq.copy(),
            reactants=self.reactants.copy(),
            reactions=self.reactions.copy(),
            products=self.products.copy()
        )
    
    def visualize(self):
        vislist = []
        vl = []
        for mem in self.seq:
            if mem.type == ActionTypes.REACTANT:
                vl.append(self.reactants[mem.idx])
            elif mem.type in [ActionTypes.PRODUCT, ActionTypes.USEPROD]:
                vl.append(self.products[mem.idx])
            
            if mem.type == ActionTypes.PRODUCT: 
                vislist.append(vl)
                vl = []
        
        l = max([rxn.GetNumReactantTemplates() for rxn in self.reactions])+1
        vislist = [vl+[None]*(l-len(vl)) for vl in vislist]
        view = Draw.MolsToGridImage(
            mols=[v for vl in vislist for v in vl],
            molsPerRow=l,
            subImgSize=(300, 200)
        )
        return view
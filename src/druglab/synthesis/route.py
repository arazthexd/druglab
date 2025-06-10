from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

from ..storage import RxnStorage, MolStorage

class StepType(Enum):
    """Types of synthesis steps."""
    START = "start"
    ADD_REACTANT = "add_reactant"  
    USE_INTERMEDIATE = "use_intermediate"
    APPLY_REACTION = "apply_reaction"
    END = "end"

@dataclass
class SynthesisStep:
    """Represents a single step in synthesis route."""
    step_type: StepType
    rxn_idx: Optional[int] = None
    reactant_idx: Optional[int] = None  # Index in RxnStorage.mol_storage
    mol_idx: Optional[int] = None       # Index in RxnStorage.mol_storage or product store
    product_idx: Optional[int] = None   # Index in products list in reaction
    metadata: Dict = field(default_factory=dict)

@dataclass(repr=False)
class SynthesisRoute:
    """Memory-efficient synthesis route using indices."""
    steps: List[SynthesisStep] = field(default_factory=list)
    final_product_idx: Optional[int] = None
    route_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def add_step(self, step: SynthesisStep) -> None:
        """Add a synthesis step."""
        self.steps.append(step)
    
    def start(self) -> None:
        """Mark route start."""
        self.add_step(SynthesisStep(StepType.START))
    
    def end(self) -> None: #, final_product_idx: Optional[int] = None) -> None:
        """Mark route end."""
        # if final_product_idx is not None:
        #     self.final_product_idx = final_product_idx
        self.add_step(SynthesisStep(StepType.END))
    
    def add_reactant(self, mol_idx: int) -> None:
        """Add a reactant molecule."""
        self.add_step(SynthesisStep(
            step_type=StepType.ADD_REACTANT,
            mol_idx=mol_idx
        ))
    
    def use_intermediate(self, mol_idx: int) -> None:
        """Use an intermediate product (ID is from a prod store)."""
        self.add_step(SynthesisStep(
            step_type=StepType.USE_INTERMEDIATE, 
            mol_idx=mol_idx
        ))
    
    def apply_reaction(self, 
                       rxn_idx: int, 
                       product_idx: int = 0,
                       mol_idx: int = None) -> None:
        """Apply a reaction and select product."""
        self.add_step(SynthesisStep(
            step_type=StepType.APPLY_REACTION,
            rxn_idx=rxn_idx,
            product_idx=product_idx,
            mol_idx=mol_idx
        ))
        if self.final_product_idx is None:
            self.final_product_idx = product_idx
    
    def get_reactant_indices(self) -> List[int]:
        """Get all reactant molecule indices used in route."""
        return [step.mol_idx for step in self.steps 
                if step.step_type == StepType.ADD_REACTANT
                and step.mol_idx is not None]
    
    def get_intermediate_indices(self) -> List[int]:
        """Get all intermediate/product molecule indices used in route."""
        return [step.mol_idx for step in self.steps 
                if step.step_type == StepType.APPLY_REACTION
                and step.mol_idx is not None]
    
    def get_reaction_indices(self) -> List[int]:
        """Get all reaction indices used in route."""
        return [step.rxn_idx for step in self.steps 
                if step.step_type == StepType.APPLY_REACTION
                and step.rxn_idx is not None]
    
    def validate(self, 
                 rxn_storage: RxnStorage, 
                 mol_storage: MolStorage) -> bool:
        """Validate route against storage."""
        try:
            for step in self.steps:
                if step.rxn_idx is not None \
                    and step.rxn_idx >= len(rxn_storage):
                    return False
                if step.mol_idx is not None \
                    and step.mol_idx >= len(mol_storage):
                    return False
            return True
        except Exception:
            return False
    
    def __len__(self) -> int:
        """Number of synthesis steps."""
        return len([s for s in self.steps 
                    if s.step_type == StepType.APPLY_REACTION])
    
    def copy(self) -> SynthesisRoute:
        """Create a deep copy of the route."""
        return SynthesisRoute(
            steps=[SynthesisStep(
                step_type=step.step_type,
                rxn_idx=step.rxn_idx,
                reactant_idx=step.reactant_idx,
                mol_idx=step.mol_idx,
                product_idx=step.product_idx,
                metadata=step.metadata.copy()
            ) for step in self.steps],
            final_product_idx=self.final_product_idx,
            route_id=self.route_id,
            metadata=self.metadata.copy()
        )
    
    def __repr__(self):
        name = self.__class__.__name__
        return (
            f"{name}(\n\t" +
            "\n\t".join([step.__repr__() for step in self.steps]) +
            "\n)"
        )
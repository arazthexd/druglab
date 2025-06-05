from __future__ import annotations
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass

from rdkit import Chem

from rdkit.Chem.rdChemReactions import ChemicalReaction as Rxn

from .route import SynthesisRoute, StepType
from ..storage import RxnStorage, MolStorage

logger = logging.getLogger(__name__)

@dataclass
class SynthesisResult:
    """Result of executing a synthesis route."""
    success: bool
    product: Chem.Mol
    intermediates: List[Chem.Mol]
    error_message: Optional[str] = None
    step_results: List[Dict] = None

class SynthesisExecutor:
    """Execute synthesis routes to generate products."""
    
    def __init__(self, rxn_storage: RxnStorage, mol_storage: MolStorage):
        self.rxn_storage = rxn_storage
        self.mol_storage = mol_storage
    
    def execute_route(self, route: SynthesisRoute) -> SynthesisResult:
        """Execute a synthesis route and return products."""
        if not route.validate(self.rxn_storage, self.mol_storage):
            return SynthesisResult(
                success=False,
                product=None,
                intermediates=[],
                error_message="Route validation failed"
            )
        
        intermediates = []
        step_results = []
        current_reactants = []
        current_products = []
        
        try:
            for step in route.steps:
                step_result = {"step_type": step.step_type.value}
                
                if step.step_type == StepType.START:
                    step_result["action"] = "Starting synthesis"
                
                elif step.step_type == StepType.ADD_REACTANT:
                    mol = self.mol_storage.molecules[step.mol_idx]
                    current_reactants.append(mol)
                    step_result["action"] = f"Added reactant {step.mol_idx}"
                    step_result["molecule"] = mol
                
                elif step.step_type == StepType.USE_INTERMEDIATE:
                    if len(current_products) == 0:
                        raise ValueError("Empty current_product variable.")
                    step_result["action"] = (f"Using intermediate idx "
                                             f"{len(current_products)-1}")
                    step_result["molecule"] = current_products.pop()
                    current_reactants.append(step_result["molecule"])
                
                elif step.step_type == StepType.APPLY_REACTION:
                    rxn: Rxn = self.rxn_storage.reactions[step.rxn_idx]
                    
                    if not current_reactants:
                        raise ValueError("No reactants available for reaction")
                    
                    # Apply reaction
                    products = [p
                                for ps in rxn.RunReactants(current_reactants)
                                for p in ps]
                    
                    
                    if not products or len(products) == 0:
                        raise ValueError(f"Reaction {step.rxn_idx} "
                                         "produced no products")
                    
                    # Select the specified product
                    if step.product_idx >= len(products):
                        raise ValueError(f"Product index {step.product_idx} "
                                         "out of range")
                    
                    product = products[step.product_idx]
                    if product is None:
                        raise ValueError(f"Product {step.product_idx} is None")
                    
                    # Sanitize product
                    try:
                        Chem.SanitizeMol(product)
                    except Exception:
                        logger.warning(f"Could not sanitize product from "
                                       f"reaction {step.rxn_idx}")
                    
                    # Store as intermediate
                    intermediates.append(product)
                    current_products.append(product)
                    
                    step_result["action"] = f"Applied reaction {step.rxn_idx}"
                    step_result["reactants"] = current_reactants
                    step_result["products"] = products
                    step_result["selected_product"] = product

                    current_reactants = []
                
                elif step.step_type == StepType.END:
                    step_result["action"] = "Synthesis complete"
                
                step_results.append(step_result)
            
            return SynthesisResult(
                success=True,
                product=intermediates[-1],
                intermediates=intermediates,
                step_results=step_results
            )
        
        except Exception as e:
            return SynthesisResult(
                success=False,
                product=None,
                intermediates=[],
                error_message=str(e),
                step_results=step_results
            )
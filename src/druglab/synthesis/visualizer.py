from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import io

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor

from .route import SynthesisRoute, StepType
from .executor import SynthesisExecutor
from ..storage import RxnStorage, MolStorage

logger = logging.getLogger(__name__)

@dataclass
class VisualizationStep:
    """Represents a visualization step with molecules and metadata."""
    step_type: StepType
    step_number: int
    reactants: List[Chem.Mol]
    products: List[Chem.Mol]
    reaction_smarts: Optional[str] = None
    reaction_name: Optional[str] = None
    description: str = ""
    metadata: Dict = None

class SynthesisRouteVisualizer:
    """Visualize synthesis routes with molecule structures."""
    
    def __init__(self, 
                 rxn_storage: RxnStorage, 
                 mol_storage: MolStorage,
                 mol_size: Tuple[int, int] = (300, 200),
                 reaction_size: Tuple[int, int] = (400, 150),
                 dpi: int = 100):
        """
        Initialize the visualizer.
        
        Args:
            rxn_storage: Storage containing reactions
            mol_storage: Storage containing molecules
            mol_size: Size of molecule images (width, height)
            reaction_size: Size of reaction scheme images
            dpi: Resolution for images
        """
        self.rxn_storage = rxn_storage
        self.mol_storage = mol_storage
        self.mol_size = mol_size
        self.reaction_size = reaction_size
        self.dpi = dpi
        self.executor = SynthesisExecutor(rxn_storage, mol_storage)
        
        # Color scheme
        self.colors = {
            'reactant': '#E3F2FD',      # Light blue
            'intermediate': '#FFF3E0',   # Light orange
            'product': '#E8F5E8',       # Light green
            'reaction': '#F3E5F5',      # Light purple
            'arrow': '#424242',         # Dark gray
            'text': '#212121',          # Almost black
            'border': '#BDBDBD'         # Gray
        }
    
    def _mol_to_image(self, 
                      mol: Chem.Mol, 
                      size: Tuple[int, int] = None,
                      highlight_atoms: List[int] = None,
                      highlight_bonds: List[int] = None) -> Image.Image:
        """Convert RDKit molecule to PIL Image."""
        if mol is None:
            # Create blank image for None molecules
            size = size or self.mol_size
            img = Image.new('RGB', size, 'white')
            return img
        
        size = size or self.mol_size
        
        # Generate 2D coordinates if needed
        if not mol.GetNumConformers():
            rdDepictor.Compute2DCoords(mol)
        
        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.SetFontSize(0.8)
        
        # Set highlighting if provided
        if highlight_atoms or highlight_bonds:
            drawer.DrawMolecule(mol, 
                              highlightAtoms=highlight_atoms or [],
                              highlightBonds=highlight_bonds or [])
        else:
            drawer.DrawMolecule(mol)
        
        drawer.FinishDrawing()
        
        # Convert to PIL Image
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        
        return img
    
    def _create_reaction_image(self, 
                               reactants: List[Chem.Mol], 
                               products: List[Chem.Mol],
                               reaction_smarts: Optional[str] = None) -> Image.Image:
        """Create an image showing the reaction scheme."""
        # Calculate layout
        n_reactants = len(reactants)
        n_products = len(products)
        mol_width, mol_height = self.mol_size
        arrow_width = 80
        spacing = 20
        
        # Total width calculation
        total_width = (max(n_reactants, n_products) * mol_width + 
                       (max(n_reactants, n_products) - 1) * spacing + 
                       arrow_width + 2 * spacing)
        total_height = mol_height + 100  # Extra space for labels
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(total_width/self.dpi, total_height/self.dpi), 
                               dpi=self.dpi)
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, total_height)
        ax.axis('off')
        
        # Draw reactants
        reactant_x_start = spacing
        reactant_y = (total_height - mol_height) // 2
        
        for i, reactant in enumerate(reactants):
            x_pos = reactant_x_start + i * (mol_width + spacing)
            
            # Get molecule image
            mol_img = self._mol_to_image(reactant)
            
            # Add to plot
            imagebox = OffsetImage(mol_img, zoom=1.0)
            ab = AnnotationBbox(imagebox, (x_pos + mol_width//2, reactant_y + mol_height//2), 
                                frameon=True, pad=0.1)
            ax.add_artist(ab)
            
            # Add "+" between reactants
            if i < len(reactants) - 1:
                ax.text(x_pos + mol_width + spacing//2, reactant_y + mol_height//2, '+', 
                        ha='center', va='center', fontsize=20, weight='bold')
        
        # Draw arrow
        arrow_x = reactant_x_start + n_reactants * mol_width + (n_reactants - 1) * spacing + spacing
        arrow_y = reactant_y + mol_height // 2
        
        arrow = FancyArrowPatch((arrow_x, arrow_y), 
                                (arrow_x + arrow_width, arrow_y),
                                arrowstyle='->', 
                                mutation_scale=20,
                                color=self.colors['arrow'],
                                linewidth=3)
        ax.add_patch(arrow)
        
        # Add reaction conditions text if available
        if reaction_smarts:
            ax.text(arrow_x + arrow_width//2, arrow_y + 30, 'Reaction', 
                    ha='center', va='center', fontsize=10, style='italic')
        
        # Draw products
        product_x_start = arrow_x + arrow_width + spacing
        
        for i, product in enumerate(products):
            x_pos = product_x_start + i * (mol_width + spacing)
            
            # Get molecule image
            mol_img = self._mol_to_image(product)
            
            # Add to plot
            imagebox = OffsetImage(mol_img, zoom=1.0)
            ab = AnnotationBbox(imagebox, (x_pos + mol_width//2, reactant_y + mol_height//2), 
                                frameon=True, pad=0.1)
            ax.add_artist(ab)
            
            # Add "+" between products
            if i < len(products) - 1:
                ax.text(x_pos + mol_width + spacing//2, reactant_y + mol_height//2, '+', 
                        ha='center', va='center', fontsize=20, weight='bold')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        return img
    
    def _prepare_visualization_steps(self, 
                                     route: SynthesisRoute,
                                     intermediates: Optional[List[Chem.Mol]] = None) -> List[VisualizationStep]:
        """Prepare visualization steps from synthesis route."""
        # Execute route if intermediates not provided
        if intermediates is None:
            result = self.executor.execute_route(route)
            if not result.success:
                raise ValueError(f"Route execution failed: {result.error_message}")
            intermediates = result.intermediates
            _ = result.step_results
        else:
            # Create dummy step results for provided intermediates
            _ = [{"step_type": step.step_type.value} for step in route.steps]
        
        viz_steps = []
        step_number = 0
        current_reactants = []
        intermediate_idx = 0
        
        for i, step in enumerate(route.steps):
            if step.step_type == StepType.START:
                viz_steps.append(VisualizationStep(
                    step_type=step.step_type,
                    step_number=0,
                    reactants=[],
                    products=[],
                    description="Starting synthesis route"
                ))
            
            elif step.step_type == StepType.ADD_REACTANT:
                mol = self.mol_storage.molecules[step.mol_idx]
                current_reactants.append(mol)
                viz_steps.append(VisualizationStep(
                    step_type=step.step_type,
                    step_number=step_number,
                    reactants=[mol],
                    products=[],
                    description=f"Add reactant (mol_idx: {step.mol_idx})"
                ))
            
            elif step.step_type == StepType.USE_INTERMEDIATE:
                if step.mol_idx is not None and step.mol_idx < len(intermediates):
                    mol = intermediates[step.mol_idx]
                elif intermediate_idx > 0:
                    mol = intermediates[intermediate_idx - 1]
                else:
                    raise ValueError("No intermediate available to use")
                
                current_reactants.append(mol)
                viz_steps.append(VisualizationStep(
                    step_type=step.step_type,
                    step_number=step_number,
                    reactants=[mol],
                    products=[],
                    description=f"Use intermediate (mol_idx: {step.mol_idx})"
                ))
            
            elif step.step_type == StepType.APPLY_REACTION:
                step_number += 1
                
                # Get reaction info
                rxn = self.rxn_storage.reactions[step.rxn_idx]
                reaction_smarts = rxn.GetSmarts() if hasattr(rxn, 'GetSmarts') else None
                
                # Get product
                if intermediate_idx < len(intermediates):
                    product = intermediates[intermediate_idx]
                    intermediate_idx += 1
                else:
                    raise ValueError(f"No intermediate available for step {step_number}")
                
                viz_steps.append(VisualizationStep(
                    step_type=step.step_type,
                    step_number=step_number,
                    reactants=current_reactants.copy(),
                    products=[product],
                    reaction_smarts=reaction_smarts,
                    description=f"Apply reaction {step.rxn_idx} → Product {step.product_idx}"
                ))
                
                current_reactants = []
            
            elif step.step_type == StepType.END:
                viz_steps.append(VisualizationStep(
                    step_type=step.step_type,
                    step_number=step_number,
                    reactants=[],
                    products=[],
                    description="Synthesis complete"
                ))
        
        return viz_steps
    
    def visualize_route(self, 
                        route: SynthesisRoute,
                        intermediates: Optional[List[Chem.Mol]] = None,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None,
                        show_molecule_names: bool = True,
                        vertical_layout: bool = True) -> plt.Figure:
        """
        Create a complete visualization of the synthesis route.
        
        Args:
            route: SynthesisRoute to visualize
            intermediates: Optional list of intermediate molecules
            title: Title for the visualization
            save_path: Path to save the figure
            show_molecule_names: Whether to show molecule names/indices
            vertical_layout: Whether to use vertical layout (vs horizontal)
            
        Returns:
            matplotlib Figure object
        """
        # Prepare visualization steps
        viz_steps = self._prepare_visualization_steps(route, intermediates)
        
        # Filter to only reaction steps for main visualization
        reaction_steps = [step for step in viz_steps 
                         if step.step_type == StepType.APPLY_REACTION]
        
        if not reaction_steps:
            raise ValueError("No reaction steps found in route")
        
        # Calculate layout
        n_steps = len(reaction_steps)
        
        if vertical_layout:
            fig_width = 12
            fig_height = max(8, n_steps * 4)
            fig, axes = plt.subplots(n_steps, 1, figsize=(fig_width, fig_height))
        else:
            fig_width = max(12, n_steps * 8)
            fig_height = 6
            fig, axes = plt.subplots(1, n_steps, figsize=(fig_width, fig_height))
        
        if n_steps == 1:
            axes = [axes]
        
        # Create each reaction step
        for i, (step, ax) in enumerate(zip(reaction_steps, axes)):
            self._draw_reaction_step(ax, step, step_number=i+1, 
                                   show_molecule_names=show_molecule_names)
        
        # Add overall title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        else:
            fig.suptitle(f'Synthesis Route ({len(reaction_steps)} steps)', 
                        fontsize=16, fontweight='bold', y=0.95)
        
        # Add route summary
        self._add_route_summary(fig, route, reaction_steps)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"Visualization saved to {save_path}")
        
        return fig
    
    def _draw_reaction_step(self, 
                           ax: plt.Axes, 
                           step: VisualizationStep,
                           step_number: int,
                           show_molecule_names: bool = True):
        """Draw a single reaction step."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        
        # Step title
        ax.text(5, 5.5, f'Step {step_number}', ha='center', va='center', 
                fontsize=14, fontweight='bold', color=self.colors['text'])
        ax.text(5, 5.1, step.description, ha='center', va='center', 
                fontsize=10, style='italic', color=self.colors['text'])
        
        # Draw reactants
        n_reactants = len(step.reactants)
        if n_reactants > 0:
            reactant_width = 3.0 / max(n_reactants, 1)
            for i, reactant in enumerate(step.reactants):
                x_pos = 1.5 + i * reactant_width
                self._draw_molecule_box(ax, reactant, x_pos, 3, 
                                      width=reactant_width*0.8, height=1.5,
                                      mol_type='reactant', 
                                      label=f'R{i+1}' if show_molecule_names else None)
                
                # Add "+" between reactants
                if i < n_reactants - 1:
                    ax.text(x_pos + reactant_width/2, 3, '+', ha='center', va='center', 
                           fontsize=16, fontweight='bold', color=self.colors['text'])
        
        # Draw arrow
        arrow = FancyArrowPatch((4.5, 3), (5.5, 3),
                                arrowstyle='->', mutation_scale=20,
                                color=self.colors['arrow'], linewidth=2)
        ax.add_patch(arrow)
        
        # Draw products
        n_products = len(step.products)
        if n_products > 0:
            product_width = 3.0 / max(n_products, 1)
            for i, product in enumerate(step.products):
                x_pos = 6.5 + i * product_width
                mol_type = 'product' if step.step_number == max([s.step_number for s in [step]]) else 'intermediate'
                self._draw_molecule_box(ax, product, x_pos, 3,
                                      width=product_width*0.8, height=1.5,
                                      mol_type=mol_type,
                                      label=f'P{i+1}' if show_molecule_names else None)
                
                # Add "+" between products
                if i < n_products - 1:
                    ax.text(x_pos + product_width/2, 3, '+', ha='center', va='center', 
                           fontsize=16, fontweight='bold', color=self.colors['text'])
    
    def _draw_molecule_box(self, 
                          ax: plt.Axes, 
                          mol: Chem.Mol, 
                          x: float, 
                          y: float,
                          width: float = 1.5, 
                          height: float = 1.5,
                          mol_type: str = 'intermediate',
                          label: Optional[str] = None):
        """Draw a molecule in a colored box."""
        # Get molecule image
        mol_img = self._mol_to_image(mol, size=(int(width*100), int(height*100)))
        
        # Create background box
        box_color = self.colors.get(mol_type, self.colors['intermediate'])
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                             boxstyle="round,pad=0.05",
                             facecolor=box_color,
                             edgecolor=self.colors['border'],
                             linewidth=1)
        ax.add_patch(box)
        
        # Add molecule image
        imagebox = OffsetImage(mol_img, zoom=0.8)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)
        
        # Add label if provided
        if label:
            ax.text(x, y - height/2 - 0.2, label, ha='center', va='top', 
                   fontsize=8, fontweight='bold', color=self.colors['text'])
        
        # Add SMILES as tooltip (in metadata)
        if mol:
            try:
                smiles = Chem.MolToSmiles(mol)
                ax.text(x, y + height/2 + 0.1, smiles[:20] + ('...' if len(smiles) > 20 else ''), 
                       ha='center', va='bottom', fontsize=6, 
                       color=self.colors['text'], alpha=0.7)
            except Exception:
                pass
    
    def _add_route_summary(self, 
                          fig: plt.Figure, 
                          route: SynthesisRoute, 
                          reaction_steps: List[VisualizationStep]):
        """Add summary information to the figure."""
        summary_text = [
            f"Route ID: {route.route_id or 'N/A'}",
            f"Total Steps: {len(reaction_steps)}",
            f"Reactants Used: {len(route.get_reactant_indices())}",
            f"Reactions Used: {len(route.get_reaction_indices())}"
        ]
        
        fig.text(0.02, 0.02, '\n'.join(summary_text), 
                fontsize=10, va='bottom', ha='left',
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor=self.colors['reaction'], 
                         alpha=0.7))
    
    def create_molecule_grid(self, 
                            molecules: List[Chem.Mol],
                            labels: Optional[List[str]] = None,
                            title: str = "Molecules",
                            cols: int = 4) -> plt.Figure:
        """Create a grid visualization of molecules."""
        n_mols = len(molecules)
        rows = (n_mols + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, mol in enumerate(molecules):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            if mol:
                # Get molecule image
                mol_img = self._mol_to_image(mol)
                ax.imshow(mol_img)
                
                # Add label
                if labels and i < len(labels):
                    ax.set_title(labels[i], fontsize=10)
                
                # Add SMILES
                try:
                    smiles = Chem.MolToSmiles(mol)
                    ax.text(0.5, -0.1, smiles[:30] + ('...' if len(smiles) > 30 else ''), 
                           transform=ax.transAxes, ha='center', va='top', 
                           fontsize=8, wrap=True)
                except Exception:
                    pass
            else:
                ax.text(0.5, 0.5, 'None', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
            
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(n_mols, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig


# Example usage function
def visualize_synthesis_route(route: SynthesisRoute,
                             rxn_storage: RxnStorage,
                             mol_storage: MolStorage,
                             intermediates: Optional[List[Chem.Mol]] = None,
                             title: Optional[str] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Convenience function to visualize a synthesis route.
    
    Args:
        route: SynthesisRoute to visualize
        rxn_storage: Reaction storage
        mol_storage: Molecule storage  
        intermediates: Optional list of intermediate molecules
        title: Optional title for the visualization
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    visualizer = SynthesisRouteVisualizer(rxn_storage, mol_storage)
    return visualizer.visualize_route(route, intermediates, title, save_path)
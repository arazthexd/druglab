from __future__ import annotations
from typing import List, Optional, Tuple
import logging
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

from .route import SynthesisRoute, StepType
from ..storage import RxnStorage, MolStorage

logger = logging.getLogger(__name__)

class SynthesisRouteVisualizer:
    """Visualizes synthesis routes using RDKit and matplotlib."""
    
    def __init__(self, 
                 route: SynthesisRoute,
                 rxn_storage: RxnStorage,
                 mol_storage: MolStorage,
                 intermediates: Optional[List[Chem.Mol]] = None):
        self.route = route
        self.rxn_storage = rxn_storage
        self.mol_storage = mol_storage
        self.intermediates = intermediates
        self.step_images = []
        self.step_labels = []

    def execute_route(self) -> List[Chem.Mol]:
        """Execute route to get intermediates if not provided."""
        from .executor import SynthesisExecutor  # Avoid circular import
        executor = SynthesisExecutor(self.rxn_storage, self.mol_storage)
        result = executor.execute_route(self.route)
        if not result.success:
            raise RuntimeError(f"Route execution failed: {result.error_message}")
        return result.intermediates

    def get_molecule_image(self, 
                           mol: Chem.Mol, 
                           size: Tuple[int, int] = (400, 400)) -> Image.Image:
        """Generate 2D image for a molecule."""
        if mol is None:
            return Image.new('RGB', size, (255, 255, 255))
        return Draw.MolToImage(mol, size=size, kekulize=True)
    
    def create_reaction_image(self, 
                              reactants: List[Chem.Mol], 
                              product: Chem.Mol) -> Image.Image:
        """Create combined image for reaction step showing reactants -> product."""
        # Generate images for all components
        reactant_imgs = [self.get_molecule_image(mol) for mol in reactants]
        product_img = self.get_molecule_image(product)
        
        # Create arrow image
        arrow_img = Image.new('RGB', (100, 100), (255, 255, 255))
        draw = ImageDraw.Draw(arrow_img)
        try:
            # Try to use a nice arrow character if font supports it
            font = ImageFont.truetype("arial.ttf", 20)
            draw.text((25, 25), "→", fill=(0, 0, 0), font=font)
        except Exception:
            # Fallback to simple arrow
            font = ImageFont.load_default(20)
            draw.text((40, 40), ">>", fill=(0, 0, 0), font=font)
        
        # Calculate dimensions for combined image
        widths = [img.width for img in reactant_imgs] + \
            [arrow_img.width, product_img.width]
        height = max(img.height for img in reactant_imgs + \
                     [arrow_img, product_img])
        total_width = sum(widths)
        
        # Create combined image
        combined = Image.new('RGB', (total_width, height), (255, 255, 255))
        x_offset = 0
        for img in reactant_imgs:
            combined.paste(img, (x_offset, (height - img.height) // 2))
            x_offset += img.width
        combined.paste(arrow_img, 
                       (x_offset, (height - arrow_img.height) // 2))
        x_offset += arrow_img.width
        combined.paste(product_img, 
                       (x_offset, (height - product_img.height) // 2))
        
        return combined

    def generate_visualization(self) -> None:
        """Parse route steps and generate visualization components."""
        if self.intermediates is None:
            intermediates = self.execute_route()
        else:
            intermediates = []
            for step in self.route.steps:
                if step.step_type == StepType.APPLY_REACTION:
                    intermediates.append(self.intermediates[step.mol_idx])
        
        current_reactants = []
        current_intermediates = []
        reaction_counter = 0
        
        for step in self.route.steps:
            if step.step_type == StepType.ADD_REACTANT:
                mol = self.mol_storage.molecules[step.mol_idx]
                current_reactants.append(mol)
                img = self.get_molecule_image(mol)
            
            elif step.step_type == StepType.USE_INTERMEDIATE:
                mol = current_intermediates.pop()
                current_reactants.append(mol)
                img = self.get_molecule_image(mol)
            
            elif step.step_type == StepType.APPLY_REACTION:
                # Get product from intermediates using reaction counter
                product = intermediates.pop(0)
                current_intermediates.append(product)
                reaction_counter += 1
                
                # Create reaction visualization
                img = self.create_reaction_image(current_reactants, product)
                self.step_images.append(img)
                self.step_labels.append(
                    f"Apply reaction #{step.rxn_idx} "
                    f"→ Product #{step.product_idx}"
                )
                current_reactants = []  # Reset reactants after reaction

    def draw_route(self, 
                   figsize: Tuple[int, int] = (12, 8), 
                   dpi: int = 100) -> plt.Figure:
        """Render the complete route visualization using matplotlib."""
        self.generate_visualization()
        
        n_steps = len(self.step_images)
        fig, axes = plt.subplots(n_steps, 1, figsize=figsize, dpi=dpi)
        if n_steps == 1:
            axes = [axes]  # Ensure axes is always a list
        
        for ax, img, label in zip(axes, self.step_images, self.step_labels):
            # Convert PIL image to numpy array for matplotlib
            ax.imshow(np.array(img))
            ax.set_title(label, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout(pad=1.0)
        return fig

    def save_visualization(self, filename: str, **kwargs) -> None:
        """Generate and save visualization to file."""
        fig = self.draw_route(**kwargs)
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
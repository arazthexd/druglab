# DrugLab

DrugLab is a Python package providing common utilities for drug discovery and cheminformatics workflows. It centralizes reusable components for file handling, molecular dataset management, and reproducible data processing pipelines. It originates from my need for reusable tools when hopping from one project to another :D

## Installation

Install directly from the source repository:

```bash
git clone https://github.com/arazthexd/druglab.git
cd druglab
pip install .
```

For development and testing:
```bash
pip install -e .[dev]
```

Note: `rdkit` is required for core cheminformatics operations and is recommended to be installed via conda (`conda install -c conda-forge rdkit`) depending on your environment.

## Architecture & Submodules
DrugLab is divided into three primary submodules that integrate to form standard workflows. The plan is simple: Whenever a new common group of utilities are needed for currently ongoing projects, new submodules can be added.

### 1. `druglab.io`
Provides file reading and writing utilities for standard chemical formats (SDF, CSV, SMILES, RXN, MOL).
* **`BatchReader`**: Processes files in defined chunks, allowing iteration over datasets that exceed available system memory.
* **`EagerReader`**: Loads all records into memory for immediate access.
* **Format Handlers**: Exposes `read_file()` and `write_file()` for format-agnostic operations based on file extensions.

### 2. `druglab.db`
Implements database-like table structures for managing groups of chemical entities. The core `BaseTable` class enforces a strict alignment across four properties:
* `objects`: A list of the parsed chemical instances (e.g., RDKit `Mol` or `ChemicalReaction` objects).
* `metadata`: A tabular structure containing scalar properties and assay data.
* `features`: A dictionary mapping string identifiers to numerical arrays (e.g., molecular fingerprints).
* `history`: An internal audit log that records pipeline transformations applied to the table.

Concrete implementations include `MoleculeTable`, `ConformerTable`, and `ReactionTable`.

### 3. `druglab.pipe`
A framework for constructing sequential processing pipelines to update, filter, and featurize tables.
* **Blocks**: Operations are defined as blocks (`BaseFeaturizer`, `BaseFilter`, `BasePreparation`, `IOBlock`).
* **Execution**: Supports multiprocessing (`n_workers`) and item-level caching to prevent redundant calculations on identical inputs.
* **Batch Orchestration**: Pipelines automatically handle batch boundaries. If a specific block requires chunked processing, the pipeline will partition the table, process the blocks, and concatenate the results downstream.

## Usage Examples

### Table Initialization and Operations

```python
from druglab.db import MoleculeTable

# Initialize a table from an external file
table = MoleculeTable.from_file("compounds.sdf")

# Compute RDKit descriptors
table.add_rdkit_descriptors()

# Filter the table based on metadata values
filtered_table = table.filter_by_metadata("MolWt < 500")

# Save the table state (objects, metadata, features, and history)
filtered_table.save("./processed_data")
```

### Pipeline Execution

```python
from druglab.db import MoleculeTable
from druglab.pipe import Pipeline
from druglab.pipe.blocks import MorganFeaturizer, MWFilter, KekulizePreparation

# Initialize data
table = MoleculeTable.from_smiles(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])

# Construct a processing pipeline
pipeline = Pipeline(steps=[
    KekulizePreparation(),
    MWFilter(max_mw=150.0),
    MorganFeaturizer(radius=2, n_bits=1024, n_workers=4, use_cache=True) 
])

# Execute the pipeline
processed_table = pipeline.run(table)

# Access results
print(f"Remaining molecules: {processed_table.n}")
print(f"Feature shape: {processed_table.features['MorganFeaturizer_radius=2_n_bits=1024'].shape}")
```

## License
MIT License.

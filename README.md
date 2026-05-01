# DrugLab

DrugLab is a Python package providing common utilities for drug discovery and cheminformatics workflows. It centralizes reusable components for file handling, molecular dataset management, and reproducible data processing pipelines. It originates from my need for reusable tools when hopping from one project to another :D

## Installation

Install directly from the source repository:

```terminal
git clone https://github.com/arazthexd/druglab.git
cd druglab
pip install .
```

For development and testing:
```terminal
pip install -e .[dev]
```

Note: `rdkit` is required for core cheminformatics operations and is recommended to be installed via conda (`conda install -c conda-forge rdkit`) depending on your environment.

## Architecture & Submodules
DrugLab is divided into primary submodules that integrate to form standard workflows. The plan is simple: Whenever a new common group of utilities are needed for currently ongoing projects, new submodules can be added.

### 1. `druglab.io`
Provides file reading and writing utilities for standard chemical formats. Its purpose is to ease the pain of working with various datasets, data formats, and merging of multiple files with various formats. Additionally, it is intended to have utilities for working with larger-than-memory databases by batch loading them into memory.
* **Format Handlers**: Exposes `read_file()` and `write_file()` for format-agnostic operations based on file extensions and potentially content.
* **`EagerReader`**: Loads all records into memory for immediate access.
* **`BatchReader`**: Processes files in defined chunks, allowing iteration over datasets that exceed available system memory.

### 2. `druglab.db`
Implements database-like table structures for managing groups of chemical entities. The main motivation behind it is the difficulties regarding working with multiple types of common data in cheminformatics pipelines. The core `BaseTable` is mainly consisted of `objects`, `metadata`, and `features`, along with a `history` for recording logs of past transformations/modifications on that table.

* `objects`: A list of the parsed chemical instances (e.g., RDKit `Mol` or `ChemicalReaction` objects).
* `metadata`: A tabular structure containing scalar properties and assay data.
* `features`: A dictionary mapping string identifiers to numerical arrays (e.g., molecular fingerprints).
* `history`: An internal audit log that records pipeline transformations applied to the table.

Concrete implementations include `MoleculeTable`, `ConformerTable`, and `ReactionTable`.

### 3. `druglab.db.backend`
In addition, behind the `BaseTable` class, is a `BaseStorageBackend` class implemented which is supposed to enable more types of storage, such as storing parts/all of the table components on-disk and enable on-disk operations. This makes `BaseTable` be operable independent of how/where data is stored and can be useful in larger-than-memory data situations.

A special implementation of the `BaseStorageBackend` class is the `OverlayBackend` class which is enables working with databases (more specifically, on-disk databases) without changing them immediately. It exposes a `.commit()` method for flushing those changes into the main database whenever intended. This helps working with such databases without worrying about the underlying database being corrupted.


### 3. `druglab.pipe`
A framework for constructing sequential processing pipelines to update, filter, and featurize tables.
* **Blocks**: Operations are defined as blocks (`BaseFeaturizer`, `BaseFilter`, `BasePreparation`, `IOBlock`).
* **Execution**: Supports multiprocessing (`n_workers`) and item-level caching to prevent redundant calculations on identical inputs.
* **Batch Orchestration**: Pipelines automatically handle batch boundaries. If a specific block requires chunked processing, the pipeline will partition the table, process the blocks, and concatenate the results downstream.

## License
MIT License.

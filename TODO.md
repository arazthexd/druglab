# DrugLab - Project Tasks & Roadmap

This file uses a priority-based Kanban-style structure to track upcoming features, bug fixes, and technical debt for DrugLab.

## 📌 Template Usage Guide
* **[🔴 High Priority]**: Critical bugs or essential features blocking active projects.
* **[🟡 Medium Priority]**: Enhancements that improve quality of life, performance, or add valuable tools.
* **[🟢 Low Priority]**: Nice-to-haves, refactoring, or exploratory ideas.
* **Status Flags**: Use `[ ]` for To-Do, `[~]` for In Progress, and `[x]` for Done.

---

## 🏃 In Progress
...

---

## 🚀 Backlog

### `druglab.io`
- [ ] **🟡 Support Compressed Files:** Support for `.sdf.gz`, `.csv.gz`, and `.smi.gz` directly in `BatchReader` and `EagerReader`.

### `druglab.db`
- [ ] **🔴 Improved Table Saving/Loading Schemes:** Create QoL improvements to the table storage schemes such as saving all objects in one file (currently each is saved separately in a directory).
- [ ] **🟡 External Database Bridging:** Add capabilities to dump/load `metadata` and `features` directly to/from SQL databases (SQLite/PostgreSQL) instead of just local directories.
- [ ] **🟡 3D Conformer Table:** Create a `ConformerTable` (subclass of `MoleculeTable` or new) specifically optimized for handling multi-conformer ensembles and 3D coordinates.
- [ ] **🟢 Memory-Mapped Optimization:** Automatically switch to `mmap` for `BaseTable.features` on table creation/modification if feature arrays exceed a specific RAM threshold.

### `druglab.pipe`
- [ ] **🔴 List Out-of-the-Box Blocks:** Create a list of important common pipeline blocks that need implementation and add them as TODO records.
- [ ] **🟢 Pipeline Serialization:** Add `pipeline.save("pipe.json")` to save a pipeline's configuration, and `Pipeline.load("pipe.json")` to rebuild it. Combine this with the `HistoryEntry` to strictly reproduce a dataset.

### Project Infrastructure & Docs
- [ ] **🟢 Unify Documentation:** Fix some differences in documentation due to differing project origins of various submodules.

---

## 🏃 In Progress
...

---

## ✅ Completed
...
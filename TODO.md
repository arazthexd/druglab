# TODOs

## Pharm
### Next...
- [ ] Develop storage class for pharmacophores
- [ ] Allow pharmacophore and profile classes to include multiple confs
- [ ] Define weights for confs and use them for mol vs mol similarities
- [ ] Utility function for profiles and pharms:
    - Mols as input, clustered confs using TFD (faster) and Butina
    - Determine pharms and profiles
    - Output pharmacophore storage instance

### Ideas...
- [ ] Profile distributions by sampling multiple conformers
- [ ] Optimization of conformers to match profiles
- [ ] Position profile to perform similarity search directly on single feats

### Done.
- [x] Clean up the pharm modules (remove unwanted methods, classes) - reworked.

## Featurize
### Next...
- [ ] No saving for featurizers... (pickling problems)
- [ ] PCA, and MCA featurizers (+ in composite featurizers)

### Ideas...
- [ ] Featurizer wrapper for molfeat package's Transformer class
- [ ] Atom, Bond Featurizers

### Done.

## Storage
### Next...

### Ideas...
- [ ] Atom, Bond Storage and searching
- [ ] Filter method

### Done.

## Synthesize
### Next...
- [ ] Optimization Algorithms (RL?) for score-based sampling

### Ideas...

### Done.
- [x] Fix indexing bug for synthesis route storage

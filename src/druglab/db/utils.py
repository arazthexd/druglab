from pathlib import Path
import pickle as pkl

def object_pkl_writer(objs, path: Path):
    with open(path / "objects.pkl", "wb") as f:
        pkl.dump(objs, f)

def object_pkl_reader(path: Path):
    with open(path / "objects.pkl", "rb") as f:
        return pkl.load(f)
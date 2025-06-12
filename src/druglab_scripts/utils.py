import tempfile
from pathlib import Path
import h5py

def _create_out_file(out_path: str | Path, overwrite: bool):
    if isinstance(out_path, str):
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and out_path.exists():
        raise FileExistsError(f"Output file {out_path} already exists.")
    if overwrite and out_path.exists():
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            out_path = Path(f.name)
    return out_path

def _get_database_len(path: Path, key: str):
    with h5py.File(path, 'r') as f:
        dbl = f[key].shape[0]
    return dbl
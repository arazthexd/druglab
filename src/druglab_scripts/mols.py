from typing import Tuple, Optional, List, Iterator
from pathlib import Path
import click
import shutil

from druglab.storage import (
    MolStorage, GenericMoleculePrepper,
    BasicMoleculeFeaturizerWrapper, BasicConformerFeaturizerWrapper
)
from druglab.featurize import (
    MOLFEATURIZER_GENERATORS, CONFFEATURIZER_GENERATORS,
    get_featurizer
)
from druglab.io import load_mols_file

from druglab_scripts.utils import _get_database_len, _create_out_file

MFNAMES = list(MOLFEATURIZER_GENERATORS.keys())
CFNAMES = list(CONFFEATURIZER_GENERATORS.keys())

def _iterate_moldb_batches(path: str | Path, 
                           batch_size: int) -> Iterator[MolStorage]:
    path = Path(path)
    batchid = 0
    for i in range(0, _get_database_len(path, 'molecules'), batch_size):
        db_batch = MolStorage.load(path, indices=slice(i, i+batch_size))
        click.echo(f"Batch {batchid} with size {len(db_batch)} was loaded.")
        batchid += 1
        yield db_batch

@click.group(name='molecules', help="Operations related to molecules")
def molecule_operations():
    pass

@molecule_operations.command(name='create-db',
                             help="Create a database from a set of files")
@click.option('-i', '--inputs', multiple=True, 
              type=click.Path(exists=True), required=True,
              help='Input files (sdf and smi are currently supported.)')
@click.option('-o', '--output', type=click.Path(), default='mols.h5',
              help="Output database file (.h5)")
@click.option('-ow', '--overwrite', is_flag=True, default=False,
              help="Overwrite existing file")
@click.option('-sd', '--smi-delimiter', type=str, default='')
def create_mols_db(inputs: Tuple[str, ...], 
              output: str,
              overwrite: bool,
              smi_delimiter: str):
    if smi_delimiter == "TAB":
        smi_delimiter = "\t"

    output: Path = Path(output).absolute()
    if overwrite and output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    click.echo("Creating a molecule database")
    for filename in inputs:
        click.echo(f"Reading file: {Path(filename).name}")

        molecules = load_mols_file(filename, smi={'delimiter': smi_delimiter})
        molecules = MolStorage(molecules)

        click.echo(f"Writing {len(molecules)} loaded mols to database")
        molecules.write(output, mode='a')

@molecule_operations.command(name='clean',
                             help="Clean a molecule database")
@click.option('-db', '--database', type=click.Path(exists=True), 
              default='mols.h5',  help='Input database (.h5 file)')
@click.option('-o', '--output', type=click.Path(), default='mols.h5',
              help="Output database file (.h5)")
@click.option('-ow', '--overwrite', is_flag=True, default=False,
              help="Overwrite existing file")
@click.option('-rf', '--remove-failed', is_flag=True, default=False,
              help="Remove molecules with failed preparations")
@click.option('-rd', '--remove-duplicates', is_flag=True, default=False,
              help="Remove duplicate molecules using SMILES")
@click.option('-ns', '--no-sanitize', is_flag=True, default=False,
              help="Sanitize molecules in the database using RDKit")
@click.option('-rc', '--remove-conformers', is_flag=True, default=False,
              help="Remove all conformers of the molecules in the database")
@click.option('-b', '--batch-size', default=1000, type=int,
              help="Batch size (For memory purposes)")
def clean_mols(database: str,
               output: str,
               overwrite: bool,
               remove_failed: bool,
               remove_duplicates: bool,
               no_sanitize: bool,
               remove_conformers: bool,
               batch_size: int,):
    out_path = _create_out_file(output, overwrite)

    click.echo("Cleaning molecules from the database")
    for molecules in _iterate_moldb_batches(database, batch_size):
        keep_ids = molecules.clean_molecules(
            remove_none=remove_failed,
            remove_duplicates=remove_duplicates,
            sanitize=not no_sanitize,
            remove_conformers=remove_conformers
        )
        click.echo(f"{len(keep_ids)} molecules were kept.")
        molecules.write(out_path, mode='a')
    
    out_path.replace(output)

@molecule_operations.command(name='prepare',
                             help="Read, prepare, and save molecules from db")
@click.option('-db', '--database', type=click.Path(exists=True), 
              default='mols.h5',  help='Input database (.h5 file)')
@click.option('-o', '--output', type=click.Path(), default='prepared.h5',
              help="Output database file (.h5)")
@click.option('-ow', '--overwrite', is_flag=True, default=False,
              help="Overwrite existing file")
@click.option('-rf', '--remove-failed', is_flag=True, default=False,
              help="Remove molecules with failed preparations")
@click.option('-b', '--batch-size', default=1000, type=int,
              help="Batch size")
@click.option('-ns', '--no-salts', is_flag=True, default=False,
              help="Don't remove salts")
@click.option('-nl', '--no-largest-frag', is_flag=True, default=False,
              help="Don't remove smaller fragments")
@click.option('-nn', '--no-neutralize', is_flag=True, default=False,
              help="Don't neutralize charges")
@click.option('-nt', '--no-tautomers', is_flag=True, default=False,
              help="Don't standardize tautomers")
@click.option('-a', '--add-hs', is_flag=True, default=False,
              help="Add hydrogen atoms")
@click.option('-r', '--remove-hs', is_flag=True, default=False,
              help="Remove hydrogen atoms")
@click.option('-cg', '--cgen', is_flag=True, default=False,
              help="Generate conformers")
@click.option('-cn', '--cgen-n', default=1, type=int,
              help="Number of conformers to generate")
@click.option('-cma', '--cgen-maxatts', default=None, type=int,
              help="Maximum number of atoms in a conformer")
@click.option('-co', '--copt', is_flag=True, default=False,
              help="Optimize conformers")
@click.option('-con', '--copt-nthreads', default=1, type=int,
              help="Number of threads to use for conformer optimization")
@click.option('-coi', '--copt-maxits', default=200, type=int,
              help="Maximum number of iterations for conformer optimization")
@click.option('-cc', '--cclust', is_flag=True, default=False,
              help="Cluster conformers")
@click.option('-ct', '--cclust-tol', default=0.3, type=float,
              help="Tolerance for conformer clustering")
@click.option('-coo', '--cclust-afteropt', is_flag=True, default=False,
              help="Cluster conformers after optimization")
@click.option('-ca', '--calign', is_flag=True, default=False,
              help="Align conformers")
@click.option('-np', '--n-processes', default=1, type=int,
              help="Number of processes to use for multiprocessing")
def prepare_mols(database: str, 
                 output: str, 
                 overwrite: bool,
                 remove_failed: bool,
                 batch_size: int,
                 no_salts: bool,
                 no_largest_frag: bool,
                 no_neutralize: bool,
                 no_tautomers: bool,
                 add_hs: bool,
                 remove_hs: bool,
                 cgen: bool,
                 cgen_n: int,
                 cgen_maxatts: Optional[int],
                 copt: bool,
                 copt_nthreads: int,
                 copt_maxits: int,
                 cclust: bool,
                 cclust_tol: float,
                 cclust_afteropt: bool,
                 calign: bool,
                 n_processes: int):
    
    prepper = GenericMoleculePrepper(
        remove_salts=not no_salts,
        keep_largest_frag=not no_largest_frag,
        neutralize=not no_neutralize,
        standardize_tautomers=not no_tautomers,
        addhs=add_hs,
        removehs=remove_hs,
        cgen=cgen,
        cgen_n=cgen_n,
        cgen_maxatts=cgen_maxatts,
        cgen_parambase=None,
        copt=copt,
        copt_nthreads=copt_nthreads,
        copt_maxits=copt_maxits,
        cclust=cclust,
        cclust_tol=cclust_tol,
        cclust_afteropt=cclust_afteropt,
        calign=calign,
        n_processes=n_processes,
    )

    out_path = _create_out_file(output, overwrite)
    
    for molecules in _iterate_moldb_batches(database, batch_size):
        molecules = prepper.modify(molecules, 
                                   in_place=True, 
                                   remove_fails=remove_failed)
        molecules.write(out_path, mode='a')
    
    out_path.replace(output)

@molecule_operations.command(name='featurize', 
                             help="Featurize molecules or their conformers")
@click.option('-db', '--database', type=click.Path(exists=True), 
              default='mols.h5',  help='Input database (.h5 file)')
@click.option('-o', '--output', type=click.Path(), default='prepared.h5',
              help="Output database file (.h5)")
@click.option('-ow', '--overwrite', is_flag=True, default=False,
              help="Overwrite existing file")
@click.option('-mf', '--mol-featurizer', multiple=True,
              type=click.Choice(MFNAMES),
              help="Featurizer for molecules")
@click.option('-cf', '--conf-featurizer', multiple=True,
              type=click.Choice(CFNAMES),
              help="Featurizer for conformers")
@click.option('-b', '--batch-size', default=1000, type=int,
              help="Batch size")
@click.option('-np', '--n-processes', default=1, type=int,
              help="Number of processes to use for multiprocessing")
def featurize_mols(database: str,
                   output: str,
                   overwrite: bool,
                   mol_featurizer: List[str],
                   conf_featurizer: List[str],
                   batch_size: int,
                   n_processes: int):

    mol_featurizers = [
        BasicMoleculeFeaturizerWrapper(get_featurizer(name), 
                                       n_processes=n_processes) 
        for name in mol_featurizer
    ]
    conf_featurizers = [
        BasicConformerFeaturizerWrapper(get_featurizer(name),
                                        n_processes=n_processes) 
        for name in conf_featurizer
    ]
    
    out_path = _create_out_file(output, overwrite)
    
    for molecules in _iterate_moldb_batches(database, batch_size):
        for mf in mol_featurizers:
            molecules = mf.featurize(molecules)
        for cf in conf_featurizers:
            molecules = cf.featurize(molecules)
        molecules.write(out_path, mode='a')
    
    out_path.replace(output)

# TODO: Filtering script
# TODO: Conversion between standard files and database script

if __name__ == "__main__":
    molecule_operations()
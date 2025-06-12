from typing import Tuple
from pathlib import Path
import click
import click_config_file
from tqdm import tqdm
import time
from functools import partial

from rdkit import Chem
from rdkit.Chem import QED
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

from druglab.storage import (
    MolStorage,
    RxnStorage,
    CustomFuncFilter
)
from druglab.io import load_rxns_file, load_mols_file
from druglab.synthesis import EfficientSynthesisRouteSampler
from druglab.synthesis.sampling.batched import SamplingConfig
from druglab.synthesis.storage import SynRouteStorage

from druglab_scripts.utils import _create_out_file

def is_mol_in_seen(mol, seen):
    return Chem.MolToSmiles(mol) not in seen

@click.group(name='reactions',
             help="Operations related to reactions and forward synthesis")
def reaction_operations():
    pass

@reaction_operations.command(name='create-db',
                             help=("Create a reaction database from reaction "
                                   "and molecule files"))
@click.option('-ri', '--reaction-inputs', multiple=True, 
              type=click.Path(exists=True), required=True,
              help='Input rxn files (only .txt, .h5 is currently supported)')
@click.option('-mi', '--molecule-inputs', multiple=True, 
              type=click.Path(exists=True), required=True,
              help=("Input mol files (only .sdf, .smi, .h5 is currently "
                    "supported)"))
@click.option('-ro', '--reaction-output', type=click.Path(), default='rxns.h5',
              help="Output reaction database file (.h5)")
@click.option('-mo', '--molecule-output', type=click.Path(), default='mols.h5',
              help="Output molecule database file (.h5)")
@click.option('-ow', '--overwrite', is_flag=True, default=False,
              help="Overwrite existing file")
@click.option('-rur', '--remove-unmatched-reactions', is_flag=True, 
              default=False, help="Remove reactions without reactants")
@click.option('-rum', '--remove-unmatched-molecules', is_flag=True,
              default=False, help="Remove molecules without reactions")
def create_rxn_db(reaction_inputs: Tuple[str, ...], 
                  molecule_inputs: Tuple[str, ...], 
                  reaction_output: str,
                  molecule_output: str,
                  overwrite: bool,
                  remove_unmatched_reactions: bool,
                  remove_unmatched_molecules: bool):
    orig_ro = reaction_output
    orig_mo = molecule_output
    reaction_output = _create_out_file(reaction_output, overwrite=True)
    if molecule_inputs:
        molecule_output = _create_out_file(molecule_output, overwrite=True)

    click.echo("Creating a reaction database")

    # TODO: Maybe not do this all on memory?
    molecules = [
        mol
        for filename in molecule_inputs
        for mol in load_mols_file(filename)
    ]
    molecules = MolStorage(molecules)
    molecules.clean_molecules()

    reactions = []
    for filename in reaction_inputs:
        click.echo(f"Reading file: {Path(filename).name}")
        new = load_rxns_file(filename)
        reactions.extend(new)
        click.echo(f"{len(new)} reactions loaded")

    reactions = RxnStorage(reactions)
    reactions.clean_reactions()

    if remove_unmatched_molecules:
        click.echo("Removing unmatched molecules")
        keep_ids = [
            i 
            for i, mol in enumerate(tqdm(molecules.molecules))
            if any(rxn.IsMoleculeReactant(mol) for rxn in reactions.reactions)
        ]
        molecules = molecules.subset(keep_ids)

    if len(molecules) > 0:
        click.echo("Matching molecules to reactions")
        reactions.match_molecules(molecules)

        if remove_unmatched_reactions:
            click.echo("Removing unmatched reactions")
            keep_ids = [
                i 
                for i, rxn in enumerate(reactions.reactions)
                if all(
                    len(reactions.get_molecules_for_reactant(molecules,i,j)) > 0
                    for j in range(rxn.GetNumReactantTemplates())
                )
            ]
            reactions = reactions.subset(keep_ids)

            if remove_unmatched_molecules:
                click.echo("Removing unmatched molecules")
                keep_ids = [
                    i 
                    for i, mol in enumerate(tqdm(molecules.molecules))
                    if any(rxn.IsMoleculeReactant(mol) for rxn in reactions.reactions)
                ]
                molecules = molecules.subset(keep_ids)
                reactions.subset_referenced_mols(keep_ids)

    click.echo(f"Writing {len(reactions)} reactions to database")
    reactions.write(reaction_output, mode='a')

    if len(molecules) > 0:
        click.echo(f"Writing {len(molecules)} molecules to database")
        molecules.write(molecule_output, mode='a')

    molecule_output.replace(orig_mo)
    reaction_output.replace(orig_ro)

@reaction_operations.command(name='sample',
                             help=("Sample products using reactions and "
                                   "reactants using a forward synthesis "
                                   "approach"))
@click.option('-rdb', '--reaction-database', 
              type=click.Path(exists=True), required=True,
              help='Input rxn database (.h5)')
@click.option('-mdb', '--molecule-database',
              type=click.Path(exists=True), required=True,
              help="Input mol database (.h5)")
@click.option('-tn', '--target-num', default=1000, type=int,
              help="Target number of products")
@click.option('-po', '--product-output', type=click.Path(), 
              default='products.h5',
              help="Output product/intermediate database file (.h5)")
@click.option('-ro', '--route-output', type=click.Path(), 
              default='routes.h5',
              help="Output synthesis routes database file (.h5)")
@click.option('-ow', '--overwrite', is_flag=True, default=False,
              help="Overwrite existing file")
@click.option('-fq', '--filter-qed', default=0.0, type=float,
              help="Filter products by QED score before saving to database")
@click_config_file.configuration_option()
def sample_products(reaction_database: str,
                    molecule_database: str,
                    target_num: int,
                    product_output: str,
                    route_output: str,
                    overwrite: bool,
                    filter_qed: float):
    
    blocker = rdBase.BlockLogs()
    import logging
    logging.basicConfig(level=logging.DEBUG, filename="log.txt", filemode="w")
    
    # WAITING TO GO TO OPTIONS:
    min_steps = 1
    max_steps = 5
    n_routes_per_template = 1000
    n_template_batches = 50
    allow_multi_prods = False
    random_seed = 2026
    n_processes = 14
    only_final = True
    
    orig_ro = route_output
    orig_po = product_output
    product_output = _create_out_file(product_output, overwrite)
    route_output = _create_out_file(route_output, overwrite)

    click.echo("Configuring synthesis route sampler")

    config = SamplingConfig(
        min_steps=min_steps,
        max_steps=max_steps,
        n_routes_per_template=n_routes_per_template,
        n_template_batches=n_template_batches,
        allow_multi_prods=allow_multi_prods,
        random_seed=random_seed
    )
    sampler = EfficientSynthesisRouteSampler(config)

    rxns = RxnStorage.load(reaction_database)
    mols = MolStorage.load(molecule_database)

    seen_prods = set()

    n_routes = 0
    n_products = 0
    tic = time.time()
    while n_routes < target_num:
        click.echo("Sampling synthesis routes")
        routes, products = sampler.sample_routes(
            rxn_storage=rxns,
            mol_storage=mols,
            n_processes=n_processes,
            only_final=only_final
        )
        products = MolStorage(products)
        routes = SynRouteStorage(routes)
        click.echo(f"Newly sampled routes: {len(routes)}")
        click.echo(f"Newly sampled products/intermediates: {len(products)}")

        # ==============

        click.echo("Cleaning products")
        keep_ids = products.clean_molecules(remove_none=True,
                                            sanitize=True)
        routes = routes.subset_by_component_ids(intermediate_ids=keep_ids)
        click.echo(f"Remaining sampled routes: {len(routes)}")
        click.echo(f"Remaining sampled products/intermediates: {len(products)}")

        # ==============

        # click.echo("Removing previously seen products")
        # storage_filter = CustomFuncFilter(
        #     partial(is_mol_in_seen, seen=seen_prods),
        #     input_keys=['molecules'],
        #     n_processes=n_processes,
        #     name='Seen SMILES Filter'
        # )
        # keep_ids = storage_filter.filter(products)
        # products = products.subset(keep_ids)
        # routes = routes.subset_by_component_ids(intermediate_ids=keep_ids)
        # seen_prods.update(
        #     [Chem.MolToSmiles(prod) for prod in products.molecules]
        # )
        # click.echo(f"Remaining sampled routes: {len(routes)}")
        # click.echo(f"Remaining sampled products/intermediates: {len(products)}")

        # ==============

        click.echo("Removing products not in final reactions")
        keep_ids = set()
        for i, route in enumerate(routes.routes):
            keep_ids.update(route.get_intermediate_indices())
        keep_ids = sorted(keep_ids)
        products = products.subset(keep_ids)
        routes = routes.subset_by_component_ids(intermediate_ids=keep_ids)
        click.echo(f"Remaining sampled routes: {len(routes)}")
        click.echo(f"Remaining sampled products/intermediates: {len(products)}")

        # ==============

        click.echo(f"Filtering products based on given config: "
                   f"QED>{filter_qed}")
        def task(mol: Chem.Mol):
            blocker = rdBase.BlockLogs()
            if filter_qed > 0.0:
                if QED.qed(mol) < filter_qed:
                    return False
            del blocker
            return True
        
        storage_filter = CustomFuncFilter(
            task,
            input_keys=['molecules'],
            n_processes=n_processes,
            name='QED Filter'
        )
        keep_ids = storage_filter.filter(products)
        products = products.subset(keep_ids)
        routes = routes.subset_by_component_ids(intermediate_ids=keep_ids)
        click.echo(f"Remaining sampled routes: {len(routes)}")
        click.echo(f"Remaining sampled products/intermediates: {len(products)}")

        # ==============

        click.echo("Reindexing route intermediate indices before saving")
        for i, route in enumerate(routes.routes):
            routes._objects['routes'][i] = SynRouteStorage._update_route_indices(
                route=route,
                product_index_mapping={
                    idx: idx + n_products
                    for idx in range(len(products))
                }
            )

        n_routes += len(routes)
        n_products += len(products)
        click.echo(f"Total number of routes: {n_routes}")
        click.echo(f"Total number of products: {n_products}")

        click.echo("Writing databases...")
        routes.write(route_output, mode='a')
        products.write(product_output, mode='a')

        click.echo("Done!")
        click.echo(f"Time elapsed: {time.time() - tic:.2f} seconds")

    route_output.replace(orig_ro)
    product_output.replace(orig_po)
    

    

import click

from druglab_scripts.mols import molecule_operations
from druglab_scripts.rxns import reaction_operations

@click.group(name='druglab')
def cli():
    pass

cli.add_command(molecule_operations)
cli.add_command(reaction_operations)

if __name__ == "__main__":
    cli()
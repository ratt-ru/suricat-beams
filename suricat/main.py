import click
import os.path
import logging
from omegaconf import OmegaConf

@click.group()
@click.option("-v", "--verbose", is_flag=True)
def cli(verbose=False):
    from . import log, set_console_logging_level
    log.info("Suricat is pleased you are minding your beams")
    if verbose:
        set_console_logging_level(logging.DEBUG)
        log.debug("Enabling extra verbose output")


schemas = OmegaConf.load(os.path.join(os.path.dirname(__file__), "cabs/suricat.yml"))

from . import beams

if __name__ == "__main__":
    cli()
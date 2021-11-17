"""
The Nuclear Receptor Signaling Atlas (NURSA) is no longer
available at https://www.nursa.org/nursa/index.jsf. The only
place I could find a copy of it was on the Harmonizome platform
at https://maayanlab.cloud/Harmonizome/dataset/NURSA+Protein-Protein+Interactions.

This script downloads and pre-processes it via a GMT (gene set file)
then saves it as JSON in this repositoryafter normalizing HGNC gene identifiers.
"""

import json
from pathlib import Path

import pystow
from indra.databases import hgnc_client
from pyobo.sources.gmt_utils import _process_line
from tqdm import tqdm

HERE = Path(__file__).parent.resolve()
PATH = HERE / "processed" / "nursa.json"
URL = "https://maayanlab.cloud/static/hdfs/harmonizome/data/nursappi/gene_set_library_crisp.gmt.gz"
MODULE = pystow.module("bio", "nursa")


def main():
    """Download and pre-process the NURSA dataset."""
    rv = {}
    with MODULE.ensure_open_gz(url=URL, mode="rt") as file:
        it = tqdm(file)
        for i, line in enumerate(it, start=1):
            try:
                source_symbol, ncbigene_id, target_symbols = _process_line(line)
            except ValueError:
                it.write(f"failed on line {i}")
                continue
            source_hgnc_id = hgnc_client.get_hgnc_from_entrez(ncbigene_id)
            if source_hgnc_id is None:
                continue
            target_hgnc_ids = {
                hgnc_client.get_hgnc_id(target_symbol) for target_symbol in target_symbols
            }
            target_hgnc_ids = sorted(x for x in target_hgnc_ids if x)
            if not target_hgnc_ids:
                continue
            rv[source_hgnc_id] = target_hgnc_ids
    with PATH.open("w") as file:
        json.dump(rv, file, indent=2)


if __name__ == "__main__":
    main()

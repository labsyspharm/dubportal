# -*- coding: utf-8 -*-

"""Run this script to prepare the differential gene expression analysis."""

import json
import pathlib
import zipfile

import click
import pandas as pd
import pyobo
from indra.databases import hgnc_client
from more_click import verbose_option

HERE = pathlib.Path(__file__).parent.resolve()
RAW = HERE.joinpath("raw")
PROCESSED = HERE.joinpath("processed")

INPUT_PATH = RAW.joinpath("results.zip")
OUTPUT_PATH = PROCESSED.joinpath("ko_dgea.json")
COLUMNS = ["hgnc_id", "hgnc_symbol", "hgnc_name", "log2FoldChange", "pvalue", "padj"]


@click.command()
@verbose_option
def main():
    rv = get_dgea()
    click.echo(f"got {len(rv)} entries")


def get_dgea():
    rv = {}
    hgnc_id_def = pyobo.get_id_definition_mapping("hgnc")

    with zipfile.ZipFile(INPUT_PATH) as zip_file:
        for zip_info in zip_file.filelist:
            if not zip_info.filename.startswith("results") or not zip_info.filename.endswith(
                ".txt"
            ):
                continue
            symbol = zip_info.filename.removeprefix("results/").removesuffix("_res_table.txt")
            gene_id = hgnc_client.get_current_hgnc_id(symbol)
            gene_symbol = hgnc_client.get_hgnc_name(gene_id)
            if gene_id is None:
                raise ValueError(f"invalid name: {zip_info.filename}")
            with zip_file.open(zip_info) as file:
                df = pd.read_csv(file, sep="\t")
                df = df[df["padj"] < 0.05]
                df["hgnc_id"] = df["gene"].map(_greedy_get)
                # FIXME manual gene mapping?
                df = df[df["hgnc_id"].notnull()]
                df["hgnc_symbol"] = df["hgnc_id"].map(hgnc_client.get_hgnc_name)
                df["hgnc_name"] = df["hgnc_id"].map(hgnc_id_def)
                rv[gene_symbol] = df[COLUMNS].to_dict("records")
    with OUTPUT_PATH.open("w") as file:
        json.dump(rv, file, indent=2, sort_keys=True)
    return rv


def _greedy_get(gene_symbol):
    gene_id = hgnc_client.get_current_hgnc_id(gene_symbol)
    if isinstance(gene_id, list):
        return gene_id[0]
    return gene_id


if __name__ == "__main__":
    main()

import json
import logging
import os
import pathlib
from operator import itemgetter
from typing import List, Optional

import click
import matplotlib.pyplot as plt
import pandas as pd
import pyobo
import requests
from jinja2 import Environment, FileSystemLoader
from matplotlib_venn import venn2
from more_click import force_option, verbose_option
from tqdm import tqdm

import famplex
from indra.databases import hgnc_client
from indra.statements import Statement, stmts_from_json

logger = logging.getLogger(__name__)
HERE = pathlib.Path(__file__).parent.resolve()
DOCS = HERE.parent.joinpath("docs")
DATA_DUR = HERE.joinpath("data")
DATA = DATA_DUR.joinpath("DUB_website_main_v2.tsv")
DATA_PROCESSED = DATA_DUR.joinpath("data.json")
NDEX_LINKS = DOCS.joinpath("network_index.json")

STATEMENTS_DIR = DATA_DUR.joinpath('statements')
STATEMENTS_DIR.mkdir(exist_ok=True, parents=True)

environment = Environment(
    autoescape=True, loader=FileSystemLoader(HERE), trim_blocks=False
)

index_template = environment.get_template("index.html")
gene_template = environment.get_template("gene.html")
about_template = environment.get_template("about.html")

GENE_FIXES = {
    "KRTAP21": "KRTAP21-1",  # This record has been split, apparently
    "KRTAP12": "KRTAP12-1",  # This record has been split, apparently
    "KRTAP20": "KRTAP20-1",
}

#: A set of HGNC gene symbol strings corresponding to DUBs,
#: based on HGNC gene family annotations
FAMPLEX_DUBS = {
    identifier
    for prefix, identifier in famplex.descendant_terms("FPLX", "DUB")
    if prefix == "HGNC"
}


def get_dub_type(gene_symbol: str) -> Optional[str]:
    """Get the top-level DUB type of the gene."""
    if gene_symbol not in FAMPLEX_DUBS:
        return None
    ancestors = list(famplex.ancestral_terms("HGNC", gene_symbol))
    # FIXME this might no work on the ones that have multiple trees
    if ancestors[-1] != ("FPLX", "DUB"):
        return None
    try:
        return ancestors[-2][1]
    except IndexError:
        return None


def get_go_type(identifier: str) -> str:
    if pyobo.has_ancestor('go', identifier, 'go', '0008150'):
        return 'Biological Process'
    elif pyobo.has_ancestor('go', identifier, 'go', '0005575'):
        return 'Cellular Component'
    elif pyobo.has_ancestor('go', identifier, 'go', '0003674'):
        return 'Molecular Function'
    else:
        return 'Other'


def get_cached_stmts(source: str, target: str, force: bool = False) -> List[Statement]:
    path = STATEMENTS_DIR.joinpath(source, f'{target}.json')
    if path.is_file() and not force:
        with path.open() as file:
            res = json.load(file)
    else:
        url = f'https://db.indra.bio/statements/from_agents?format=json&subject={source}&object={target}'
        res = requests.get(url).json()
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open('w') as file:
            json.dump(res, file, indent=2, sort_keys=True)
    return stmts_from_json(res['statements'].values())


def process_symbol_list(symbols):
    rv = []
    for symbol in symbols.strip().split("/"):
        gene_id = hgnc_client.get_current_hgnc_id(symbol.strip())
        if gene_id is None:
            print("missing id for", gene_id)
            continue
            # raise
        rv.append(gene_id)
    return rv


def get_processed_data():
    all_gsea_df = pd.read_csv(DATA_DUR.joinpath('gsea.tsv'), sep='\t')
    all_gsea_groups = all_gsea_df.groupby(by=['hgnc_id', 'hgnc_symbol'])
    symbol_to_gsea = {
        hgnc_symbol: gsea_df
        for (hgnc_id, hgnc_symbol), gsea_df in all_gsea_groups
    }

    with DATA_DUR.joinpath('dgea.json').open() as file:
        symbol_to_dgea = json.load(file)

    df = pd.read_csv(DATA, sep="\t")
    df["fraction_cell_lines_dependent_on_DUB"].fillna(0.0, inplace=True)

    df["dub_hgnc_id"] = df["DUB"].map(hgnc_client.get_current_hgnc_id)
    df["dub_hgnc_symbol"] = df["dub_hgnc_id"].map(hgnc_client.get_hgnc_name)
    del df["DUB"]

    rv = {}
    for (dub_hgnc_id, dub_hgnc_symbol), sdf in tqdm(df.groupby(["dub_hgnc_id", "dub_hgnc_symbol"]), unit='DUB'):
        gsea_df: pd.DataFrame = symbol_to_gsea.get(dub_hgnc_symbol)
        if gsea_df is None:
            go = []
        else:
            go = [
                dict(
                    identifier=row['go_id'],
                    name=row['go_name'],
                    type=get_go_type(row['go_id'].removeprefix('GO:')),
                    p=row['pvalue'],
                    p_adj=row['p.adjust'],
                    q=row['qvalue'],
                )
                for _, row in gsea_df.sort_values('qvalue', ascending=True).iterrows()
            ]

        fraction_dependent = sdf.iloc[0]["fraction_cell_lines_dependent_on_DUB"]
        papers = int(sdf.iloc[0]['PubMed_papers'].replace(",", ""))

        depmap_results = []
        for _, row in tqdm(sdf.iterrows(), unit='dep', leave=False):
            depmap_gene = row["DepMap_coDependency"]
            if pd.isna(depmap_gene):
                continue
            depmap_gene = GENE_FIXES.get(depmap_gene, depmap_gene)
            depmap_gene_id = hgnc_client.get_current_hgnc_id(depmap_gene)
            if depmap_gene_id is None:
                raise ValueError(
                    f"could not map depmap dependency for {dub_hgnc_symbol} to HGNC ID: {depmap_gene}"
                )
            depmap_gene_symbol = hgnc_client.get_hgnc_name(depmap_gene_id)

            stmts = get_cached_stmts(dub_hgnc_symbol, depmap_gene_symbol)

            depmap_result = dict(
                hgnc_id=depmap_gene_id,
                hgnc_symbol=depmap_gene_symbol,
                hgnc_name=pyobo.get_definition("hgnc", depmap_gene_id),
                correlation=row["DepMap_correlation"],
                interactions=dict(
                    biogrid=row['Biogrid'] == 'yes',
                    intact=row['IntAct'] == 'yes',
                    nursa=row['NURSA'] == 'yes',
                    pc=row['PathwayCommons'] == 'yes',
                    ppid=row['PPID_support'] == 'yes',
                    indra=len(stmts) > 0,
                ),
            )

            ccle_corr = row["CCLE_Proteomics_correlation"]
            if pd.notna(ccle_corr):
                depmap_result["ccle"] = dict(
                    correlation=ccle_corr,
                    p_adj=row["CCLE_Proteomics_p_adj"],
                    z=row["CCLE_Proteomics_z_score"],
                    significant=row["CCLE_Proteomics_z_score_sig"],
                )

            cmap_score = row["CMAP_Score"]
            if pd.notna(cmap_score):
                depmap_result["cmap"] = dict(
                    score=cmap_score,
                    type=row["CMAP_Perturbation_Type"],
                )
            depmap_results.append(depmap_result)

        # FamPlex identifier for the DUB class
        dub_class = get_dub_type(dub_hgnc_symbol)

        # External IDs
        entrez_id = hgnc_client.get_entrez_id(dub_hgnc_id)

        rv[dub_hgnc_symbol] = dict(
            hgnc_id=dub_hgnc_id,
            hgnc_symbol=dub_hgnc_symbol,
            hgnc_name=pyobo.get_definition("hgnc", dub_hgnc_id),
            uniprot_id=hgnc_client.get_uniprot_id(dub_hgnc_id),
            entrez_id=entrez_id,
            # description=pyobo.get_definition('ncbigene', entrez_id),
            mgi_id=hgnc_client.get_mouse_id(dub_hgnc_id),
            rgd_id=hgnc_client.get_rat_id(dub_hgnc_id),
            #: If this DUB is not annotated in HGNC/FamPlex
            dub_class=dub_class,
            # rnaseq=rnaseq,
            papers=papers,
            fraction_cell_lines_dependent=fraction_dependent,
            go=go,
            depmap=depmap_results,
            dgea=symbol_to_dgea.get(dub_hgnc_symbol),
        )
    return rv


def get_rv(force: bool = True):
    if DATA_PROCESSED.is_file() and not force:
        with DATA_PROCESSED.open() as file:
            return json.load(file)

    rv = get_processed_data()
    with DATA_PROCESSED.open("w") as file:
        json.dump(rv, file, indent=2, sort_keys=True)

    # Make a venn diagram describing the overlaps
    venn2([set(rv), FAMPLEX_DUBS], ["DUB Portal", "FamPlex"])
    plt.tight_layout()
    plt.savefig(DOCS.joinpath("overlap.svg"))

    return rv


def _d(symbols):
    return [
        dict(identifier=identifier, symbol=symbol, name=pyobo.get_definition('hgnc', identifier))
        for identifier, symbol in (
            (hgnc_client.get_current_hgnc_id(symbol), symbol)
            for symbol in symbols)
    ]


@click.command()
@force_option
@verbose_option
def main(force: bool):
    rv = get_rv(force=force)

    rows = sorted(rv.values(), key=itemgetter('hgnc_symbol'))
    index_html = index_template.render(rows=rows)
    with open(os.path.join(DOCS, "index.html"), "w") as file:
        print(index_html, file=file)

    with NDEX_LINKS.open() as file:
        ndex_links = json.load(file)

    unique_dubportal = _d(sorted(set(rv) - FAMPLEX_DUBS))
    unique_famplex = _d(sorted(FAMPLEX_DUBS - set(rv)))

    about_html = about_template.render(unique_famplex=unique_famplex, unique_dubportal=unique_dubportal)
    about_dir = DOCS.joinpath('about')
    about_dir.mkdir(exist_ok=True, parents=True)
    with about_dir.joinpath('index.html').open('w') as file:
        print(about_html, file=file)

    for row in rows:
        gene_html = gene_template.render(
            record=row, ndex=ndex_links[row["hgnc_symbol"]]
        )
        directory = DOCS.joinpath(row["hgnc_symbol"])
        directory.mkdir(exist_ok=True, parents=True)
        with directory.joinpath("index.html").open("w") as file:
            print(gene_html, file=file)


if __name__ == "__main__":
    main()

import json
import logging
import os
import pathlib
from functools import lru_cache
from operator import itemgetter
from typing import List, Optional

import click
import markupsafe
import matplotlib.pyplot as plt
import pandas as pd
import pyobo
import pystow
import requests
from jinja2 import Environment, FileSystemLoader
from matplotlib_venn import venn2
from more_click import force_option, verbose_option
from tqdm import tqdm

import famplex
from indra.assemblers.html import HtmlAssembler
from indra.databases import hgnc_client
from indra.sources.indra_db_rest import get_statements
from indra.statements import (
    Desumoylation, Deubiquitination, RegulateActivity, RegulateAmount, RemoveModification, Statement, stmts_from_json,
    stmts_from_json_file, stmts_to_json_file,
)
from indra.tools import assemble_corpus as ac

logger = logging.getLogger(__name__)
HERE = pathlib.Path(__file__).parent.resolve()
DOCS = HERE.parent.joinpath("docs")
DATA_DUR = HERE.joinpath("data")
DATA = DATA_DUR.joinpath("DUB_website_main_v2.tsv")
DATA_PROCESSED = DATA_DUR.joinpath("data.json")
NDEX_LINKS = DOCS.joinpath("network_index.json")

STATEMENTS_DIR = DATA_DUR.joinpath("statements")
STATEMENTS_DIR.mkdir(exist_ok=True, parents=True)

environment = Environment(
    autoescape=True, loader=FileSystemLoader(HERE), trim_blocks=False
)

index_template = environment.get_template("index.html")
gene_template = environment.get_template("gene.html")
about_template = environment.get_template("about.html")
stmt_template = environment.get_template("statements_view.html")

DUB = "Deubiquitinase"
GENE_FIXES = {
    "KRTAP21": "KRTAP21-1",  # This record has been split, apparently
    "KRTAP12": "KRTAP12-1",  # This record has been split, apparently
    "KRTAP20": "KRTAP20-1",
}

#: A set of HGNC gene symbol strings corresponding to DUBs,
#: based on HGNC gene family annotations
FAMPLEX_DUBS = {
    identifier
    for prefix, identifier in famplex.descendant_terms("FPLX", DUB)
    if prefix == "HGNC"
}


def get_dub_type(gene_symbol: str) -> Optional[str]:
    """Get the top-level DUB type of the gene."""
    if gene_symbol not in FAMPLEX_DUBS:
        return None
    ancestors = list(famplex.ancestral_terms("HGNC", gene_symbol))
    # FIXME this might no work on the ones that have multiple trees
    if ancestors[-1] != ("FPLX", DUB):
        return None
    try:
        return ancestors[-2][1]
    except IndexError:
        return None


def get_go_type(identifier: str) -> str:
    if pyobo.has_ancestor("go", identifier, "go", "0008150"):
        return "Biological Process"
    elif pyobo.has_ancestor("go", identifier, "go", "0005575"):
        return "Cellular Component"
    elif pyobo.has_ancestor("go", identifier, "go", "0003674"):
        return "Molecular Function"
    else:
        return "Other"


def get_cached_stmts_single(hgnc_id, force: bool = False) -> list[Statement]:
    path = pystow.join("dubportal", "single", name=f"{hgnc_id}.json")
    if path.is_file() and not force:
        stmts = stmts_from_json_file(path)
    else:
        ip = get_statements(agents=[f"{hgnc_id}@HGNC"], ev_limit=10000)
        stmts = ip.statements
        stmts_to_json_file(stmts, path)

    stmts = dubportal_assembly(stmts)
    return stmts


def dubportal_assembly(stmts: list[Statement]) -> list[Statement]:
    stmts = filter_out_medscan(stmts)
    stmts = filter_curations(stmts)
    stmts = only_dubbing(stmts)
    stmts = first_n_evidences(stmts)
    return stmts


def only_dubbing(stmts: list[Statement]) -> list[Statement]:
    return [
        stmt
        for stmt in stmts
        if isinstance(
            stmt, (Deubiquitination, Desumoylation, RegulateActivity, RegulateAmount)
        )
    ]


def first_n_evidences(stmts: list[Statement], n: int = 10) -> list[Statement]:
    for stmt in stmts:
        stmt.evidence = stmt.evidence[:n]
    return stmts


def filter_out_medscan(stmts: list[Statement]) -> list[Statement]:
    logger.debug("Starting medscan filter with %d statements" % len(stmts))
    new_stmts = []
    for stmt in stmts:
        new_evidence = []
        for ev in stmt.evidence:
            if ev.source_api == "medscan":
                continue
            new_evidence.append(ev)
        stmt.evidence = new_evidence
        if new_evidence:
            new_stmts.append(stmt)
    logger.debug("Finished medscan filter with %d statements" % len(new_stmts))
    return new_stmts


def filter_curations(stmts: list[Statement]) -> list[Statement]:
    curs = safe_get_curations()
    if curs is not None:
        logger.info("filtering %d curations", len(curs))
        stmts = ac.filter_by_curation(stmts, curs)
    return stmts


@lru_cache(maxsize=1)
def safe_get_curations():
    try:
        from indra_db.client.principal.curation import get_curations
    except ImportError:
        return None
    try:
        return get_curations()
    except:
        return None


def get_cached_stmts(source: str, target: str, force: bool = False) -> List[Statement]:
    path = STATEMENTS_DIR.joinpath(source, f"{target}.json")
    if path.is_file() and not force:
        with path.open() as file:
            res = json.load(file)
    else:
        url = f"https://db.indra.bio/statements/from_agents?format=json&subject={source}&object={target}"
        res = requests.get(url).json()
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as file:
            json.dump(res, file, indent=2, sort_keys=True)
    return stmts_from_json(res["statements"].values())


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
    all_gsea_df = pd.read_csv(DATA_DUR.joinpath("gsea.tsv"), sep="\t")
    all_gsea_groups = all_gsea_df.groupby(by=["hgnc_id", "hgnc_symbol"])
    symbol_to_gsea = {
        hgnc_symbol: gsea_df for (hgnc_id, hgnc_symbol), gsea_df in all_gsea_groups
    }

    with DATA_DUR.joinpath("dgea.json").open() as file:
        symbol_to_dgea = json.load(file)

    df = pd.read_csv(DATA, sep="\t")
    df["fraction_cell_lines_dependent_on_DUB"].fillna("Unmeasured", inplace=True)

    df["dub_hgnc_id"] = df["DUB"].map(hgnc_client.get_current_hgnc_id)
    df["dub_hgnc_symbol"] = df["dub_hgnc_id"].map(hgnc_client.get_hgnc_name)
    del df["DUB"]

    rv = {}
    for (dub_hgnc_id, dub_hgnc_symbol), sdf in tqdm(
        df.groupby(["dub_hgnc_id", "dub_hgnc_symbol"]), unit="DUB"
    ):
        gsea_df: pd.DataFrame = symbol_to_gsea.get(dub_hgnc_symbol)
        if gsea_df is None:
            go = []
        else:
            go = [
                dict(
                    identifier=row["go_id"],
                    name=row["go_name"],
                    type=get_go_type(row["go_id"].removeprefix("GO:")),
                    p=row["pvalue"],
                    p_adj=row["p.adjust"],
                    q=row["qvalue"],
                )
                for _, row in gsea_df.sort_values("qvalue", ascending=True).iterrows()
            ]

        fraction_dependent = sdf.iloc[0]["fraction_cell_lines_dependent_on_DUB"]
        papers = int(sdf.iloc[0]["PubMed_papers"].replace(",", ""))

        depmap_results = []
        for _, row in tqdm(sdf.iterrows(), unit="dep", leave=False):
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
                    biogrid=row["Biogrid"] == "yes",
                    intact=row["IntAct"] == "yes",
                    nursa=row["NURSA"] == "yes",
                    pc=row["PathwayCommons"] == "yes",
                    ppid=row["PPID_support"] == "yes",
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
        dict(
            identifier=identifier,
            symbol=symbol,
            name=pyobo.get_definition("hgnc", identifier),
        )
        for identifier, symbol in (
            (hgnc_client.get_current_hgnc_id(symbol), symbol) for symbol in symbols
        )
    ]


@click.command()
@force_option
@verbose_option
def main(force: bool):
    rv = get_rv(force=force)

    # Load KO gene set enrichment analysis
    with DATA_DUR.joinpath('ko_gsea.json').open() as file:
        ko_gsea_dict = json.load(file)
    for hgnc_symbol, enrichments in ko_gsea_dict.items():
        rv[hgnc_symbol]['ko_gsea'] = enrichments

    # Load INDRA statements
    dub_symbol_statments = {}
    for key, value in rv.items():
        gene_stmts = get_cached_stmts_single(value["hgnc_id"])
        gene_stmts = ac.filter_grounded_only(gene_stmts)
        dub_symbol_statments[value["hgnc_symbol"]] = gene_stmts
        rv[key]["n_statements"] = len(gene_stmts)
        rv[key]["n_dub_statements"] = sum(
            isinstance(s, Deubiquitination) for s in gene_stmts
        )
        rv[key]["n_other_statements"] = len(gene_stmts) - rv[key]["n_dub_statements"]

    rows = sorted(rv.values(), key=itemgetter("hgnc_symbol"))
    index_html = index_template.render(rows=rows)
    with open(os.path.join(DOCS, "index.html"), "w") as file:
        print(index_html, file=file)

    with NDEX_LINKS.open() as file:
        ndex_links = json.load(file)

    unique_dubportal = _d(sorted(set(rv) - FAMPLEX_DUBS))
    unique_famplex = _d(sorted(FAMPLEX_DUBS - set(rv)))

    about_html = about_template.render(
        unique_famplex=unique_famplex, unique_dubportal=unique_dubportal
    )
    about_dir = DOCS.joinpath("about")
    about_dir.mkdir(exist_ok=True, parents=True)
    with about_dir.joinpath("index.html").open("w") as file:
        print(about_html, file=file)

    for row in tqdm(rows):
        hgnc_symbol = row["hgnc_symbol"]

        stmts = dub_symbol_statments.get(hgnc_symbol, [])
        dub_assembler = HtmlAssembler(
            [stmt for stmt in stmts if isinstance(stmt, RemoveModification)],
            db_rest_url="https://db.indra.bio",
        )
        dub_stmt_html = dub_assembler.make_model(
            template=stmt_template, grouping_level="statement"
        )
        other_assembler = HtmlAssembler(
            [stmt for stmt in stmts if not isinstance(stmt, RemoveModification)],
            db_rest_url="https://db.indra.bio",
        )
        other_stmt_html = other_assembler.make_model(template=stmt_template)

        gene_html = gene_template.render(
            record=row,
            ndex=ndex_links.get(hgnc_symbol),
            dub_stmt_html=markupsafe.Markup(dub_stmt_html),
            other_stmt_html=markupsafe.Markup(other_stmt_html),
        )
        directory = DOCS.joinpath(row["hgnc_symbol"])
        directory.mkdir(exist_ok=True, parents=True)
        with directory.joinpath("index.html").open("w") as file:
            print(gene_html, file=file)


if __name__ == "__main__":
    main()

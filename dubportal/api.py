import json
import logging
import os
import pathlib
from collections import defaultdict
from functools import lru_cache
from operator import itemgetter
from typing import Mapping, Optional

import bioversions
import click
import famplex
import markupsafe
import matplotlib.pyplot as plt
import pandas as pd
import pyobo
import pystow
import seaborn as sns
from indra.assemblers.html import HtmlAssembler
from indra.databases import go_client, hgnc_client
from indra.sources.indra_db_rest import get_statements
from indra.statements import (
    Desumoylation,
    Deubiquitination,
    RegulateActivity,
    RegulateAmount,
    RemoveModification,
    Statement,
    stmts_from_json_file,
    stmts_to_json_file,
)
from indra.tools import assemble_corpus as ac
from jinja2 import Environment, FileSystemLoader
from matplotlib_venn import venn2
from more_click import force_option, verbose_option
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

HERE = pathlib.Path(__file__).parent.resolve()
DOCS = HERE.parent.joinpath("docs")
RAW = HERE.joinpath("raw")
PROCESSED = HERE.joinpath("processed")
INPUT_PATH = RAW.joinpath("DUB_website_main_v2.tsv")
OUTPUT_PATH = PROCESSED.joinpath("data.json")
NDEX_LINKS = DOCS.joinpath("network_index.json")

STATEMENTS_DIR = RAW.joinpath("statements")
STATEMENTS_DIR.mkdir(exist_ok=True, parents=True)

environment = Environment(autoescape=True, loader=FileSystemLoader(HERE), trim_blocks=False)

index_template = environment.get_template("index.html")
gene_template = environment.get_template("gene.html")
about_template = environment.get_template("about.html")
stmt_template = environment.get_template("statements_view.html")

DUB = "Deubiquitinase"
GENE_FIXES = {
    "KRTAP21": "KRTAP21-1",  # These records have been split, apparently
    "KRTAP12": "KRTAP12-1",
    "KRTAP20": "KRTAP20-1",
}

#: A set of HGNC gene symbol strings corresponding to DUBs,
#: based on HGNC gene family annotations
FAMPLEX_DUBS = {
    identifier for prefix, identifier in famplex.descendant_terms("FPLX", DUB) if prefix == "HGNC"
}


def get_dub_family(gene_symbol: str) -> Optional[str]:
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


def get_go_type(identifier: str) -> Optional[str]:
    """Get the GO type by the GO identifier."""
    go_namespace = go_client.get_namespace(f"GO:{identifier}")
    if go_namespace is None:
        return None
    return go_namespace.replace("_", " ").title()


def get_gene_statements(hgnc_id: str, force: bool = False) -> list[Statement]:
    """Get INDRA statements for the given gene."""
    path = pystow.join("dubportal", "single", name=f"{hgnc_id}.json")
    path_meta = pystow.join("dubportal", "single", name=f"{hgnc_id}_meta.json")
    if path.is_file() and not force:
        return stmts_from_json_file(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    ip = get_statements(agents=[f"{hgnc_id}@HGNC"], ev_limit=30)
    stmts = ip.statements
    stmts_to_json_file(stmts, path)
    soruce_counts = ip.get_source_counts()
    ev_counts = ip.get_ev_counts()
    meta = {'source_counts': soruce_counts,
            'ev_counts': ev_counts}
    with open(path_meta, 'w') as fh:
        json.dump(meta, fh, indent=1)
    return stmts, meta


def get_interaction_stmts(source: str, target: str, force: bool = False) -> list[Statement]:
    """Get INDRA statements for the given interaction between source/target."""
    path = pystow.join("dubportal", "interaction", source, name=f"{target}.json")
    if path.is_file() and not force:
        return stmts_from_json_file(path)

    path.parent.mkdir(exist_ok=True, parents=True)
    ip = get_statements(subject=source, object=target, ev_limit=30)
    stmts_to_json_file(ip.statements, path)
    return ip.statements


def dubportal_preassembly(stmts: list[Statement]) -> list[Statement]:
    stmts = filter_out_medscan(stmts)
    stmts = filter_curations(stmts)
    stmts = only_dubbing(stmts)
    stmts = first_k_evidences(stmts, k=10)
    stmts = ac.filter_grounded_only(stmts)
    return stmts


def filter_stmt_type(stmts: list[Statement], types) -> list[Statement]:
    return [stmt for stmt in stmts if isinstance(stmt, types)]


def only_dubbing(stmts: list[Statement]) -> list[Statement]:
    """Filter to statements of the given type"""
    return filter_stmt_type(
        stmts, (Deubiquitination, Desumoylation, RegulateActivity, RegulateAmount)
    )


def first_k_evidences(stmts: list[Statement], *, k: int = 10) -> list[Statement]:
    for stmt in stmts:
        stmt.evidence = stmt.evidence[:k]
    return stmts


def remove_source_api(stmts: list[Statement], source_api: str) -> list[Statement]:
    """Remove evidences with the given source API then filter statements with no evidences."""
    rv = []
    init_ev_count = 0
    final_ev_count = 0
    for stmt in stmts:
        init_ev_count += len(stmt.evidence)
        stmt.evidence = [ev for ev in stmt.evidence if ev.source_api != source_api]
        final_ev_count += len(stmt.evidence)
        if stmt.evidence:
            rv.append(stmt)
    if len(rv) < len(stmts) or final_ev_count < init_ev_count:
        logger.info(
            "Filtered %s from %d->%d evidences and %d->%d statements",
            source_api,
            init_ev_count,
            final_ev_count,
            len(stmts),
            len(rv),
        )
    return rv


def filter_out_medscan(stmts: list[Statement]) -> list[Statement]:
    return remove_source_api(stmts, "medscan")


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
        return get_curations() or None
    except:
        return None


def shared_reactome(hgnc_id_1: str, hgnc_id_2: str) -> set[str]:
    uniprot_to_pathways = get_protein_to_pathways()
    uniprot_a = hgnc_client.get_uniprot_id(hgnc_id_1)
    uniprot_a_pathways = uniprot_to_pathways.get(uniprot_a, set())
    uniprot_b = hgnc_client.get_uniprot_id(hgnc_id_2)
    uniprot_b_pathways = uniprot_to_pathways.get(uniprot_b, set())
    return uniprot_a_pathways.intersection(uniprot_b_pathways)


@lru_cache(maxsize=1)
def get_protein_to_pathways() -> Mapping[str, set[str]]:
    """Get protein to pathways from reactome."""
    version = bioversions.get_version("reactome")
    url = f"https://reactome.org/download/{version}/UniProt2Reactome_All_Levels.txt"
    rv = defaultdict(set)
    df = pd.read_csv(url, sep="\t", header=None, usecols=[0, 1], dtype=str)
    print(df.head())
    for uniprot_id, reactome_id in df.values:
        rv[uniprot_id].add(reactome_id)
    return dict(rv)


def get_processed_data() -> dict[str, any]:
    with PROCESSED.joinpath("dep_gsea.json").open() as file:
        symbol_to_depmap_enrichment = json.load(file)
    with PROCESSED.joinpath("ko_dgea.json").open() as file:
        symbol_to_ko_dgea = json.load(file)
    with PROCESSED.joinpath("ko_gsea.json").open() as file:
        symbol_to_ko_gsea = json.load(file)

    df = pd.read_csv(INPUT_PATH, sep="\t")
    df["fraction_cell_lines_dependent_on_DUB"].fillna("Unmeasured", inplace=True)

    df["dub_hgnc_id"] = df["DUB"].map(hgnc_client.get_current_hgnc_id)
    df["dub_hgnc_symbol"] = df["dub_hgnc_id"].map(hgnc_client.get_hgnc_name)
    del df["DUB"]

    rv = {}

    df["DepMap_coDependency"] = df["DepMap_coDependency"].map(lambda s: GENE_FIXES.get(s, s))
    df["dep_gene_id"] = df["DepMap_coDependency"].map(hgnc_client.get_current_hgnc_id)
    df["dep_gene_symbol"] = df["dep_gene_id"].map(hgnc_client.get_hgnc_name, na_action="ignore")

    for (dub_hgnc_id, dub_hgnc_symbol), sdf in tqdm(
        df.groupby(["dub_hgnc_id", "dub_hgnc_symbol"]),
        unit="DUB",
        desc="Processing database",
    ):
        # Non-grouping operations
        fraction_dependent = sdf.iloc[0]["fraction_cell_lines_dependent_on_DUB"]
        papers = int(sdf.iloc[0]["PubMed_papers"].replace(",", ""))

        # FamPlex identifier for the DUB class
        dub_family = get_dub_family(dub_hgnc_symbol)

        # External IDs
        entrez_id = hgnc_client.get_entrez_id(dub_hgnc_id)

        # Get gene dependencies for this DUB
        depmap_gene_records = []
        for _, row in tqdm(sdf.iterrows(), unit="dep", leave=False, desc="Mapping dependencies"):
            depmap_gene_id = row["dep_gene_id"]
            if pd.isna(depmap_gene_id) or pd.isnull(depmap_gene_id) or not depmap_gene_id:
                continue
            depmap_gene_symbol = row["dep_gene_symbol"]
            stmts = get_interaction_stmts(dub_hgnc_symbol, depmap_gene_symbol)

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
                    indra=len(stmts),
                    reactome=len(shared_reactome(dub_hgnc_id, depmap_gene_id)),
                    dge=any(
                        r["hgnc_id"] == depmap_gene_id
                        for r in symbol_to_ko_dgea.get(dub_hgnc_id, [])
                    ),
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
            depmap_gene_records.append(depmap_result)

        # Sort descending by absolute value
        depmap_gene_records = sorted(
            depmap_gene_records,
            key=lambda record: abs(record["correlation"]),
            reverse=True,
        )

        # Get over-representation analysis records for this DUB based on gene dependencies
        depmap_enrichment_records = [
            dict(
                identifier=record["go_id"],
                name=record["go_name"],
                type=get_go_type(record["go_id"]),
                p=record["pvalue"],
                p_adj=record["p.adjust"],
                q=record["qvalue"],
            )
            for record in symbol_to_depmap_enrichment.get(dub_hgnc_symbol, [])
        ]

        rv[dub_hgnc_symbol] = dict(
            hgnc_id=dub_hgnc_id,
            hgnc_symbol=dub_hgnc_symbol,
            hgnc_name=pyobo.get_definition("hgnc", dub_hgnc_id),
            # description=pyobo.get_definition('ncbigene', entrez_id),
            dub_family=dub_family,
            xrefs=dict(
                uniprot_id=hgnc_client.get_uniprot_id(dub_hgnc_id),
                entrez_id=entrez_id,
            ),
            orthologs=cdict(
                mgi=hgnc_client.get_mouse_id(dub_hgnc_id),
                rgd=hgnc_client.get_rat_id(dub_hgnc_id),
            ),
            papers=papers,
            fraction_cell_lines_dependent=fraction_dependent,
            depmap=dict(
                genes=depmap_gene_records,
                enrichment=depmap_enrichment_records,
            ),
            knockdown=dict(
                genes=symbol_to_ko_dgea.get(dub_hgnc_symbol),
                enrichment=symbol_to_ko_gsea.get(dub_hgnc_symbol),
            ),
        )
    return rv


def cdict(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}


def get_rv(force: bool):
    if OUTPUT_PATH.is_file() and not force:
        with OUTPUT_PATH.open() as file:
            return json.load(file)

    rv = get_processed_data()
    with OUTPUT_PATH.open("w") as file:
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
    with logging_redirect_tqdm():
        _main_helper(force)


def _main_helper(force: bool):
    rv = get_rv(force=force)

    fraction_direct_explained = {
        symbol: (
            sum(
                (
                    0 < sr["interactions"]["indra"]
                    or 0 < sr["interactions"]["reactome"]
                    or sr["interactions"]["biogrid"]
                    or sr["interactions"]["intact"]
                    or sr["interactions"]["nursa"]
                    or sr["interactions"]["pc"]
                    or sr["interactions"]["dge"]
                )
                for sr in record["depmap"]["genes"]
            )
            / len(record["depmap"]["genes"])
        )
        for symbol, record in rv.items()
        if record["depmap"] and record["depmap"]["genes"]
    }
    fig, ax = plt.subplots()
    sns.histplot(list(fraction_direct_explained.values()), ax=ax)
    ax.set_title("Expained DUB Dependencies")
    ax.set_ylabel("Number DUBs")
    ax.set_xlabel("Percentage of Top K Dependencies Explained")
    fig.savefig(DOCS.joinpath("explanations.svg"))
    fig.savefig(DOCS.joinpath("explanations.png"), dpi=300)
    plt.close(fig)

    # Load INDRA statements
    dub_symbol_statments = {}
    for key, value in rv.items():
        gene_stmts, gene_stmts_meta = get_gene_statements(value["hgnc_id"])
        gene_stmts = dubportal_preassembly(gene_stmts)
        dub_symbol_statments[value["hgnc_symbol"]] = gene_stmts
        rv[key]["n_statements"] = len(gene_stmts)
        rv[key]["n_dub_statements"] = sum(isinstance(stmt, Deubiquitination) for stmt in gene_stmts)
        rv[key]["n_other_statements"] = len(gene_stmts) - rv[key]["n_dub_statements"]

    rows = sorted(rv.values(), key=itemgetter("hgnc_symbol"))
    index_html = index_template.render(rows=rows)
    with open(os.path.join(DOCS, "index.html"), "w") as file:
        print(index_html, file=file)

    unique_dubportal = _d(sorted(set(rv) - FAMPLEX_DUBS))
    unique_famplex = _d(sorted(FAMPLEX_DUBS - set(rv)))

    about_html = about_template.render(
        unique_famplex=unique_famplex, unique_dubportal=unique_dubportal
    )
    about_dir = DOCS.joinpath("about")
    about_dir.mkdir(exist_ok=True, parents=True)
    with about_dir.joinpath("index.html").open("w") as file:
        print(about_html, file=file)

    for row in tqdm(rows, desc="Putting it all together"):
        hgnc_symbol = row["hgnc_symbol"]

        stmts = dub_symbol_statments.get(hgnc_symbol, [])
        dub_assembler = HtmlAssembler(
            [stmt for stmt in stmts if isinstance(stmt, RemoveModification)],
            db_rest_url="https://db.indra.bio",
            source_counts=gene_stmts_meta['source_counts'],
            ev_counts=gene_stmts_meta['ev_counts'],
        )
        dub_stmt_html = dub_assembler.make_model(template=stmt_template, grouping_level="statement")
        other_assembler = HtmlAssembler(
            [stmt for stmt in stmts if not isinstance(stmt, RemoveModification)],
            db_rest_url="https://db.indra.bio",
        )
        other_stmt_html = other_assembler.make_model(template=stmt_template)

        gene_html = gene_template.render(
            record=row,
            dub_stmt_html=markupsafe.Markup(dub_stmt_html),
            other_stmt_html=markupsafe.Markup(other_stmt_html),
            source_counts=gene_stmts_meta['source_counts'],
            ev_counts=gene_stmts_meta['ev_counts'],
        )
        directory = DOCS.joinpath(row["hgnc_symbol"])
        directory.mkdir(exist_ok=True, parents=True)
        with directory.joinpath("index.html").open("w") as file:
            print(gene_html, file=file)


if __name__ == "__main__":
    logging.getLogger("indra.tools.assemble_corpus").setLevel(logging.WARNING)
    main()

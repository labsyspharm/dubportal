import json
import logging
import os
import pathlib
from typing import Optional

import click
import famplex
import gilda
import matplotlib.pyplot as plt
import pandas as pd
import pyobo
from indra.databases import go_client, hgnc_client
from jinja2 import Environment, FileSystemLoader
from matplotlib_venn import venn2
from more_click import force_option, verbose_option

logger = logging.getLogger(__name__)
HERE = pathlib.Path(__file__).parent.resolve()
DOCS = HERE.parent.joinpath("docs")
DATA = HERE.joinpath("data", "DUB_website_main_v2.tsv")
DATA_PROCESED = HERE.joinpath("data", "data.json")
NDEX_LINKS = DOCS.joinpath("network_index.json")

environment = Environment(
    autoescape=True, loader=FileSystemLoader(HERE), trim_blocks=False
)

index_template = environment.get_template("index.html")
gene_template = environment.get_template("gene.html")

GENE_FIXES = {
    "KRTAP21": "KRTAP21-1",  # This record has been split, apparently
    "KRTAP12": "KRTAP12-1",  # This record has been split, apparently
    "KRTAP20": "KRTAP20-1",
}
GO_FIXES = {
    "GO_TOLL_LIKE_RECEPTOR_SIGNALING_PATHWAY": "toll-like receptor 13 signaling pathway",
    "GO_PROTEIN_K11_LINKED_UBIQUITINATION": "protein K11-linked ubiquitination",
    "GO_HISTONE_H2A_K63_LINKED_UBIQUITINATION": "histone H2A K63-linked ubiquitination",
    "GO_NEGATIVE_REGULATION_OF_CELL_CYCLE_G2_M_PHASE_TRANSITION": "negative regulation of cell cycle G2/M phase transition",
    "GO_REGULATION_OF_NIK_NF_KAPPAB_SIGNALING": "negative regulation of NIK/NF-kappaB signaling",
    "GO_DNA_DAMAGE_RESPONSE_SIGNAL_TRANSDUCTION_BY_P53_CLASS_MEDIATOR": "DNA damage response, signal transduction by p53 class mediator",
    "GO_SAGA_TYPE_COMPLEX": "SAGA-type complex",
    "GO_G_PROTEIN_BETA_GAMMA_SUBUNIT_COMPLEX_BINDING": "G-protein beta/gamma-subunit complex binding",
    "GO_NEGATIVE_REGULATION_OF_PROTEASOMAL_UBIQUITIN_DEPENDENT_PROTEIN_CATABOLIC_PROCESS": "negative regulation of proteasomal ubiquitin-dependent protein catabolic process",
    "GO_BASE_EXCISION_REPAIR": "base-excision repair",
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


def process_go_label(go_term):
    if go_term == "none significant":
        return
    go_term = GO_FIXES.get(go_term, go_term)
    go_term_norm = go_term.removeprefix("GO_").replace("_", " ").lower()
    go_id = go_client.get_go_id_from_label(go_term_norm)
    if go_id:
        return go_id

    matches = gilda.ground(go_term_norm)
    for match in matches:
        if match.term.db == "GO":
            return match.term.id


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
    df = pd.read_csv(DATA, sep="\t")
    df.rename(
        inplace=True,
        columns={
            # The GO gene set enrichment analysis was done for each DUB on all of its co-dependencies. This
            #  is the top reported GO term name
            "GO_enriched_in_codependencies": "go_mmsig_name",
            # p-value of GO gene set enrichment analysis
            "pvalue": "go_p",
            # adjusted p-value of GO gene set enrichment analysis
            "p.adjust": "go_p_adj",
            # q-value of the most significant GO term in gene set enrichment analysis
            "qvalue": "go_q",
            # The list of genes associated to the most significant GO term
            "geneID": "go_gene_symbols",
        },
    )

    df["fraction_cell_lines_dependent_on_DUB"].fillna(0.0, inplace=True)

    df["dub_hgnc_id"] = df["DUB"].map(hgnc_client.get_current_hgnc_id)
    df["dub_hgnc_symbol"] = df["dub_hgnc_id"].map(hgnc_client.get_hgnc_name)
    del df["DUB"]

    df["go_id"] = df["go_mmsig_name"].map(process_go_label)
    df["go_name"] = df["go_id"].map(go_client.get_go_label, na_action="ignore")
    df["go_gene_ids"] = df["go_gene_symbols"].map(
        process_symbol_list, na_action="ignore"
    )
    del df["go_gene_symbols"]
    del df["go_mmsig_name"]

    rv = {}
    for (dub_hgnc_id, dub_hgnc_symbol), sdf in df.groupby(["dub_hgnc_id", "dub_hgnc_symbol"]):
        row = sdf.iloc[0]
        go = []
        if row["go_id"]:
            go.append(
                dict(
                    identifier=row["go_id"],
                    name=row["go_name"],
                    p=row["go_p"],
                    p_adj=row["go_p_adj"],
                    q=row["go_q"],
                    genes=[
                        {
                            "hgnc_id": go_gene_id,
                            "hgnc_name": hgnc_client.get_hgnc_name(go_gene_id),
                        }
                        for go_gene_id in row["go_gene_ids"]
                    ],
                )
            )

        fraction_dependent = row["fraction_cell_lines_dependent_on_DUB"]
        papers = int(row['PubMed_papers'].replace(",", ""))

        depmap_results = []
        for _, row in sdf.iterrows():
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
        )
    return rv


def get_rv(force: bool = True):
    if DATA_PROCESED.is_file() and not force:
        with DATA_PROCESED.open() as file:
            return json.load(file)

    rv = get_processed_data()
    with DATA_PROCESED.open("w") as file:
        json.dump(rv, file, indent=2, sort_keys=True)

    # Make a venn diagram describing the overlaps
    venn2([set(rv), FAMPLEX_DUBS], ["DUB Portal", "FamPlex"])
    plt.tight_layout()
    plt.savefig(DOCS.joinpath("overlap.svg"))

    return rv


@click.command()
@force_option
@verbose_option
def main(force: bool):
    rv = get_rv(force=force)

    rows = list(rv.values())
    index_html = index_template.render(rows=rows)
    with open(os.path.join(DOCS, "index.html"), "w") as file:
        print(index_html, file=file)

    with NDEX_LINKS.open() as file:
        ndex_links = json.load(file)

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

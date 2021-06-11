import json
import logging
import os
import pathlib

import click
import gilda
import pandas as pd
import pyobo
from indra.databases import go_client, hgnc_client
from jinja2 import Environment, FileSystemLoader
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

GROUP_KEY = [
    "DUB",
    "RNAseq",
    "PubMed_papers",
    "fraction_cell_lines_dependent_on_DUB",
    # The GO gene set enrichment analysis was done for each DUB on all of its co-dependencies. This
    #  is the top reported GO term name
    "GO_enriched_in_codependencies",
    # p-value of GO gene set enrichment analysis
    "pvalue",
    # adjusted p-value of GO gene set enrichment analysis
    "p.adjust",
    # q-avalue of GO gene set enrichment analysis
    "qvalue",
    # The list of genes associated to the GO term
    "geneID",
]
UNGROUPED_KEY = [
    "DepMap_coDependency",
    "DepMap_correlation",
    # whether codependent gene is correlated with the DUB in terms of protein abundance
    "CCLE_Proteomics_correlation",
    "CCLE_Proteomics_p_adj",
    "CCLE_Proteomics_z_score",
    "CCLE_Proteomics_z_score_sig",
    "CMAP_Score",
    "CMAP_Perturbation_Type",
]


def get_processed_data():
    df = pd.read_csv(DATA, sep="\t")
    rv = {}
    for (
        dub,
        rnaseq,
        n_papers,
        fraction_dependent,
        go_term,
        go_p,
        go_padj,
        go_q,
        go_genes,
    ), sdf in df.groupby(GROUP_KEY):
        dub_hgnc_id = hgnc_client.get_current_hgnc_id(dub)
        if dub_hgnc_id is None:
            raise
        dub_hgnc_symbol = hgnc_client.get_hgnc_name(dub_hgnc_id)

        if go_term == "none significant":
            go = []
        else:
            go_term = GO_FIXES.get(go_term, go_term)
            go_term_norm = go_term.removeprefix("GO_").replace("_", " ").lower()
            go_id = go_client.get_go_id_from_label(go_term_norm)
            if go_id is None:
                matches = gilda.ground(go_term_norm)
                for match in matches:
                    if match.term.db == "GO":
                        go_id = match.term.id
                        break
            if go_id is None:
                logger.warning(f"could not normalize: {go_term}")
                go_name = None
            else:
                go_name = go_client.get_go_label(go_id)

            # these are the genes in the GO term
            go_gene_dicts = []
            for go_gene in go_genes.strip().split("/"):
                go_gene_id = hgnc_client.get_current_hgnc_id(go_gene.strip())
                if go_gene_id is None:
                    raise
                go_gene_dicts.append(
                    {
                        "hgnc_id": go_gene_id,
                        "hgnc_name": hgnc_client.get_hgnc_name(go_gene_id),
                    }
                )
            go = [
                dict(
                    identifier=go_id,
                    name=go_name,
                    genes=go_gene_dicts,
                    p=go_p,
                    p_adj=go_padj,
                    q=go_q,
                ),
            ]

        depmap_results = []
        for (
            gene,
            depmap_corr,
            ccle_corr,
            ccle_p_adj,
            ccle_z,
            ccle_z_sig,
            cmap_score,
            cmap_type,
        ) in sdf[UNGROUPED_KEY].values:
            gene = GENE_FIXES.get(gene, gene)
            depmap_gene_id = hgnc_client.get_current_hgnc_id(gene)
            if depmap_gene_id is None:
                raise ValueError(
                    f"could not map depmap dependency for {dub} to HGNC ID: {gene}"
                )
            depmap_gene_symbol = hgnc_client.get_hgnc_name(depmap_gene_id)
            depmap_result = dict(
                hgnc_id=depmap_gene_id,
                hgnc_symbol=depmap_gene_symbol,
                correlation=depmap_corr,
            )
            if pd.notna(ccle_corr):
                depmap_result["ccle"] = dict(
                    correlation=ccle_corr,
                    p_adj=ccle_p_adj,
                    z=ccle_z,
                    significant=ccle_z_sig,
                )
            if pd.notna(cmap_score):
                depmap_result["cmap"] = dict(
                    score=cmap_score,
                    type=cmap_type,
                )
            depmap_results.append(depmap_result)
        rv[dub] = dict(
            hgnc_id=dub_hgnc_id,
            hgnc_symbol=dub_hgnc_symbol,
            uniprot_id=hgnc_client.get_uniprot_id(dub_hgnc_id),
            entrez_id=hgnc_client.get_entrez_id(dub_hgnc_id),
            mgi_id=hgnc_client.get_mouse_id(dub_hgnc_id),
            rgd_id=hgnc_client.get_rat_id(dub_hgnc_id),
            description=pyobo.get_definition("hgnc", dub_hgnc_id),
            rnaseq=rnaseq,
            papers=int(n_papers.replace(",", "")),
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
    return rv


@click.command()
@force_option
@verbose_option
def main(force: bool):
    rows = list(get_rv(force=force).values())

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

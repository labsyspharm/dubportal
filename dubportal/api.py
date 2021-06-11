import json
import logging
import os
import pathlib

import gilda
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from indra.databases import go_client, hgnc_client

logger = logging.getLogger(__name__)
HERE = pathlib.Path(__file__).parent.resolve()
DOCS = HERE.parent.joinpath("docs")
DATA = HERE.joinpath("data", "DUB_website_main_v2.tsv")
DATA_PROCESED = HERE.joinpath("data", "data.json")
environment = Environment(
    autoescape=True, loader=FileSystemLoader(HERE), trim_blocks=False
)

index_template = environment.get_template("index.html")


def get_df() -> pd.DataFrame:
    """Get the pre-processed dataframe."""
    df = pd.read_csv(DATA, sep="\t")
    return df


def html():
    df = get_df()
    main_keys = [
        "geneID",
        "PubMed_papers",
        "fraction_cell_lines_dependent_on_DUB",
        "pvalue",
        "p.adjust",
        "qvalue",
    ]
    main_data = {}
    for _, row in df.iterrows():
        main_data[row["DUB"]] = {k: row[k] for k in main_keys}
    rows = [(dub, *data.values()) for dub, data in main_data.items()]
    print(rows)

    terms_df = pd.DataFrame(rows, columns=["DUB", *main_keys])
    index_html = index_template.render(terms_df=terms_df)
    with open(os.path.join(DOCS, "index.html"), "w") as file:
        print(index_html, file=file)


GENE_FIXES = {
    'KRTAP21': 'KRTAP21-1',  # This record has been split, apparently
}
GO_FIXES = {
    'GO_TOLL_LIKE_RECEPTOR_SIGNALING_PATHWAY': 'toll-like receptor 13 signaling pathway',
}


def main():
    df = get_df()
    pd.set_option('display.max_columns', None)
    pd.set_option("precision", 3)
    del df['enriched_lineages']  # always "link"
    gkey = [
        'DUB', 'RNAseq', 'PubMed_papers',
        'fraction_cell_lines_dependent_on_DUB',
        # The GO gene set enrichment analysis was done for each DUB on all of its co-dependencies. This
        #  is the top reported GO term name
        'GO_enriched_in_codependencies',
        # p-value of GO gene set enrichment analysis
        'pvalue',
        # adjusted p-value of GO gene set enrichment analysis
        'p.adjust',
        # q-avalue of GO gene set enrichment analysis
        'qvalue',
        # The list of genes associated to the GO term
        'geneID',
    ]
    rv = {}
    for (dub, rnaseq, n_papers, fraction_dependent, go_term, go_p, go_padj, go_q, go_genes), sdf in df.groupby(gkey):
        # columns= [c for c in sdf.columns if c not in gkey]
        columns = [
            'DepMap_coDependency',
            'DepMap_correlation',
            # whether codependent gene is correlated with the DUB in terms of protein abundance
            'CCLE_Proteomics_correlation',
            'CCLE_Proteomics_p_adj',
            'CCLE_Proteomics_z_score',
            'CCLE_Proteomics_z_score_sig',
            'CMAP_Score',
            'CMAP_Perturbation_Type',
        ]
        # print(sdf[columns])
        dub_hgnc_id = hgnc_client.get_current_hgnc_id(dub)
        if dub_hgnc_id is None:
            raise
        dub_hgnc_symbol = hgnc_client.get_hgnc_name(dub_hgnc_id)

        if go_term == 'none significant':
            go = {}
        else:
            go_term = GO_FIXES.get(go_term, go_term)
            go_term_norm = go_term.removeprefix('GO_').replace('_', ' ').lower()
            go_id = go_client.get_go_id_from_label(go_term_norm)
            if go_id is None:
                matches = gilda.ground(go_term_norm)
                for match in matches:
                    if match.term.db == 'GO':
                        go_id = match.term.id
                        break
            if go_id is None:
                logger.warning(f'could not normalize: {go_term}')
                go_name = None
            else:
                go_name = go_client.get_go_label(go_id)

            # these are the genes in the GO term
            go_gene_dicts = []
            for go_gene in go_genes.strip().split('/'):
                go_gene_id = hgnc_client.get_current_hgnc_id(go_gene.strip())
                if go_gene_id is None:
                    raise
                go_gene_dicts.append({
                    'hgnc_id': go_gene_id,
                    'hgnc_name': hgnc_client.get_hgnc_name(go_gene_id),
                })
            go = dict(
                identifier=go_id,
                name=go_name,
                label=go_term,
                genes=go_gene_dicts,
                p=go_p,
                p_adj=go_padj,
                q=go_q,
            )

        depmap_results = []
        for gene, depmap_corr, ccle_corr, ccle_p_adj, ccle_z, ccle_z_sig, cmap_score, cmap_type in sdf[columns].values:
            gene = GENE_FIXES.get(gene, gene)
            depmap_gene_id = hgnc_client.get_current_hgnc_id(gene)
            if depmap_gene_id is None:
                logger.warning(f'could not map depmap dependency for {dub} to HGNC ID: {gene}')
                continue
            gene = hgnc_client.get_hgnc_name(depmap_gene_id)
            depmap_result = dict(
                hgnc_id=depmap_gene_id,
                hgnc_symbol=gene,
                correlation=depmap_corr,
            )
            if pd.notna(ccle_corr):
                depmap_result['ccle'] = dict(
                    correlation=ccle_corr,
                    p_adj=ccle_p_adj,
                    z=ccle_z,
                    significant=ccle_z_sig,
                )
            if pd.notna(cmap_score):
                depmap_result['cmap'] = dict(
                    score=cmap_score,
                    type=cmap_type,
                )
            depmap_results.append(depmap_result)

        rv[dub] = dict(
            hgnc_id=dub_hgnc_id,
            hgnc_symbol=dub_hgnc_symbol,
            rnaseq=rnaseq,
            papers=n_papers,
            fraction_cell_lines_dependent=fraction_dependent,
            go=go,
        )

    with DATA_PROCESED.open('w') as file:
        json.dump(rv, file, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

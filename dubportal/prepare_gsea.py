# -*- coding: utf-8 -*-

"""Run this script to prepare the GSEA data in the data/ directory."""

import pathlib

import pandas as pd
import pyobo

from indra.databases import go_client, hgnc_client

DATA = pathlib.Path(__file__).parent.resolve().joinpath("data")

GO_FIXES = {
    "GO_G1_DNA_DAMAGE_CHECKPOINT": "mitotic G1 DNA damage checkpoint signaling",
    "GO_HETEROTRIMERIC_G_PROTEIN_COMPLEX": "heterotrimeric G-protein complex",
    "GO_INNATE_IMMUNE_RESPONSE_ACTIVATING_CELL_SURFACE_RECEPTOR_SIGNALING_PATHWAY": "innate immune response activating cell surface receptor signaling pathway",
    "GO_NUCLEAR_TRANSCRIPTION_FACTOR_COMPLEX": "transcription regulator complex",
    "GO_PLASMA_MEMBRANE_RECEPTOR_COMPLEX": "plasma membrane signaling receptor complex",
    "GO_REGULATION_OF_ESTABLISHMENT_OF_PLANAR_POLARITY": "regulation of establishment of planar polarity",
    "GO_RNA_POLYMERASE_II_TRANSCRIPTION_FACTOR_COMPLEX": "RNA polymerase II-specific DNA-binding transcription factor binding",
    "GO_TETHERING_COMPLEX": "vesicle tethering complex",
    "GO_TRANSCRIPTION_FACTOR_COMPLEX": "transcription regulator complex",
    "GO_TRANSPORT_OF_VIRUS": "intracellular transport of virus",
}


def main() -> None:
    """Load the original results and output a cleaned and normalized file."""
    df = pd.read_csv(DATA.joinpath("sigGO_DepMap_SigCodependencies.tsv"), sep="\t")
    reverse_mapping = get_reverse_mapping()
    df["ID"] = df["ID"].map(lambda s: clean_go_prefix(GO_FIXES.get(s, s)))
    df["go_id"] = df["ID"].map(reverse_mapping.get)
    df["go_name"] = df["go_id"].map(go_client.get_go_label, na_action="ignore")
    df["hgnc_symbol"] = df["DUB"].map(
        lambda s: hgnc_client.get_hgnc_name(hgnc_client.get_current_hgnc_id(s))
    )
    del df["DUB"]
    del df["Description"]
    df = df[
        ["hgnc_id", "hgnc_symbol", "go_id", "go_name", "pvalue", "p.adjust", "qvalue"]
    ]

    # FIXME these IDs are not taking in the fixes from above
    df = df[df["go_id"].notnull()]
    # unmapped = sorted(set(df.loc[df['go_id'].isnull(), 'ID'].unique()))
    # for um in unmapped:
    #    print(um)

    df.to_csv(DATA.joinpath("gsea.json"), sep="\t", index=False)


def get_reverse_mapping() -> dict[str, str]:
    """Get a mapping from messy MSig names to GO identifiers."""
    msig_to_go = pyobo.get_filtered_xrefs("msig", "go")
    rv = {}
    for identifier, name in pyobo.get_id_name_mapping("msig").items():
        go_id = msig_to_go.get(identifier)
        if go_id is None:
            continue
        rv[clean_go_prefix(name)] = go_id
    return rv


def clean_go_prefix(s: str) -> str:
    return (
        s.removeprefix("GOBP_")
        .removeprefix("GO_")
        .removeprefix("GOCC_")
        .removeprefix("GOMF_")
    )


if __name__ == "__main__":
    main()

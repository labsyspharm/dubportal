import json
import pathlib
from typing import Optional

import pandas as pd
import pyobo

from indra.databases import hgnc_client

DATA_DIR = pathlib.Path(__file__).parent.resolve().joinpath("data")
PATH = DATA_DIR.joinpath("sigGSEA_DGE1_topKOs.tsv")
PROCESSED_JSON_PATH = DATA_DIR.joinpath("ko_gsea.json")
PROBLEMS_PATH = DATA_DIR.joinpath("ko_gsea_issues.tsv")
MSIG_NAMES = pyobo.get_name_id_mapping("msig")


def get_msig_id(name: str) -> Optional[str]:
    if name in MSIG_NAMES:
        return MSIG_NAMES[name]
    if name.startswith("GO_"):
        for prefix in ("GOBP_", "GOMF_", "GOCC_", ""):
            nn = prefix + name[len("GO_"):]
            if nn in MSIG_NAMES:
                return MSIG_NAMES[nn]


def main():
    df = pd.read_csv(PATH, sep="\t")
    df["hgnc_id"] = df["perturbation"].map(hgnc_client.get_current_hgnc_id)
    df["hgnc_symbol"] = df["hgnc_id"].map(hgnc_client.get_hgnc_name)
    df["msig_id"] = df["pathway"].map(get_msig_id)
    del df["cellline"]
    del df["cellline_treatment"]
    del df["perturbation"]
    del df["leadingEdge"]
    del df["size"]

    problems = df[df["msig_id"].isnull()]
    problems.to_csv(PROBLEMS_PATH, sep="\t", index=False)

    # TODO are there better ways to map? Spot check seems like a versioning issue
    df = df[df["msig_id"].notnull()]
    p, i, n = zip(*(_get(row) for _, row in df.iterrows()))
    df["prefix"] = p
    df["identifier"] = i
    df["name"] = n
    del df["pathway"]

    rv = {
        hgnc_symbol: [
            {key: value for key, value in record.items() if pd.notnull(value)}
            for record in sdf.sort_values("padj").to_dict("records")
        ]
        for hgnc_symbol, sdf in df.groupby("hgnc_symbol")
    }
    with PROCESSED_JSON_PATH.open("w") as file:
        json.dump(rv, file, indent=2, sort_keys=True)


KEYS = [
    "reactome",
    "go",
    "kegg.pathway",
]


def _get(row: dict[str, any]) -> tuple[str, str, str]:
    msig_id = row["msig_id"]
    for prefix in KEYS:
        identifier = pyobo.get_xref("msig", msig_id, prefix)
        if identifier:
            identifier = (
                identifier.removeprefix(prefix)
                    .removeprefix(prefix.upper())
                    .removeprefix(":")
            )
            return prefix, identifier, pyobo.get_name(prefix, identifier)
    return "msig", msig_id, row["pathway"]


if __name__ == "__main__":
    main()

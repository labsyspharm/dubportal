import json
import pathlib

import pandas as pd
import pyobo

from indra.databases import hgnc_client
import bioregistry
DATA_DIR = pathlib.Path(__file__).parent.resolve().joinpath('data')
PATH = DATA_DIR.joinpath('sigGSEA_DGE1_topKOs.tsv')
PROCESSED_JSON_PATH = DATA_DIR.joinpath('ko_gsea.json')
PROBLEMS_PATH = DATA_DIR.joinpath('ko_gsea_issues.tsv')

MSIG_NAMES = pyobo.get_name_id_mapping('msig')


def mmsig_map(name: str):
    if name in MSIG_NAMES:
        return MSIG_NAMES[name]
    if name.startswith('GO_'):
        for prefix in ('GOBP_', 'GOMF_', 'GOCC_', ''):
            nn = prefix + name[len('GO_'):]
            if nn in MSIG_NAMES:
                return MSIG_NAMES[nn]
    print('could not map', name)


def main():
    df = pd.read_csv(PATH, sep='\t')
    df['hgnc_id'] = df['perturbation'].map(hgnc_client.get_current_hgnc_id)
    df['hgnc_symbol'] = df['hgnc_id'].map(hgnc_client.get_hgnc_name)
    df['msig_id'] = df['pathway'].map(mmsig_map)
    del df['cellline']
    del df['cellline_treatment']
    del df['perturbation']
    del df['leadingEdge']
    del df['size']

    problems = df[df['msig_id'].isnull()]
    problems.to_csv(PROBLEMS_PATH, sep='\t', index=False)

    # TODO are there better ways to map? Spot check seems like a versioning issue
    df = df[df['msig_id'].notnull()]
    p, i, n = zip(*(
        _get(row)
        for _, row in df.iterrows()
    ))
    df['prefix'] = p
    df['identifier'] = i
    df['name'] = n
    del df['pathway']

    j = {
        hgnc_symbol: [
            {k: v for k, v in r.items() if pd.notnull(v)}
            for r in sdf.sort_values('padj').to_dict('records')
        ]
        for hgnc_symbol, sdf in df.groupby('hgnc_symbol')
    }
    with PROCESSED_JSON_PATH.open('w') as file:
        json.dump(j, file, indent=2, sort_keys=True)


KEYS = [
    'reactome',
    'go',
    # 'kegg.pathway',
]


def _get(row):
    msig_id = row['msig_id']
    for prefix in KEYS:
        identifier = pyobo.get_xref('msig', msig_id, prefix)
        if identifier:
            identifier = identifier.removeprefix(prefix).removeprefix(prefix.upper()).removeprefix(':')
            return prefix, identifier, pyobo.get_name(prefix, identifier)
    return 'msig', msig_id, row['pathway']


if __name__ == '__main__':
    main()

import json
import pickle
import pandas
import logging
from indra.tools import assemble_corpus as ac
from indra.belief import BeliefEngine
from indra.databases import hgnc_client
from indra.sources.indra_db_rest import get_statements
from indra_db.client.principal.curation import get_curations
from indra.databases import ndex_client
from indra.assemblers.cx import CxAssembler
from indra.assemblers.cx.hub_layout import add_semantic_hub_layout

from .api import DATA


curs = get_curations()

logger = logging.getLogger(__name__)


def get_statements_for_dub(dub):
    logger.info('Getting statements for %s' % dub)
    hgnc_id = hgnc_client.get_current_hgnc_id(dub)
    if hgnc_id is None:
        logger.warning('Could not get HGNC ID for %s' % dub)
        return None
    ip = get_statements(agents=['%s@HGNC' % hgnc_id],
                        ev_limit=10000)
    stmts = filter_out_medscan(ip.statements)
    stmts = ac.filter_by_curation(stmts, curs)
    stmts = sorted(stmts, key=lambda x: len(x.evidence), reverse=True)
    return stmts


def filter_out_medscan(stmts):
    logger.info('Starting medscan filter with %d statements' % len(stmts))
    new_stmts = []
    for stmt in stmts:
        new_evidence = []
        for ev in stmt.evidence:
            if ev.source_api == 'medscan':
                continue
            new_evidence.append(ev)
        stmt.evidence = new_evidence
        if new_evidence:
            new_stmts.append(stmt)
    logger.info('Finished medscan filter with %d statements' % len(new_stmts))
    return new_stmts


def assemble_statements(stmts):
    """Run assembly steps on statements."""
    be = BeliefEngine()
    for kinase, kinase_stmts in stmts.items():
        stmts[kinase] = ac.filter_human_only(kinase_stmts)
        be.set_prior_probs(stmts[kinase])
    return stmts


def upload_network(dub, stmts):
    cxa = CxAssembler(stmts)
    cxa.make_model()
    add_semantic_hub_layout(cxa.cx, dub)
    model_id = cxa.upload_model()
    ndex_client.set_style(model_id)
    ndex_client.add_to_network_set(model_id,
                                   '381f3a8d-cac9-11eb-9a85-0ac135e8bacf')
    return model_id


if __name__ == '__main__':
    df = pandas.read_csv(DATA, sep='\t')
    dubs = sorted(df['DUB'].unique())
    network_index = {}
    all_stmts = {}
    for dub in dubs:
        stmts = get_statements_for_dub(dub)
        network_id = upload_network(dub, stmts)
        network_index[dub] = network_id
        all_stmts[dub] = stmts

    with open('network_index.json', 'w') as fh:
        json.dump(network_index, fh)
    with open('statements.pkl', 'wb') as fh:
        pickle.dump(all_stmts, fh, protocol=5)
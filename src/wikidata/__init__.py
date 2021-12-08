# -*- coding: utf-8 -*-

import utils
from src.wikidata import neighbors_parser as nb_parser


def get_k_hop_neighbors(k: int) -> set:
    """ Retrieves the set of k-hop neigbors from Wikidata for the Wikidata resources in the graph. 
    """
    utils.get_logger().info(f'Wikidata: Loading {k}-hop neighboring entities into memory..')

    global __WIKI_NEIGHBORS__
   
    if '__WIKI_NEIGHBORS__' not in globals():
        if k == 1:
            wiki_attributes_map = utils.load_cache('wiki_attributes_map')
            entities = wiki_attributes_map.keys()
        else:
            if not utils.load_cache('wiki_' + str(k-1) + '_hop_neighbors_list'):
                raise Exception(f'{k-1} neighbors need to be computed first.')
            
            triples = utils.load_cache('wiki_' + str(k-1) + '_hop_neighbors_list')                
            entities = set([triple[2] for triple in triples])
        
        initializer = lambda: nb_parser.retrieve_wikidata_neighbors(list(entities))
        __WIKI_NEIGHBORS__ = utils.load_or_create_cache('wiki_' + str(k) + '_hop_neighbors_list', initializer)

    utils.get_logger().info('Wikidata: Loaded.\n')

    return __WIKI_NEIGHBORS__ 

def _clear_global_var() -> None:
    """ 
    Clears the global variable storing k-hop neighbors. 
    Necessary when k changes during the same run, in order to allow the (k+1)-hop neighbors to be computed.
    """
    global __WIKI_NEIGHBORS__
    del __WIKI_NEIGHBORS__

# -*- coding: utf-8 -*-

import utils
from tqdm import tqdm
from collections import defaultdict 
import src.news_kg.util as news_kg_util
import src.wikidata.query as wiki_query
from qwikidata.linked_data_interface import get_entity_dict_from_api

def retrieve_wikidata_neighbors(entities: list) -> set:
    """ Retrieves all neighbors representing entities 
    from Wikidata for the given Wikidata entities set.
    """
    utils.get_logger().debug(f'Wikidata: Retrieving neighbors from Wikidata for {len(entities)} entities.')

    # Load or create the map of Wikidata entities to neighbor entities
    if utils.load_cache('wiki_neighbors_map') is None:
        wiki_neighbors_map = initialize_wiki_neighbors_map()
    else:
        wiki_neighbors_map = utils.load_cache('wiki_neighbors_map')

    # Load map of Wikidata attributes
    wiki_attributes_map = utils.load_cache('wiki_attributes_map')

    # Retrieve neighbors
    triples = set()
    steps = 0
    for entity in tqdm(entities):
        steps += 1

        if entity in wiki_neighbors_map.keys():
            # If entity is already in the map directly retrieve its stored neighbors
            triples.update(wiki_neighbors_map[entity])
        else:
            # Entity is not in the map, so query Wikidata for its neighbors
            try:
                data = get_entity_dict_from_api(entity.split('/')[-1])
            except Exception:
                # Catches errors thrown for entities that no longer appear in Wikidata
                continue 
            claims = data['claims']
            properties = [prop for prop in data['claims']]
            for prop in properties:
                for item in claims[prop]:
                    if (('datavalue' in item['mainsnak']) and (type(item['mainsnak']['datavalue']['value'])==dict) and ('id' in item['mainsnak']['datavalue']['value'])):
                        triple = (entity, news_kg_util.pid2wikidata_property(prop), news_kg_util.qid2wikidata_resource(item['mainsnak']['datavalue']['value']['id']))
                        triples.add(triple)
                        wiki_neighbors_map[entity].add(triple)
            
            # Update the Wiki attributes map
            if not entity in wiki_attributes_map.keys():
                label, aliases = wiki_query.get_entity_attributes(data)
                wiki_attributes_map[entity]['label'] = label
                wiki_attributes_map[entity]['aliases'] = aliases

        # Cache Wiki maps ever 1000 steps
        if steps % 1000 == 0:
            utils.update_cache('wiki_neighbors_map', wiki_neighbors_map)
            utils.update_cache('wiki_attributes_map', wiki_attributes_map)

    utils.get_logger().debug(f'Retrieved {len(triples)} neighbors.')
    utils.get_logger().debug(f'Size of Wiki attributes map: {len(wiki_attributes_map)}.')

    # Update caches of Wiki map
    utils.update_cache('wiki_neighbors_map', wiki_neighbors_map)
    utils.update_cache('wiki_attributes_map', wiki_attributes_map)

    return list(triples)


def initialize_wiki_neighbors_map() -> dict:
    initializer = lambda: defaultdict(set)
    return utils.load_or_create_cache('wiki_neighbors_map', initializer)

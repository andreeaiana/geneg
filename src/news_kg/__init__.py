# -*- coding: utf-8 -*-

import utils
from src.news_kg.graph import GeNeG
import src.news_kg.serialize as news_kg_serialize


def get_base_graph() -> GeNeG:
    """ Retrieves the base graph created from the news dataset. """
    utils.get_logger().info('GeNeG: Loading base knowledge graph into memory..')
    global __BASE_GRAPH__
    if '__BASE_GRAPH__' not in globals():
        initializer = lambda: GeNeG.build_base_graph()
        __BASE_GRAPH__ = utils.load_or_create_cache('geneg_base', initializer)
    utils.get_logger().info('GeNeG: Loaded.\n')
    return __BASE_GRAPH__


def get_entities_graph() -> GeNeG:
    """ Retrieves entities graph created from news and entities from base graph, enriched with k-hop neighbours from Wikidata. """
    utils.get_logger().info('GeNeG: Loading entities knowledge graph into memory..')
    global __ENTITIES_GRAPH__
    if '__ENTITIES_GRAPH__' not in globals():
        initializer = lambda: get_base_graph().construct_entities_graph()
        __ENTITIES_GRAPH__ = utils.load_or_create_cache('geneg_entities', initializer)
    utils.get_logger().info('GeNeG: Loaded.\n')
    return __ENTITIES_GRAPH__

def get_complete_graph() -> GeNeG:
    """ Retrieves the base graph enriched with k-hop neighbours from Wikidata. """
    utils.get_logger().info('GeNeG: Loading the complete graph into memory...')
    global __COMPLETE_GRAPH__
    if '__COMPLETE_GRAPH__' not in globals():
        initializer = lambda: get_base_graph().construct_complete_graph()
        __COMPLETE_GRAPH__ = utils.load_or_create_cache('geneg_complete', initializer)
    utils.get_logger().info('GeNeG: Loaded.\n')
    return __COMPLETE_GRAPH__

def serialize_final_graph(graph_type: str):
    """ Serializes the final GeNeG. """
    if graph_type == 'base':
        base_graph = get_base_graph()
        news_kg_serialize.serialize_base_graph(base_graph)  
    elif graph_type == 'entities':
        entities_graph = get_entities_graph()
        news_kg_serialize.serialize_entities_graph(entities_graph)
    elif graph_type == 'complete':
        complete_graph = get_complete_graph()
        news_kg_serialize.serialize_complete_graph(complete_graph)
    else:
        raise Exception(f'Type of graph "{graph_type}" unknown.')

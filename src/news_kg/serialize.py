# - coding: utf-8 -*-

""" Functionality to serialize the individual parts of GeNeG. """

import bz2
import utils
from typing import List
import src.util.rdf as rdf_util
import src.news_kg.util as news_kg_util
import src.util.serialize as serialize_util
from src.news_kg.graph import GeNeG


def serialize_base_graph(graph: GeNeG):
    """ Serialize the complete base graph as individual files. """
    utils.get_logger().info('GeNeG: Serializing the base graph as individual files..')

    _write_lines_to_file(_get_lines_metadata(graph), 'results.geneg_base.metadata')
    _write_lines_to_file(_get_lines_instance_types(graph), 'results.geneg_base.instances_types')
    _write_lines_to_file(_get_lines_instances_metadata_literals(graph), 'results.geneg_base.instances_metadata_literals')
    _write_lines_to_file(_get_lines_instances_metadata_resources(graph), 'results.geneg_base.instances_metadata_resources')
    _write_lines_to_file(_get_lines_instances_content_relations(graph), 'results.geneg_base.instances_content_relations')
    _write_lines_to_file(_get_lines_instances_event_mapping(graph), 'results.geneg_base.instances_event_mapping')
    _write_lines_to_file(_get_lines_event_relations(graph), 'results.geneg_base.event_relations')

    utils.get_logger().info('GeNeG: Completed serialization.\n')

def serialize_entities_graph(graph: GeNeG):
    """ Serialize the complete entity graph as individual files. """
    utils.get_logger().info('GeNeG: Serializing the entity graph as individual files..')

    _write_lines_to_file(_get_lines_metadata(graph), 'results.geneg_entities.metadata')
    _write_lines_to_file(_get_lines_instance_types(graph), 'results.geneg_entities.instances_types')
    _write_lines_to_file(_get_lines_instances_metadata_resources(graph), 'results.geneg_entities.instances_metadata_resources')
    _write_lines_to_file(_get_lines_instances_event_mapping(graph), 'results.geneg_entities.instances_event_mapping')
    _write_lines_to_file(_get_lines_event_relations(graph), 'results.geneg_entities.event_relations')
    _write_lines_to_file(_get_lines_wiki_relations(graph), 'results.geneg_entities.wiki_relations')

    utils.get_logger().info('GeNeG: Completed serialization.\n')


def serialize_complete_graph(graph: GeNeG):
    """ Serialize the complete graph as individual files. """
    utils.get_logger().info('GeNeG: Serializing the complete graph as individual files..')

    _write_lines_to_file(_get_lines_metadata(graph), 'results.geneg_complete.metadata')
    _write_lines_to_file(_get_lines_instance_types(graph), 'results.geneg_complete.instances_types')
    _write_lines_to_file(_get_lines_instances_metadata_literals(graph), 'results.geneg_complete.instances_metadata_literals')
    _write_lines_to_file(_get_lines_instances_metadata_resources(graph), 'results.geneg_complete.instances_metadata_resources')
    _write_lines_to_file(_get_lines_instances_content_relations(graph), 'results.geneg_complete.instances_content_relations')
    _write_lines_to_file(_get_lines_instances_event_mapping(graph), 'results.geneg_complete.instances_event_mapping')
    _write_lines_to_file(_get_lines_event_relations(graph), 'results.geneg_complete.event_relations')
    _write_lines_to_file(_get_lines_wiki_relations(graph), 'results.geneg_complete.wiki_relations')

    utils.get_logger().info('GeNeG: Completed serialization.\n')


def _write_lines_to_file(lines: List, filepath_config: str) -> None:
    filepath = utils.get_results_file(filepath_config)
    with bz2.open(filepath, mode='wt') as f:
        f.writelines(lines)


def _get_lines_metadata(graph: GeNeG) -> List:
    """ Serialize metadata. """
    utils.get_logger().debug('GeNeG: Serializing metadata..')
    
    void_resource = 'http://geneg.net/.well-known/void'
    description = 'The GeNeG is a knowledge graph built using a German news dataset collected from 39 media outlets and enriched with additional resources extracted from Wikidata.'
    entity_count = len(graph.get_geneg_resources()) + len(graph.get_wikidata_resources())
    property_count = len(graph.get_all_properties())
    return [
            serialize_util.as_object_triple(void_resource, rdf_util.PREDICATE_TYPE, 'http://rdfs.org/ns/void#Dataset'),
            serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/elements/1.1/title', 'GeNeG'),
            serialize_util.as_literal_triple(void_resource, rdf_util.PREDICATE_LABEL, 'GeNeG'),
            serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/elements/1.1/description', description),
            serialize_util.as_object_triple(void_resource, 'http://purl.org/dc/terms/license', 'https://opensource.org/licenses/MIT'),
            serialize_util.as_object_triple(void_resource, 'http://purl.org/dc/terms/license', 'http://creativecommons.org/licenses/by-sa/3.0/'),
            serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/terms/creator', 'Andreea Iana, Heiko Paulheim'),
            serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/terms/created', _get_creation_date()),
            serialize_util.as_literal_triple(void_resource, 'http://purl.org/dc/terms/publisher', 'Andreea Iana'),
            serialize_util.as_literal_triple(void_resource, 'http://rdfs.org/ns/void#uriSpace', news_kg_util.NAMESPACE_RENEWRS_RESOURCE),
            serialize_util.as_literal_triple(void_resource, 'http://rdfs.org/ns/void#entities', entity_count),
            serialize_util.as_literal_triple(void_resource, 'http://rdfs.org/ns/void#properties', property_count),
#            serialize_util.as_object_triple(void_resource, 'http://xmlns.com/foaf/0.1/homepage', 'http://geneg.net'),
#            serialize_util.as_object_triple(void_resource, 'http://rdfs.org/ns/void#sparqlEndpoint', 'http://geneg.net/sparql'),
   ]


def _get_creation_date() -> str: 
    return utils.get_config('geneg.creation_date')


def _get_lines_instance_types(graph: GeNeG) -> List:
    """ Serialize types of resources. """
    utils.get_logger().debug('GeNeG: Serializing instance types..')
    
    list_instances_types = list()

    for res, res_type in graph.get_edges_for_property(rdf_util.PREDICATE_TYPE):
        list_instances_types.append(serialize_util.as_object_triple(res, rdf_util.PREDICATE_TYPE, res_type))
    return list_instances_types


def _get_lines_instances_metadata_literals(graph: GeNeG) -> List:
    """ Serialize metadata facts containing literals (i.e. url, dates, polarity). """
    utils.get_logger().debug('GeNeG: Serializing literal metadata facts..')

    properties = [rdf_util.PREDICATE_URL, rdf_util.PREDICATE_DATE_PUBLISHED, rdf_util.PREDICATE_DATE_MODIFIED, rdf_util.PREDICATE_POLARITY]
    return _get_lines_instances_relations(graph, properties)


def _get_lines_instances_metadata_resources(graph: GeNeG) -> List:
    """ Serialize metadata facts containing resources (i.e. publisher, author, keywords). """
    utils.get_logger().debug('GeNeG: Serializing resource metadata facts..')
    
    properties = [rdf_util.PREDICATE_PUBLISHER, rdf_util.PREDICATE_AUTHOR, rdf_util.PREDICATE_KEYWORDS]
    return _get_lines_instances_relations(graph, properties)


def _get_lines_instances_content_relations(graph: GeNeG) -> List:
    """ Serialize resource content facts (i.e. title, abstract, article body). """
    utils.get_logger().debug('GeNeG: Serializing resource content facts..')

    properties = [rdf_util.PREDICATE_HEADLINE, rdf_util.PREDICATE_ABSTRACT, rdf_util.PREDICATE_ARTICLE_BODY]
    return _get_lines_instances_relations(graph, properties)


def _get_lines_instances_event_mapping(graph: GeNeG) -> List:
    """ Serialize event mapping for news article resources. """
    utils.get_logger().debug('GeNeG: Serializing mapping of news article resources to events..')
    article_nodes = graph.get_article_nodes()
    event_nodes = graph.get_event_nodes()

    instance_event_mappings = list()
    for res, event in graph.get_edges_for_property(rdf_util.PREDICATE_ABOUT):
        if event in event_nodes:
            instance_event_mappings.append(serialize_util.as_object_triple(res, rdf_util.PREDICATE_ABOUT, event))

    return instance_event_mappings


def _get_lines_event_relations(graph: GeNeG) -> List:
    """ Serialize facts about event resources. """
    utils.get_logger().debug('GeNeG: Serializing event resource facts..')

    properties = [rdf_util.PREDICATE_HAS_ACTOR, rdf_util.PREDICATE_HAS_PLACE, rdf_util.PREDICATE_MENTIONS]
    return _get_lines_instances_relations(graph, properties)


def _get_lines_wiki_relations(graph: GeNeG) -> List:
    """ Serialize facts containing relations from Wikidata. """
    utils.get_logger().debug('GeNeG: Serializing Wiki resource facts..')

    properties = [prop for prop in graph.get_all_properties() if 'wiki' in prop]
    return _get_lines_instances_relations(graph, properties)


def _get_lines_instances_relations(graph: GeNeG, properties: List) -> List:
    """ Serialize resource facts for the given properties. """
    lines_instance_relations = list()
    instances_relations = set()

    for prop in properties:
        instances_relations.update({(res, prop, val) for (res, val) in graph.get_edges_for_property(prop)})

    for s, p, o in instances_relations:
        if news_kg_util.is_geneg_resource(o) or news_kg_util.is_wikidata_resource(o):
            lines_instance_relations.append(serialize_util.as_object_triple(s, p, o))
        else:
            lines_instance_relations.append(serialize_util.as_literal_triple(s, p, o))
    return lines_instance_relations    

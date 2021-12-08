# -*- coding: utf-8 -*-

import uuid
import utils
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from operator import itemgetter
from thefuzz import process, fuzz
from difflib import SequenceMatcher
from typing import List, Set, Dict
from collections import defaultdict
from typing import Dict, Union, Tuple, List

import src.util.rdf as rdf_util
import src.news_kg.nlp as nlp_util
import src.news_kg.util as news_kg_util
from src.util.base_graph import BaseGraph
import src.util.data_loading as data_loading
import src.wikidata.query as wiki_query
import src.wikidata as wiki_util


class GeNeG(BaseGraph):
    """ A knowledge graph built from German news dataset. """

    # initialisations
    def __init__(self, graph: nx.MultiDiGraph) -> None:
        super().__init__(graph)
        self.label2node_map = defaultdict(set)
        self.alias2node_map = defaultdict(set)

        wiki_map_initializer = lambda: defaultdict(dict)
        self.wiki_attributes_map = utils.load_or_create_cache('wiki_attributes_map', wiki_map_initializer)

    # Nodes attribute definitions
    ATTRIBUTE_LABEL = 'label'
    ATTRIBUTE_ALIAS = 'alias'

    def _check_node_exists(self, node: str) -> None:
        if not self.has_node(node):
            raise Exception(f'Node {node} not in graph')

    def get_node_label(self, node: str) -> str:
        """ Returns the label of a node. """
        if not (news_kg_util.is_geneg_resource(node) or news_kg_util.is_wikidata_resource(node)):
            raise Exception(f'Node {node} is not a resource')
        self._check_node_exists(node)
        return self._get_attr(node, self.ATTRIBUTE_LABEL)

    def set_node_label(self, node: str, label: str) -> None:
        """ Sets the label of a node. """
        if not (news_kg_util.is_geneg_resource(node) or news_kg_util.is_wikidata_resource(node)):
            raise Exception(f'Node {node} is not a resource')
        if label == None:
            self._set_attr(node, self.ATTRIBUTE_LABEL, label)
        else:
            if not len(label) == 1:
                self._set_attr(node, self.ATTRIBUTE_LABEL, label)

    def get_node_alias(self, node: str) -> Set[str]:
        """ Returns the alias(es) of a node. """
        if not (news_kg_util.is_geneg_resource(node) or news_kg_util.is_wikidata_resource(node)):
            raise Exception(f'Node {node} is not a resource')
        self._check_node_exists(node)
        return self._get_attr(node, self.ATTRIBUTE_ALIAS)

    def set_node_alias(self, node: str, alias: str) -> None:
        """ Adds a new alias for an existing node. """
        # Update list of aliases with given value, if attribute already set
        if not len(alias) == 1:
            if self._get_attr(node, self.ATTRIBUTE_ALIAS):
                self._update_attr(node, self.ATTRIBUTE_ALIAS, alias)
            else:
                # Otherwise initialize alias list
                self._set_attr(node, self.ATTRIBUTE_ALIAS, set())
                self._update_attr(node, self.ATTRIBUTE_ALIAS, alias)

    def remove_node_alias(self, node: str, alias: str) -> None:
        """ Removes an alias from the alias list of a node. """
        if not alias in self._get_attr(node, self.ATTRIBUTE_ALIAS):
            raise Exception(f'Node {node} does not have the alias {alias}')
        self._remove_attr_val(node, self.ATTRIBUTE_ALIAS, alias)

    def _set_nodes_attributes(self, nodes_w_attributes: Dict[str, Union[str, List[str]]], set_label: bool, set_alias: bool) -> None:
        """ Sets the attributes(i.e. label, alias) for a list of nodes. 

        Modifies the node variable in place.

        Args:
            nodes_w_attributes (dict): A dictionary of nodes with a dictionary of attributes, e.g. {node: {attr_1:value}, {attr_2: value}} 
            set_label (bool): Whether to set the label of the node.
            set_alias (bool): Whether to set the alias of the node
        """
        for node in list(nodes_w_attributes.keys()):
            self._check_node_exists(node)
            if set_label:
                self.set_node_label(node, nodes_w_attributes[node][self.ATTRIBUTE_LABEL])
            if set_alias:
                self.set_node_alias(node, nodes_w_attributes[node][self.ATTRIBUTE_ALIAS])

    def get_edges_by_key(self, key: str) -> Set[Tuple[str, str]]:
        """ Returns all edges with the given key contained in the graph. """
        edges = [(u, v) for u, v, k in self.get_edges(keys=True) if k==key]
        return set(edges)
    
    def get_geneg_resources(self) -> Set[str]:
        """ Returns all the GeNeG resources contained in the graph. """
        geneg_resources = [node for node in self.nodes if news_kg_util.is_geneg_resource(node)]
        return set(geneg_resources)

    def get_wikidata_resources(self) -> Set[str]:
        """ Returns all entities linked to Wikidata. """
        wikidata_resources = [node for node in self.nodes if news_kg_util.is_wikidata_resource(node)]
        return set(wikidata_resources)
    
    def get_unlinked_resources(self) -> Set[str]:
        """ Returns any resource that is not linked to Wikidata and that does not represent an article or event node. """
        resources = self.get_geneg_resources()
        articles = self.get_article_nodes()
        events = self.get_event_nodes()
        return set(resources).difference(set(articles).union(set(events)))

    def get_article_nodes(self) -> List[str]:
        """ Returns the ids of all nodes representing a GeNeG resource corresponding to an article. """
        article_nodes = [node for node in self.get_geneg_resources() if len(node)==43 and 'news_' in node and not '_evt' in node]
        return article_nodes

    def get_event_nodes(self) -> List[str]:
        """ Returns the ids of all nodes representing a GeNeG resource corresponding to an event. """
        event_nodes = [node for node in self.get_geneg_resources() if len(node)==47 and 'news_' in node and '_evt' in node]
        return event_nodes
    
    def get_all_properties(self) -> Set[str]:
        """ Returns all properties used in GeNeG. """
        properties = [k for _, _,k in self.get_edges(keys=True)]
        return set(properties)

    def get_properties_for_node(self, node: str) -> Set[str]:
        """ Returns all properties that the given node has (i.e. given node is the source node of the edge). """
        properties = [k for u, v, k in self.get_edges(keys=True) if u==node]
        return set(properties)

    def get_subjects_for_property(self, prop: str) -> Set[str]:
        """ Returns all subject nodes that have the given property (i.e. all source nodes for edges of type 'prop'). """
        source_nodes = [u for u, v, k in self.get_edges(keys=True) if k==prop]
        return set(source_nodes)

    def get_objects_for_property(self, prop: str) -> Set[str]:
        """ Returns all objects of a given property (i.e. all target nodes for edges of type 'prop'). """
        objects = [v for u, v, k in self.get_edges(keys=True) if k==prop]
        return set(objects)

    def get_edges_for_property(self, prop: str) -> Set[Tuple[str, str]]:
        """ Returns all pairs of nodes sharing an edge of type 'prop'. """
        nodes = [(u, v) for u, v, k in self.get_edges(keys=True) if k==prop]
        return set(nodes)

    def get_subjects(self, prop: str, target_node: str) -> Set[str]:
        """ Returns all source nodes for the given property and target node. """
        source_nodes = [u for u, v, k in self.get_edges(keys=True) if k==prop and v==target_node]
        return set(source_nodes)

    def get_objects(self, prop: str, source_node: str) -> Set[str]:
        """ Returns all target nodes for the given property and source node. """
        target_nodes = [v for u, v, k in self.get_edges(keys=True) if k==prop and u==source_node]
        return set(target_nodes)

    def get_source_for_target(self, target_node: str) -> Set[str]:
        """ Returns, for the given target node, all source nodes for all properties. """
        source_nodes = [u for u, v, _ in self.get_edges(keys=True) if v==target_node]
        return set(source_nodes)

    def get_target_for_source(self, source_node: str) -> Set[str]:
        """ Returns, for the given source node, all target nodes for all properties. """
        target_nodes = [v for u, v, _ in self.get_edges(keys=True) if u==source_node]
        return set(target_nodes)

    def remove_subjecs_for_property(self, prop: str) -> None:
        """ Removes all source nodes with an edge of type 'prop' from the graph. """
        remove_nodes = self.get_subjects_for_property(prop)
        self._remove_nodes(remove_nodes)

    def remove_edges_for_property(self, prop: str) -> None:
        """ Removes all edges of type 'prop' from the graph. """
        remove_edges = self.get_edges_for_property(prop)
        self._remove_edges(remove_edges)

    def _retrieve_wiki_attributes(self, qid: str) -> None:
        """ Retrives the label and alias for a Wikidata QID. """
        try:
            attr_dict = wiki_query.get_attr_dict(qid)
            label, aliases = wiki_query.get_entity_attributes(attr_dict)
            self.wiki_attributes_map[qid]['label'] = label
            self.wiki_attributes_map[qid]['aliases'] = aliases
        except Exception:
            # Catches errors thrown for entities that no longer appear in Wikidata
            utils.get_logger().debug(f'Entity {qid} cannot be found in Wikidata.')
        
        
    def get_wikidata_label(self, qids: list) -> List[str]:
        """ Returns the Wikidata labels of the given entity QIDs, either from the cached map, or by querying Wikidata.
        """
        # Retrieve labels and aliases of Wikidata QIDs that have not been yet mapped
        non_mapped_qids = list(set([qid for qid in qids if not qid in self.wiki_attributes_map.keys()]))
        utils.get_logger().debug(f'GeNeG: Retrieving attributes for {len(non_mapped_qids)} Wikidata entities.')
        for qid in tqdm(non_mapped_qids):
            self._retrieve_wiki_attributes(qid)

        labels = [self.wiki_attributes_map[qid]['label'] for qid in qids]
        return labels

    @property
    def properties_frequencies(self) -> Dict[str, float]:
        """ Returns the frequencies of all types of properties in the graph. """
        properties = self.get_all_properties()
        properties_frequencies = dict()
        for property_url in properties:
            if 'schema' in property_url:
                prop = 'schema:' + property_url.split('/')[-1]
            elif 'sem' in property_url:
                prop = property_url.split('#')[-1]
            elif 'geneg' in property_url:
                prop = 'geneg:' + property_url.split(news_kg_util.NAMESPACE_RENEWRS_PROPERTY)[-1]
            elif 'wiki/Property' in property_url:
                prop = 'wiki:' + property_url.split(':')[-1]
            else:
                prop = 'rdf:type'
            properties_frequencies[prop] = len(self.get_edges_by_key(property_url)) / len(self.edges)

        return properties_frequencies

    def get_property_frequency(self, property_url: str) -> float: 
        """ Returns the frequency of 'property'. """
        if 'schema' in property_url:
            prop = 'schema:' + property_url.split('/')[-1]
        elif 'sem' in property_url:
            prop = property_url.split('#')[-1]
        elif 'geneg' in property_url:
            prop = 'geneg:' + property_url.split(news_kg_util.NAMESPACE_RENEWRS_PROPERTY)[-1]
        elif 'wiki/Property' in property_url:
            prop = 'wiki:' + property_url.split(':')[-1]
        else:
            prop = 'rdf:type'
        return self.properties_frequencies[prop]

    @property
    def nodes_frequencies(self) -> Dict[str, float]:
        """ Returns the frequencies of all types of nodes in the graph. """
        properties = self.get_all_properties()
        properties.remove('https://schema.org/isBasedOn') # This only connects existing article nodes
        nodes_frequencies = dict()
        for property_url in properties:
            if 'schema' in property_url:
                prop = 'schema:' + property_url.split('/')[-1]
            elif 'sem' in property_url:
                prop = property_url.split('#')[-1]
            elif 'geneg' in property_url:
                prop = property_url.split(news_kg_util.NAMESPACE_RENEWRS_PROPERTY)[-1]
            elif 'wiki/Property' in property_url:
                prop = 'wiki:' + property_url.split(':')[-1]
            else:
                prop = 'rdf:type'
            nodes_frequencies[prop] = len(self.get_objects_for_property(property_url)) / len(self.nodes)

        article_nodes = self.get_article_nodes()
        event_nodes = self.get_event_nodes() 

        nodes_frequencies['articles'] = len(article_nodes) / len(self.nodes)
        nodes_frequencies['events'] = len(event_nodes) / len(self.nodes)

        return nodes_frequencies

    @property
    def statistics(self) -> str:
        # Average degree
        avg_in_degree = np.mean([deg for _, deg in self.in_degree()])
        avg_out_degree = np.mean([deg for _, deg in self.out_degree()])

        # Frequency of different types of nodes
        geneg_resources = self.get_geneg_resources()
        articles = self.get_article_nodes()
        events = self.get_event_nodes()
        other_geneg_resources = self.get_unlinked_resources()
        wikidata_resources = self.get_wikidata_resources()

        count_resources = len(geneg_resources) + len(wikidata_resources)
        count_literals = len(self.nodes) - count_resources

        return '\n'.join([
            '{:^43}'.format('STATISTICS'),
            '=' * 43,
            '\n',
            '{:^43}'.format('General'),
            '-' * 43,
            '{:<30} | {:>10}'.format('nodes', self.number_of_nodes()),
            '{:<30} | {:>10}'.format('edges', self.number_of_edges()),
            '{:<30} | {:>10}'.format('average in-degree', round(avg_in_degree, 4)),
            '{:<30} | {:>10}'.format('average out-degree', round(avg_out_degree, 4)),
            '\n',

            '-' * 43,
            '{:^43}'.format('Nodes'),
            '-' * 43,

            '{:<30} | {:>10}'.format('All nodes', len(self.nodes)),
            '\n',
            '{:<30}   {:>10}'.format('', '-frequency-'),
            '  {:<28} | {:>10}'.format('resources', round(count_resources/len(self.nodes), 4)),
            '  {:<28} | {:>10}'.format('literals', round(count_literals/len(self.nodes), 4)),

            '\n',
            '{:<30}   {:>10}'.format('', '-frequency-'),
            '\n'.join("  {:<28} | {:>10}".format(k, round(v, 4)) for k, v in sorted(self.nodes_frequencies.items(), key=lambda item: item[1])),
            '\n',

            '{:<30} | {:>10}'.format('Resources', count_resources),
            '\n',
            '{:<30}   {:>10}'.format('', '-frequency-'),
            '  {:<28} | {:>10}'.format('Articles', round(len(articles) / count_resources, 4)),
            '  {:<28} | {:>10}'.format('Events', round(len(events) / count_resources, 4)),
            '  {:<28} | {:>10}'.format('GeNeG resources', round(len(other_geneg_resources) / count_resources, 4)),
            '  {:<28} | {:>10}'.format('Wikidata resources', round(len(wikidata_resources) / count_resources, 4)),
            '\n',

                    
            '\n',
            '-' * 43,
            '{:^43}'.format('Edges'),
            '-' * 43,
            
            '{:<30} | {:>10}'.format('properties', len(self.get_all_properties())),
            '\n',
            '{:<30}   {:>10}'.format('', '-frequency-'),
            '\n'.join("  {:<28} | {:>10}".format(k, round(v, 4)) for k, v in sorted(self.properties_frequencies.items(), key=lambda item: item[1]))
            ])

    @classmethod
    def build_base_graph(cls):
        utils.get_logger().info('GeNeG: Starting to construct knowledge graph from dataset..\n')

        # Load dataset
        dataset = data_loading.load_dataset()
        utils.get_logger().info(f'GeNeG: Loaded dataset with {len(dataset)} articles.\n')

        # Initialize graph
        graph = GeNeG(nx.MultiDiGraph())

        # Add nodes to graph for each article in the dataset
        utils.get_logger().info('GeNeG: Adding new nodes for each article in the dataset..')
        dataset = graph._add_article_nodes(dataset)
        # utils.update_cache('dataset', dataset)
        utils.get_logger().info(f'GeNeG: Added {len(graph.nodes)} new nodes to the graph for each article in the dataset.\n')

        # Add corresponding edges and nodes to the graph for each feaure of the article
        utils.get_logger().info('GeNeG: Adding relations to graph..')
        graph._add_node_type_relation()
        graph._add_publisher_relation(dataset)
        graph._add_provenance_relation(dataset)
        graph._add_published_date_relation(dataset)
        graph._add_last_modified_relation(dataset)
        graph._add_title_relation(dataset)
        graph._add_description_relation(dataset)
        graph._add_body_relation(dataset)
        graph._add_is_based_on_relation(dataset)
        graph._add_keywords_relation(dataset)
        graph._add_author_person_relation(dataset)
        graph._add_author_organization_relation(dataset)
        graph._add_polarity_relation(dataset)
        utils.get_logger().info(f'GeNeG: Added a total of {len(graph.nodes)} nodes and {len(graph.edges)} edges to the graph from the news articles.\n')
        
        utils.get_logger().info('GeNeG: Adding event nodes  and relations to the graph..')
        graph._add_event_relation(dataset)
        graph._add_event_place_relation(dataset)
        graph._add_event_actor_relation(dataset)
        graph._add_event_mention(dataset)
        graph._add_event_mention_part(dataset)
        utils.get_logger().info(f'GeNeG: Added a total of {len(graph.nodes)} nodes and {len(graph.edges)} edges to the graph from the news articles.\n')

        # Remove nodes without an alias
        graph._remove_nodes_without_alias()        

        # Create the mapping of labels and aliases to node ids
        graph._create_mappings2node_id()

        # Merge nodes with similar labels
        utils.get_logger().info('GeNeG: Merging nodes based on labels and aliases.')
        graph.merge_nodes()
        utils.get_logger().info(f'GeNeG: The resulting graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.\n')

        # Remove infrequent nodes
        graph.remove_infrequent_unlinked_resources()
        utils.get_logger().info(f'GeNeG: The resulting graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.\n')
       
        # Cache Wiki map
        utils.update_cache('wiki_attributes_map', self.wiki_attributes_map)

        return graph

    def construct_entities_graph(self):
        utils.get_logger().info('GeNeG: Starting to construct entity knowledge graph from base knowledge graph..\n')

        # Remove literal nodes from the graph
        utils.get_logger().info(f'GeNeG: Removing literal nodes from the graph.')
        
        resources = self.get_geneg_resources().union(self.get_wikidata_resources())
        literals = self.nodes.difference(resources)
        utils.get_logger().debug(f'Found {len(literals)} literal nodes.')
        self._remove_nodes(literals)

        utils.get_logger().info(f'GeNeG: The resulting graph has {len(self.nodes)} nodes and {len(self.edges)} edges.\n')        

        # Add k-hop neighbors
        self.wiki_attributes_map = utils.load_cache('wiki_attributes_map')
        utils.get_logger().info(f'GeNeG: Extending graph with neighbors from Wikidata.')
        self.add_k_hop_neighbors()
        utils.get_logger().info(f'GeNeG: The resulting graph has {len(self.nodes)} nodes and {len(self.edges)} edges.\n')       

        return self

    def construct_complete_graph(self):
        utils.get_logger().info('GeNeG: Starting to construct complete knowledge graph from base knowledge graph..\n')

        # Add k-hop neighbors
        self.wiki_attributes_map = utils.load_cache('wiki_attributes_map')
        utils.get_logger().info(f'GeNeG: Extending graph with neighbors from Wikidata.')
        self.add_k_hop_neighbors()
        utils.get_logger().info(f'GeNeG: The resulting graph has {len(self.nodes)} nodes and {len(self.edges)} edges.\n')       

        return self

    def add_k_hop_neighbors(self) -> None:
        """ Extend entities gaph with k-hop neighbours. """
        k_hop = utils.get_config('geneg.k_hop')
        for k in range(1, k_hop+1):
            triples = wiki_util.get_k_hop_neighbors(k)
            source_nodes = set([triple[0] for triple in triples])
            target_nodes = set([triple[2] for triple in triples])
            utils.get_logger().debug(f'GeNeG: Extending graph with {len(triples)} triples representing {len(target_nodes)} {k}-hop neigbors of {len(source_nodes)} Wikidata entities.')
            
            # Add edges
            edge_types = set([triple[1] for triple in triples])
            for edge_type in tqdm(edge_types):
                edges = list(set([(triple[0], triple[2], triple[1]) for triple in triples if triple[1] == edge_type]))
                self._add_edges(edges)

            # Update mappings
            source_nodes_w_attr = {node: {self.ATTRIBUTE_LABEL: self.wiki_attributes_map[node]['label'], self.ATTRIBUTE_ALIAS: self.wiki_attributes_map[node]['aliases']} 
                            for node in source_nodes if node in self.wiki_attributes_map.keys()}
            self._set_nodes_attributes(source_nodes_w_attr, set_label=True, set_alias=True)

            if k == k_hop:
                # Last hop, remove sink entities
                self.remove_sink_entities()
                target_nodes = [node for node in target_nodes if self.has_node(node)]

                # Retrieve labels and aliases for the k-hop neighbors that are not yet mapped
                non_mapped_qids = [qid for qid in target_nodes if not qid in self.wiki_attributes_map.keys()]
                utils.get_logger().debug(f'GeNeG: Retrieving attributes for {len(non_mapped_qids)} Wikidata entities.')
                for qid in tqdm(non_mapped_qids):
                    self._retrieve_wiki_attributes(qid)

                # Update mappings
                target_nodes_w_attr = {node: {self.ATTRIBUTE_LABEL: self.wiki_attributes_map[node]['label'], self.ATTRIBUTE_ALIAS: self.wiki_attributes_map[node]['aliases']} 
                                for node in target_nodes if node in self.wiki_attributes_map.keys()}
                self._set_nodes_attributes(target_nodes_w_attr, set_label=True, set_alias=True)

                # Cache Wiki map
                utils.update_cache('wiki_attributes_map', self.wiki_attributes_map)

            # Clear global variable corresponding to the k-hop neighbors, to compute the (k+1)-hop neighbors
            wiki_util._clear_global_var()

            utils.get_logger().debug(f'GeNeG: The resulting graph has {len(self.nodes)} nodes and {len(self.edges)} edges.\n')

    def merge_nodes(self) -> None:
        """ Merges nodes with identical or similar labels. """
        self._merge_nodes_w_identical_labels()
        self._merge_nodes_w_similar_labels()   
        self._merge_nodes_w_identical_aliases()
        self._merge_nodes_w_similar_aliases()  
    
    def _merge_nodes_w_identical_labels(self) -> None:
        """ Merges nodes with identical labels, where one is a GeNeG resource, and the other is a Wikidata resource.
        If multiple Wikidata resources have the same label, it merges to the one with more aliases. 
        """
        utils.get_logger().info('GeNeG: Merging nodes with identical labels.')
        
        nodes2merge = list()

        for label in self.label2node_map.keys():
            nodes = self.label2node_map[label]
            if len(nodes) > 1 and len(nodes) < 4 and any(news_kg_util.is_geneg_resource(node) for node in nodes):
                nodes2merge.append(nodes)
        
        matched_nodes_count = 0

        if nodes2merge:
            for nodes in tqdm(nodes2merge):
                nodes = list(nodes)
                unlinked_node = [node for node in nodes if news_kg_util.is_geneg_resource(node)][0]
                nodes.remove(unlinked_node)
                if len(nodes)==1:
                    matching_node = nodes[0]
                else:
                    # Select candidate node with more aliases
                    matching_node = nodes[0] if len(self.get_node_alias(nodes[0])) > len(self.get_node_alias(nodes[1])) else nodes[1]
                
                # Update alias and label mapping
                for alias in self.get_node_alias(unlinked_node):
                    self.alias2node_map[alias].add(matching_node)
                self._remove_nodes_from_mappings([unlinked_node])

                # Merge the nodes
                self.contracted_nodes(matching_node, unlinked_node, attr2update=[self.ATTRIBUTE_ALIAS])
                matched_nodes_count += 1

        utils.get_logger().debug(f'GeNeG: Merged {matched_nodes_count} nodes with identical labels to Wikidata resources.')
        utils.get_logger().debug(f'GeNeG: The resulting graph has {len(self.nodes)} nodes and {len(self.edges)} edges.\n')

    def _merge_nodes_w_similar_labels(self) -> None:
        """ Merge unlinked nodes with similar labels to either Wikidata resources or other unlinked nodes. 
        Similarity of labels is based on the token_sort_ratio. A pair is considered a match if the label 
        smilarity is higher than a predefined threshold. 
        """
        utils.get_logger().debug('GeNeG: Merging nodes with similar labels.')

        nodes2merge = [(node, self.get_node_label(node)) for node in self.get_unlinked_resources() if not self.get_node_label(node) == None]

        matched2wiki_count = 0
        matched2geneg_count = 0

        for unlinked_node, label in tqdm(nodes2merge):
            if not label == None:
                candidate_labels = list(self.label2node_map.keys())
                if None in candidate_labels:
                    candidate_labels.remove(None)

                candidate_labels = [l.replace('_', ' ') for l in candidate_labels]
                processed_label = label.replace('_', ' ')
                if processed_label in candidate_labels:
                    candidate_labels.remove(processed_label)

                candidate = process.extractOne(processed_label, candidate_labels, scorer=fuzz.token_sort_ratio, score_cutoff=utils.get_config('geneg.min_token_sort_ratio'))
                if candidate:
                    matching_node_label = candidate[0].replace(' ', '_')
                    matching_node_ids = list(self.label2node_map[matching_node_label])
                    matching_node_wiki_ids = [node_id for node_id in matching_node_ids if news_kg_util.is_wikidata_resource(node_id)]
                    if matching_node_wiki_ids:
                        # The candidate nodes are only Wikidata resources
                        if len(matching_node_wiki_ids) == 1:
                            matching_node = matching_node_wiki_ids[0]
                        else:
                            matching_node_wiki_ids_w_alias = [(node_id, len(self.get_node_alias(node_id))) for node_id in matching_node_wiki_ids]
                            matching_node = max(matching_node_wiki_ids_w_alias, key=itemgetter(1))[0]
                        matched2wiki_count += 1
                    else:
                        # The candidate nodes are other GeNeG resources
                        if unlinked_node in matching_node_ids:
                            matching_node_ids.remove(unlinked_node)
                        if matching_node_ids:        
                            matching_node_ids_w_alias = [(node_id, len(self.get_node_alias(node_id))) for node_id in matching_node_ids]
                            matching_node = max(matching_node_ids_w_alias, key=itemgetter(1))[0]
                            matched2geneg_count += 1
                        else:
                            matching_node = None

                    if not matching_node == None:
                        # Update alias and label mapping
                        for alias in self.get_node_alias(unlinked_node):
                            self.alias2node_map[alias].add(matching_node)
                        self._remove_nodes_from_mappings([unlinked_node])
    
                        # Merge the nodes
                        self.contracted_nodes(matching_node, unlinked_node, attr2update=[self.ATTRIBUTE_ALIAS])

        utils.get_logger().debug(f'GeNeG: Merged {matched2wiki_count} nodes with similar labels to Wikidata resources.')
        utils.get_logger().debug(f'GeNeG: Merged {matched2geneg_count} nodes with similar labels to other GeNeG resources.')
        utils.get_logger().debug(f'GeNeG: Merged a total of {matched2wiki_count + matched2geneg_count} nodes with similar labels.')
        utils.get_logger().debug(f'GeNeG: The resulting graph has {len(self.nodes)} nodes and {len(self.edges)} edges.\n')
    
    def _merge_nodes_w_identical_aliases(self) -> None:
        """ Merges nodes with identical aliases, where one is a GeNeG resource, and the other is a Wikidata resource. """
        utils.get_logger().info('GeNeG: Merging nodes with identical aliases.')
        
        nodes2merge = list()

        for alias in self.alias2node_map.keys():
            nodes = self.alias2node_map[alias]
            if len(nodes) == 2 and any(news_kg_util.is_geneg_resource(node) for node in nodes) and any(news_kg_util.is_wikidata_resource(node) for node in nodes):
                nodes2merge.append(nodes)

        matched_nodes_count = 0

        if nodes2merge:
            for nodes in tqdm(nodes2merge):
                nodes = list(nodes)
                try:
                    if news_kg_util.is_geneg_resource(nodes[0]):
                        unlinked_node = nodes[0]
                        matching_node = nodes[1]
                    else:
                        unlinked_node = nodes[1]
                        matching_node = nodes[0]

                    # Update alias and label mapping
                    for alias in self.get_node_alias(unlinked_node):
                        self.alias2node_map[alias].add(matching_node)
                    self._remove_nodes_from_mappings([unlinked_node])
                
                    # Merge the nodes
                    self.contracted_nodes(matching_node, unlinked_node, attr2update=[self.ATTRIBUTE_ALIAS])
                    matched_nodes_count += 1
                except Exception as e:
                    print(nodes, e)
        
        utils.get_logger().debug(f'GeNeG: Merged {matched_nodes_count} nodes with identical aliases to Wikidata resources.')
        utils.get_logger().debug(f'GeNeG: The resulting graph has {len(self.nodes)} nodes and {len(self.edges)} edges.\n')

    def _merge_nodes_w_similar_aliases(self) -> None:
        """ Merge unlinked nodes with similar aliases to either Wikidata resources or other unlinked nodes. 
        Similarity of aliases is based on the token_sort_ratio. A pair is considered a match if the alias 
        smilarity is higher than a predefined threshold. 
        """
        utils.get_logger().debug('GeNeG: Merging nodes with similar aliases.')

        nodes2merge = [(node, list(self.get_node_alias(node))) for node in self.get_unlinked_resources() if not self.get_node_alias(node) == None]

        matched2wiki_count = 0
        matched2geneg_count = 0

        for unlinked_node, aliases in tqdm(nodes2merge):
            candidate_aliases = list(self.alias2node_map.keys())
            for unlinked_node_alias in aliases:
                if unlinked_node_alias in candidate_aliases:
                    candidate_aliases.remove(unlinked_node_alias)

            for alias in aliases:
                candidate = process.extractOne(alias, candidate_aliases, scorer=fuzz.token_sort_ratio, score_cutoff=utils.get_config('geneg.min_token_sort_ratio'))
                if candidate:
                    matching_node_ids = list(self.alias2node_map[candidate[0]])
                    matching_node_wiki_ids = [node_id for node_id in matching_node_ids if news_kg_util.is_wikidata_resource(node_id)]
                    if matching_node_wiki_ids:
                        # The candidate nodes are only Wikidata resources
                        if len(matching_node_wiki_ids) == 1:
                            matching_node = matching_node_wiki_ids[0]
                        else:
                            matching_node_wiki_ids_w_alias = [(node_id, len(self.get_node_alias(node_id))) for node_id in matching_node_wiki_ids]
                            matching_node = max(matching_node_wiki_ids_w_alias, key=itemgetter(1))[0]
                        matched2wiki_count += 1
                    else:
                        # The candidate nodes are other GeNeG resources
                        if unlinked_node in matching_node_ids:
                            matching_node_ids.remove(unlinked_node)
                        if matching_node_ids:
                            matching_node_ids_w_alias = [(node_id, len(self.get_node_alias(node_id))) for node_id in matching_node_ids]
                            matching_node = max(matching_node_ids_w_alias, key=itemgetter(1))[0]
                            matched2geneg_count += 1
                        else:
                            matching_node = None
                    
                    if not matching_node == None:
                        # Update alias and label mapping
                        for alias in aliases:
                            self.alias2node_map[alias].add(matching_node)
                        self._remove_nodes_from_mappings([unlinked_node])
                       
                        # Merge the nodes
                        self.contracted_nodes(matching_node, unlinked_node, attr2update=[self.ATTRIBUTE_ALIAS])

                        # Match already found, no need to iterate through the remaining aliases
                        break 
        
        utils.get_logger().debug(f'GeNeG: Merged {matched2wiki_count} nodes with similar aliases to Wikidata resources.')
        utils.get_logger().debug(f'GeNeG: Merged {matched2geneg_count} nodes with similar aliases to other GeNeG resources.')
        utils.get_logger().debug(f'GeNeG: Merged a total of {matched2wiki_count + matched2geneg_count} nodes with similar aliases.')
        utils.get_logger().debug(f'GeNeG: The resulting graph has {len(self.nodes)} nodes and {len(self.edges)} edges.\n')

    def _create_mappings2node_id(self) -> None:
        """ Creates the mapping of node labels and aliases to node ids for all non-literal nodes. """
        resources = self.get_unlinked_resources().union(self.get_wikidata_resources())
        utils.get_logger().info(f'GeNeG: Creating mappings of node labels and aliases to ids for {len(resources)} nodes.\n')
        for node_id in resources:
            label = self.get_node_label(node_id)
            if label:
                self.label2node_map[label].add(node_id)
            aliases = self.get_node_alias(node_id)
            if aliases:
                for alias in aliases:
                    self.alias2node_map[alias].add(node_id)

    def _remove_nodes_without_alias(self) -> None:
        """ Removes resource nodes without an alias. """
        utils.get_logger().info(f'GeNeG: Removing nodes without an alias.')
        resources = self.get_unlinked_resources().union(self.get_wikidata_resources())
        
        removed_nodes = set()
        for node_id in resources:
            aliases = self.get_node_alias(node_id)
            if not aliases or all(len(alias)==1 for alias in list(aliases)):
                removed_nodes.add(node_id)
        self._remove_nodes(removed_nodes)
        utils.get_logger().debug(f'GeNeG: Removed {len(removed_nodes)} nodes without an alias.\n')

    def remove_sink_entities(self) -> None:
        """ Removes entities that have a frequency (i.e. in-degree) less than the min. threshold. """
        utils.get_logger().info(f'GeNeG: Removing sink entities from the graph.')
        wikidata_resources = self.get_wikidata_resources()
        sink_entities = [node for node in wikidata_resources if self.degree(node) < utils.get_config('geneg.max_sink_entities_frequency')]
        utils.get_logger().debug(f'GeNeG: Found {len(sink_entities)} sink entities.')

        # Remove sink entities from label and alias mappings
        self._remove_nodes_from_mappings(sink_entities)

         # Remove sink entities from graph
        self._remove_nodes(sink_entities)

        # Remove unused labels and aliases from mapping to node ids
        utils.get_logger().info(f'GeNeG: Updating mappings of labels and aliases to node ids.')
        self._remove_unused_labels_from_map()
        self._remove_unused_aliases_from_map()

    def remove_infrequent_unlinked_resources(self) -> None:
        """ Removes any unlinked resource that has a frequency (i.e. in-degree) less than the min. threshold. """
        utils.get_logger().info(f'GeNeG: Removing infrequent nodes from the graph.')
        unlinked_resources = self.get_unlinked_resources()
        infrequent_nodes = [node for node in unlinked_resources if self.degree(node) < utils.get_config('geneg.min_unlinked_resource_frequency')]
        utils.get_logger().debug(f'GeNeG: Found {len(infrequent_nodes)} infrequent nodes.')

        # Remove infrequent nodes from label and alias mappings
        self._remove_nodes_from_mappings(infrequent_nodes) 

        # Remove infrequent nodes from graph
        self._remove_nodes(infrequent_nodes)

        # Remove unused labels and aliases from mapping to node ids
        utils.get_logger().info(f'GeNeG: Updating mappings of labels and aliases to node ids.')
        self._remove_unused_labels_from_map()
        self._remove_unused_aliases_from_map()

    def _remove_nodes_from_mappings(self, nodes_to_remove: List[str]) -> None:
        """ Removes the node ids from the given list from the label and alias maps. """
        for node_id in nodes_to_remove:
            label = self.get_node_label(node_id)
            if label and node_id in self.label2node_map[label]:
                self.label2node_map[label].remove(node_id)
                if not self.label2node_map[label]:
                    self.label2node_map.pop(label, None)

            aliases = self.get_node_alias(node_id)
            if aliases:
                for alias in aliases:
                    if node_id in self.alias2node_map[alias]:  
                        self.alias2node_map[alias].remove(node_id)
                    if not self.alias2node_map[alias]:
                        self.alias2node_map.pop(alias, None)

    def _remove_unused_labels_from_map(self) -> None:
        """ Removes labels that are no longer mapped to node ids from the mapping dictionary. """
        labels_to_remove = [label for label in self.label2node_map.keys() if not self.label2node_map[label]]
        utils.get_logger().debug(f'GeNeG: Found {len(labels_to_remove)} unused labels.')
        for label in labels_to_remove:
            self.label2node_map.pop(label, None)

    def _remove_unused_aliases_from_map(self) -> None:
        """ Removes aliases that are no longer mapped to node ids from the mapping dictionary. """
        aliases_to_remove = [alias for alias in self.alias2node_map.keys() if not self.alias2node_map[alias]]
        utils.get_logger().debug(f'GeNeG: Found {len(aliases_to_remove)} unused aliases.\n')
        for alias in aliases_to_remove:
            self.alias2node_map.pop(alias, None)

    def _generate_node_id(self) -> str:
        """ Generates an ID for a node representing an article. """
        node_id = str(uuid.uuid4())[:13].replace('-', '')
        return node_id
    
    def _convert_to_geneg_resource(self) -> None:
        """ Generates a unique node id for nodes representing an article. """
        resource_id = news_kg_util.id2geneg_resource('news_' + self._generate_node_id())
        return resource_id if not self.has_node(resource_id) else self._convert_to_geneg_resource()

    def _add_article_nodes(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """ Add articles as new nodes in the graph. """
        # node_ids = [self._convert_to_geneg_resource() for _ in range(len(dataset))]
        node_ids = list(dataset['node_id'])
        self._add_nodes(node_ids)

        # Add newly created node IDs to dataframe for disambiguation of articles with the same title
        # dataset['node_id'] = node_ids
        return dataset

    def _node_id2idx(self, dataset: pd.DataFrame, node_id: str) -> pd.Int64Index:
        """ Returns the index in the dataset of the article with the given node id. """
        self._check_node_exists(node_id)
        article_idx = dataset.loc[dataset['node_id']==node_id].index
        return article_idx

    def _article_node2event(self, node_id: str) -> str:
        """ Returs the event node id corresponding to the given article node id. """
        return node_id + '_evt'

    def _event_node2article(self, node_id: str) -> str:
        """ Returns the article node id corresponding to the given event node id. """
        return node_id.split('_evt')[0]

    def _add_node_type_relation(self, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'rdf:type' between the set of given source nodes and the target node 'schema:NewsArticle'. """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "rdf:type" relations for {len(source_nodes)} nodes.')
        target_nodes = [rdf_util.CLASS_NEWS_ARTICLE] * len(source_nodes)
        edge_type = [rdf_util.PREDICATE_TYPE] * len(source_nodes)
        self._add_edges(zip(source_nodes, target_nodes, edge_type))
        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_publisher_relation(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'schema:publisher' between the given source node(s) and the publisher of the corresponding article(s) representing the target node(s). """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "schema:publisher" relations for {len(source_nodes)} nodes.')

        target_nodes = [dataset['disambiguated_news_outlet'].loc[self._node_id2idx(dataset, node)].item() for node in source_nodes]
        edge_type = [rdf_util.PREDICATE_PUBLISHER] * len(source_nodes)

        # Node labels 
        linked_nodes = list(set([news_kg_util.qid2wikidata_resource(node[1]) for node in target_nodes if node[1]!='NIL']))
        linked_nodes_labels = self.get_wikidata_label([node for node in linked_nodes]) 
        linked_nodes_labels = [nlp_util.get_canonical_label(label) for label in linked_nodes_labels]
        linked_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[linked_nodes, linked_nodes_labels])}
    
        unlinked_nodes_labels = [nlp_util.get_canonical_label(nlp_util.clean_label(node[0], entity_type='ORG')) for node in target_nodes if node[1] == 'NIL']
        unlinked_nodes = [news_kg_util.label2geneg_resource(label) for label in unlinked_nodes_labels]
        unlinked_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[unlinked_nodes, unlinked_nodes_labels])}

        # Node aliases
        nodes_aliases = [node[0] for node in target_nodes]
        target_nodes = [news_kg_util.qid2wikidata_resource(node[1]) if node[1]!='NIL' else news_kg_util.label2geneg_resource(nlp_util.get_canonical_label(nlp_util.clean_label(node[0], entity_type='ORG'))) for node in target_nodes]
        nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: nlp_util.normalize_unicodedata(item[1])} for item in zip(*[target_nodes, nodes_aliases])}
        linked_nodes_aliases = [self.wiki_attributes_map[node]['aliases'] for node in linked_nodes]
        linked_nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: item[1]} for item in zip(*[linked_nodes, linked_nodes_aliases]) if item[1]}

        # Add edges
        self._add_edges(zip(source_nodes, target_nodes, edge_type))

        # Set the node attributes
        self._set_nodes_attributes(linked_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(unlinked_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(nodes_w_aliases, set_label=False, set_alias=True)
        self._set_nodes_attributes(linked_nodes_w_aliases, set_label=False, set_alias=True)

        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_provenance_relation(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'schema:url' between the given source node(s) and the provenance URL of the corresponding article(s) representing the target node(s). """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "schema:url" relations for {len(source_nodes)} nodes.')
        
        target_nodes = [dataset['provenance'].loc[self._node_id2idx(dataset, node)].item() for node in source_nodes]
        edge_type = [rdf_util.PREDICATE_URL] * len(source_nodes)

        self._add_edges(zip(source_nodes, target_nodes, edge_type))
        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')
    
    def _add_published_date_relation(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'schema:datePublished' between the given source node(s) and the publishing date of the corresponding article(s) representing the target node(s). """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "schema:datePublished" relations for {len(source_nodes)} nodes.')

        target_nodes = [dataset['creation_date'].loc[self._node_id2idx(dataset, node)].item() for node in source_nodes]
        edge_type = [rdf_util.PREDICATE_DATE_PUBLISHED] * len(source_nodes)

        self._add_edges(zip(source_nodes, target_nodes, edge_type))
        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_last_modified_relation(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'schema:dateModified' between the given source node(s) and the modification date of the corresponding article(s) representing the target node(s). """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "schema:dateModified" relations for {len(source_nodes)} nodes.')

        target_nodes = [dataset['last_modified'].loc[self._node_id2idx(dataset, node)].item() for node in source_nodes]
        edge_type = [rdf_util.PREDICATE_DATE_MODIFIED] * len(source_nodes)

        self._add_edges(zip(source_nodes, target_nodes, edge_type))
        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_title_relation(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'schema:headline' between the given source node(s) and the title of the corresponding article(s) representing the target node(s). """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "schema:headline" relations for {len(source_nodes)} nodes.')

        target_nodes = [dataset['title'].loc[self._node_id2idx(dataset, node)].item() for node in source_nodes]
        edge_type = [rdf_util.PREDICATE_HEADLINE] * len(source_nodes)
        
        self._add_edges(zip(source_nodes, target_nodes, edge_type))
        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_description_relation(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'schema:abstract' between the given source node(s) and the description of the corresponding article(s) representing the target node(s). """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "schema:abstract" relations for {len(source_nodes)} nodes.')
        target_nodes = [dataset['description'].loc[self._node_id2idx(dataset, node)].item() for node in source_nodes]
        
        # Do not add edges if description if missing (i.e. empty string)
        edges_description = [(source_nodes[i], target_nodes[i]) for i in range(len(source_nodes)) if not target_nodes[i]=='']

        edges = [(edge[0], edge[1], rdf_util.PREDICATE_ABSTRACT) for edge in edges_description]
        self._add_edges(edges)
        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_body_relation(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'schema:articleBody' between the given source node(s) and the body of the corresponding article(s) representing the target node(s). """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "schema:articleBody" relations for {len(source_nodes)} nodes.')

        target_nodes = [dataset['body'].loc[self._node_id2idx(dataset, node)].item() for node in source_nodes]
        edge_type = [rdf_util.PREDICATE_ARTICLE_BODY] * len(source_nodes)

        self._add_edges(zip(source_nodes, target_nodes, edge_type))
        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_is_based_on_relation(self, dataset: pd.DataFrame) -> None:
        """ Adds edges of type 'schema:isBasedOn' between news which are updated versions of each other (they contain overlapping elements). """
        utils.get_logger().debug(f'GeNeG: Adding "schema:isBasedOn" relations.')
        updated_articles = dataset[dataset.duplicated(subset='title', keep=False)==True]
        grouped_updated_articles = updated_articles.groupby('title').apply(lambda x: x.index.to_list())
        updated_articles_idx = grouped_updated_articles.values.tolist()
 
        overlapping_articles_idx = list()
        for idx_pair in updated_articles_idx:
            overlap_ratio = self._get_overlap_ratio(dataset.loc[idx_pair[0]]['body'], dataset.loc[idx_pair[1]]['body'])
            overlap_threshold = 0.3
            if overlap_ratio >= overlap_threshold:
                # Consider only articles with an overlap ratio higher than the threshold
                sample = updated_articles.loc[idx_pair]

                if sample.loc[idx_pair[0]]['creation_date'] != sample.loc[idx_pair[1]]['creation_date']:
                    # Order by creation date
                    older_article_idx = sample.loc[sample['creation_date']==sample['creation_date'].min()].index
                    newer_article_idx = sample.loc[sample['creation_date']==sample['creation_date'].max()].index
                    overlapping_articles_idx.append((dataset.loc[newer_article_idx.values[0]]['node_id'], dataset.loc[older_article_idx.values[0]]['node_id']))
                elif sample.loc[idx_pair[0]]['last_modified'] != sample.loc[idx_pair[1]]['last_modified']:
                    # Order by last modification date
                    older_article_idx = sample.loc[sample['last_modified']==sample['last_modified'].min()].index
                    newer_article_idx = sample.loc[sample['last_modified']==sample['last_modified'].max()].index
                    overlapping_articles_idx.append((dataset.loc[newer_article_idx.values[0]]['node_id'], dataset.loc[older_article_idx.values[0]]['node_id']))
                else: 
                    # Order by artile length
                    older_article_idx = sample.loc[sample['body'].str.len().idxmin()].name
                    newer_article_idx = sample.loc[sample['body'].str.len().idxmax()].name
                    overlapping_articles_idx.append((dataset.loc[newer_article_idx]['node_id'], dataset.loc[older_article_idx]['node_id']))

        edges = [(article[0], article[1], rdf_util.PREDICATE_IS_BASED_ON) for article in overlapping_articles_idx]
        self._add_edges(edges)
        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')


    def _add_keywords_relation(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'schema:keywords' between the given source node(s) and the news keywords assigned to the corresponding article(s) representing the target node(s). """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "schema:keywords" relations for {len(source_nodes)} nodes.')
 
        target_nodes = [dataset['disambiguated_news_keywords'].loc[self._node_id2idx(dataset, self._event_node2article(node))].item() for node in source_nodes]
        edges = [(source_nodes[i], entity) for i in range(len(source_nodes)) for entity in target_nodes[i] if not len(target_nodes[i])==0]
       
        # Node labels
        linked_nodes = list(set([news_kg_util.qid2wikidata_resource(edge[1][1]) for edge in edges if edge[1][1]!='NIL']))
        linked_nodes_labels = self.get_wikidata_label([node for node in linked_nodes])
        linked_nodes_labels = [nlp_util.get_canonical_label(label) for label in linked_nodes_labels]
        linked_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[linked_nodes, linked_nodes_labels])}

        unlinked_nodes_labels = [nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='PER')) for edge in edges if edge[1][1]=='NIL']
        unlinked_nodes = [news_kg_util.label2geneg_resource(label) for label in unlinked_nodes_labels]
        unlinked_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[unlinked_nodes, unlinked_nodes_labels])}

        # Node aliases
        nodes_aliases = [edge[1][0] for edge in edges]
        edges = [(edge[0], news_kg_util.qid2wikidata_resource(edge[1][1])) if edge[1][1]!='NIL' else (edge[0], news_kg_util.label2geneg_resource(nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='PER')))) for edge in edges]
        target_nodes = [edge[1] for edge in edges]
        nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: nlp_util.normalize_unicodedata(item[1])} for item in zip(*[target_nodes, nodes_aliases])}
        linked_nodes_aliases = [self.wiki_attributes_map[node]['aliases'] for node in linked_nodes]
        linked_nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: item[1]} for item in zip(*[linked_nodes, linked_nodes_aliases]) if item[1]}

        # Add edges
        edges = [(edge[0], edge[1], rdf_util.PREDICATE_KEYWORDS) for edge in list(set(edges))]
        self._add_edges(edges)

        # Set node attributes
        self._set_nodes_attributes(linked_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(unlinked_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(nodes_w_aliases, set_label=False, set_alias=True)
        self._set_nodes_attributes(linked_nodes_w_aliases, set_label=False, set_alias=True)

        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_author_person_relation(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'schema:author' edges between the given source node(s) and the authors of the corresponding article(s) representing the target node(s). """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "schema:author" relations for {len(source_nodes)} (author person) nodes.')

        target_nodes = [dataset['disambiguated_author_person'].loc[self._node_id2idx(dataset, node)].item() for node in source_nodes]
        edges = [(source_nodes[i], author) for i in range(len(source_nodes)) for author in target_nodes[i] if not len(target_nodes[i])==0]
        
        # Node labels 
        linked_nodes = list(set([news_kg_util.qid2wikidata_resource(edge[1][1]) for edge in edges if edge[1][1]!='NIL']))
        linked_nodes_labels = self.get_wikidata_label([node for node in linked_nodes])
        linked_nodes_labels = [nlp_util.get_canonical_label(label) for label in linked_nodes_labels]
        linked_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[linked_nodes, linked_nodes_labels])}

        unlinked_nodes_labels = [nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='PER')) for edge in edges if edge[1][1]=='NIL']
        unlinked_nodes = [news_kg_util.label2geneg_resource(label) for label in unlinked_nodes_labels]
        unlinked_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[unlinked_nodes, unlinked_nodes_labels])}

        # Node aliases
        nodes_aliases = [edge[1][0] for edge in edges]
        edges = [(edge[0], news_kg_util.qid2wikidata_resource(edge[1][1])) if edge[1][1]!='NIL' else (edge[0], news_kg_util.label2geneg_resource(nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='PER')))) for edge in edges]
        target_nodes = [edge[1] for edge in edges]
        nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: nlp_util.normalize_unicodedata(item[1])} for item in zip(*[target_nodes, nodes_aliases])}
        linked_nodes_aliases = [self.wiki_attributes_map[node]['aliases'] for node in linked_nodes]
        linked_nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: item[1]} for item in zip(*[linked_nodes, linked_nodes_aliases]) if item[1]}

        # Add edges
        edges = [(edge[0], edge[1], rdf_util.PREDICATE_AUTHOR) for edge in list(set(edges))]
        self._add_edges(edges)

        # Set node attributes
        self._set_nodes_attributes(linked_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(unlinked_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(nodes_w_aliases, set_label=False, set_alias=True)
        self._set_nodes_attributes(linked_nodes_w_aliases, set_label=False, set_alias=True)

        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_author_organization_relation(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'schema:author' edges between the given source node(s) and the organization authors of the corresponding article(s) representing the target node(s). """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "schema:author" relations for {len(source_nodes)} (author organization) nodes.')

        target_nodes = [dataset['disambiguated_author_organization'].loc[self._node_id2idx(dataset, node)].item() for node in source_nodes]
        edges = [(source_nodes[i], author) for i in range(len(source_nodes)) for author in target_nodes[i] if not len(target_nodes[i])==0]
        
        # Node labels
        linked_nodes = list(set([news_kg_util.qid2wikidata_resource(edge[1][1]) for edge in edges if edge[1][1]!='NIL']))
        linked_nodes_labels = self.get_wikidata_label([node for node in linked_nodes])
        linked_nodes_labels = [nlp_util.get_canonical_label(label) for label in linked_nodes_labels]
        linked_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[linked_nodes, linked_nodes_labels])}

        unlinked_nodes_labels = [nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='ORG')) for edge in edges if edge[1][1]=='NIL']
        unlinked_nodes = [news_kg_util.label2geneg_resource(label) for label in unlinked_nodes_labels]
        unlinked_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[unlinked_nodes, unlinked_nodes_labels])}

        # Node aliases
        nodes_aliases = [edge[1][0] for edge in edges]
        edges = [(edge[0], news_kg_util.qid2wikidata_resource(edge[1][1])) if edge[1][1]!='NIL' else (edge[0], news_kg_util.label2geneg_resource(nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='ORG')))) for edge in edges]
        target_nodes = [edge[1] for edge in edges]
        nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: nlp_util.normalize_unicodedata(item[1])} for item in zip(*[target_nodes, nodes_aliases])}
        linked_nodes_aliases = [self.wiki_attributes_map[node]['aliases'] for node in linked_nodes]
        linked_nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: item[1]} for item in zip(*[linked_nodes, linked_nodes_aliases]) if item[1]}

        # Add edges
        edges = [(edge[0], edge[1], rdf_util.PREDICATE_AUTHOR) for edge in list(set(edges))]
        self._add_edges(edges)

        # Set node attributes
        self._set_nodes_attributes(linked_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(unlinked_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(nodes_w_aliases, set_label=False, set_alias=True)
        self._set_nodes_attributes(linked_nodes_w_aliases, set_label=False, set_alias=True)

        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_polarity_relation(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'geneg:polarity' between the given source node(s) and the body of the corresponding article(s) representing the target node(s). """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()

        # Remove nodes without a polarity score
        idx_nodes_without_polarity_score = dataset[dataset['sentiment_score'].isnull()].index.tolist()
        for idx in idx_nodes_without_polarity_score:
            source_nodes.remove(dataset.loc[idx]['node_id'])

        utils.get_logger().debug(f'GeNeG: Adding "geneg:polarity" relations for {len(source_nodes)} nodes.')

        target_nodes = [dataset['sentiment_score'].loc[self._node_id2idx(dataset, node)].item() for node in source_nodes]
        edge_type = [rdf_util.PREDICATE_POLARITY] * len(source_nodes)

        self._add_edges(zip(source_nodes, target_nodes, edge_type))
        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_event_relation(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds a new node to represent an event mentioned in the news article and an edge of type 'schema:about' for any source node corresponding to an article with at least one named entity extracted. For each added event node, it adds an 'rdf:type' relations. """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "schema:about" relations for {len(source_nodes)} event nodes.')

        source_nodes = [node for node in source_nodes if self._article_has_event(dataset, self._node_id2idx(dataset, node))] 
        target_nodes = [self._article_node2event(node) for node in source_nodes]
        edge_type = [rdf_util.PREDICATE_ABOUT] * len(source_nodes)

        self._add_edges(zip(source_nodes, target_nodes, edge_type))
        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

        self._add_event_type_relation(target_nodes)
    
    def _article_has_event(self, dataset: pd.DataFrame, idx: pd.Int64Index) -> bool:
        """ Checks if the article with the given index has at least one type of named entity extracted. """
        return any(
            dataset.loc[idx]['PER_all'].item() or 
            dataset.loc[idx]['LOC_all'].item() or 
            dataset.loc[idx]['ORG_all'].item() or 
            dataset.loc[idx]['OTH_all'].item() or
            dataset.loc[idx]['PERpart_all'].item() or 
            dataset.loc[idx]['LOCpart_all'].item() or 
            dataset.loc[idx]['ORGpart_all'].item() or 
            dataset.loc[idx]['OTHpart_all'].item()
            ) 

    def _add_event_type_relation(self, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'rdf:type' between the set of given source nodes and the target node 'sem:Event'. """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "rdf:type" relations for {len(source_nodes)} event nodes.')

        target_nodes = [rdf_util.CLASS_EVENT] * len(source_nodes)
        edge_type = [rdf_util.PREDICATE_TYPE] * len(source_nodes)

        self._add_edges(zip(source_nodes, target_nodes, edge_type))
        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_event_actor_relation(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'sem:hasActor' edges between the given source node(s) and the extracted person or location entities of the corresponding article(s) representing the target node(s). """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "sem:hasActor" relations for {len(source_nodes)} event nodes (for person entities).')
        
        # Get corresponding event nodes
        source_nodes = [self._article_node2event(node) for node in source_nodes]
        source_nodes = [node for node in source_nodes if self.has_node(node)]
            
        # Named entities representing persons
        per_target_nodes = [dataset['PER_all'].loc[self._node_id2idx(dataset, self._event_node2article(node))].item() for node in source_nodes]
        per_edges = [(source_nodes[i], entity) for i in range(len(source_nodes)) for entity in per_target_nodes[i] if not len(per_target_nodes[i])==0]
        
        # Node labels
        linked_per_nodes = list(set([news_kg_util.qid2wikidata_resource(edge[1][1]) for edge in per_edges if edge[1][1]!='NIL']))
        linked_per_nodes_labels = self.get_wikidata_label([node for node in linked_per_nodes])
        linked_per_nodes_labels = [nlp_util.get_canonical_label(label) for label in linked_per_nodes_labels]
        linked_per_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[linked_per_nodes, linked_per_nodes_labels])}
        
        unlinked_per_nodes_labels = [nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='PER')) for edge in per_edges if edge[1][1]=='NIL']
        unlinked_per_nodes = [news_kg_util.label2geneg_resource(label) for label in unlinked_per_nodes_labels]
        unlinked_per_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[unlinked_per_nodes, unlinked_per_nodes_labels])}
        
        # Node aliases
        per_nodes_aliases = [edge[1][0] for edge in per_edges]
        per_edges = [(edge[0], news_kg_util.qid2wikidata_resource(edge[1][1])) if edge[1][1]!='NIL' else (edge[0], news_kg_util.label2geneg_resource(nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='PER')))) for edge in per_edges]
        target_per_nodes = [edge[1] for edge in per_edges]
        per_nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: nlp_util.normalize_unicodedata(item[1])} for item in zip(*[target_per_nodes, per_nodes_aliases])}
        linked_per_nodes_aliases = [self.wiki_attributes_map[node]['aliases'] for node in linked_per_nodes]
        linked_per_nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: item[1]} for item in zip(*[linked_per_nodes, linked_per_nodes_aliases]) if item[1]}

        # Add edges
        per_edges = [(edge[0], edge[1], rdf_util.PREDICATE_HAS_ACTOR) for edge in list(set(per_edges))]
        self._add_edges(per_edges)
        
        # Set node attributes
        self._set_nodes_attributes(linked_per_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(unlinked_per_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(per_nodes_w_aliases, set_label=False, set_alias=True)
        self._set_nodes_attributes(linked_per_nodes_w_aliases, set_label=False, set_alias=True)

        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

        # Named entities representing organizations
        utils.get_logger().debug(f'GeNeG: Adding "sem:hasActor" relations for {len(source_nodes)} event nodes (for organization entities).')
        org_target_nodes = [dataset['ORG_all'].loc[self._node_id2idx(dataset, self._event_node2article(node))].item() for node in source_nodes]
        org_edges = [(source_nodes[i], entity) for i in range(len(source_nodes)) for entity in org_target_nodes[i] if not len(org_target_nodes[i])==0]
        
        # Node labels
        linked_org_nodes = list(set([news_kg_util.qid2wikidata_resource(edge[1][1]) for edge in org_edges if edge[1][1]!='NIL']))
        linked_org_nodes_labels = self.get_wikidata_label([node for node in linked_org_nodes])
        linked_org_nodes_labels = [nlp_util.get_canonical_label(label) for label in linked_org_nodes_labels]
        linked_org_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[linked_org_nodes, linked_org_nodes_labels])}
        
        unlinked_org_nodes_labels = [nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='ORG')) for edge in org_edges if edge[1][1]=='NIL']
        unlinked_org_nodes = [news_kg_util.label2geneg_resource(label) for label in unlinked_org_nodes_labels]
        unlinked_org_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[unlinked_org_nodes, unlinked_org_nodes_labels])}
        
        # Node aliases
        org_nodes_aliases = [edge[1][0] for edge in org_edges]
        org_edges = [(edge[0], news_kg_util.qid2wikidata_resource(edge[1][1])) if edge[1][1]!='NIL' else (edge[0], news_kg_util.label2geneg_resource(nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='ORG')))) for edge in org_edges]
        target_org_nodes = [edge[1] for edge in org_edges]
        org_nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: nlp_util.normalize_unicodedata(item[1])} for item in zip(*[target_org_nodes, org_nodes_aliases])}
        linked_org_nodes_aliases = [self.wiki_attributes_map[node]['aliases'] for node in linked_org_nodes]
        linked_org_nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: item[1]} for item in zip(*[linked_org_nodes, linked_org_nodes_aliases]) if item[1]}

        # Add edges
        org_edges = [(edge[0], edge[1], rdf_util.PREDICATE_HAS_ACTOR) for edge in list(set(org_edges))]
        self._add_edges(org_edges)
  
        # Set node attributes
        self._set_nodes_attributes(linked_org_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(unlinked_org_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(org_nodes_w_aliases, set_label=False, set_alias=True)
        self._set_nodes_attributes(linked_org_nodes_w_aliases, set_label=False, set_alias=True)

        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_event_place_relation(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'sem:hasPlace' edges between the given source node(s) and the extracted location entities of the corresponding article(s) representing the target node(s). """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "sem:hasPlace" relations for {len(source_nodes)} event nodes.')
        
        # Get corresponding event nodes
        source_nodes = [self._article_node2event(node) for node in source_nodes]
        source_nodes = [node for node in source_nodes if self.has_node(node)]

        target_nodes = [dataset['LOC_all'].loc[self._node_id2idx(dataset, self._event_node2article(node))].item() for node in source_nodes]
        edges = [(source_nodes[i], entity) for i in range(len(source_nodes)) for entity in target_nodes[i] if not len(target_nodes[i])==0]
        
        # Node labels
        linked_nodes = list(set([news_kg_util.qid2wikidata_resource(edge[1][1]) for edge in edges if edge[1][1]!='NIL']))
        linked_nodes_labels = self.get_wikidata_label([node for node in linked_nodes])
        linked_nodes_labels = [nlp_util.get_canonical_label(label) for label in linked_nodes_labels]
        linked_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[linked_nodes, linked_nodes_labels])}

        unlinked_nodes_labels = [nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='LOC')) for edge in edges if edge[1][1]=='NIL']
        unlinked_nodes = [news_kg_util.label2geneg_resource(label) for label in unlinked_nodes_labels]
        unlinked_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[unlinked_nodes, unlinked_nodes_labels])}

        # Node aliases
        nodes_aliases = [edge[1][0] for edge in edges]
        edges = [(edge[0], news_kg_util.qid2wikidata_resource(edge[1][1])) if edge[1][1]!='NIL' else (edge[0], news_kg_util.label2geneg_resource(nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='LOC')))) for edge in edges]
        target_nodes = [edge[1] for edge in edges]
        nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: nlp_util.normalize_unicodedata(item[1])} for item in zip(*[target_nodes, nodes_aliases])}
        linked_nodes_aliases = [self.wiki_attributes_map[node]['aliases'] for node in linked_nodes]
        linked_nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: item[1]} for item in zip(*[linked_nodes, linked_nodes_aliases]) if item[1]}

        # Add edges
        edges = [(edge[0], edge[1], rdf_util.PREDICATE_HAS_PLACE) for edge in list(set(edges))]
        self._add_edges(edges)
       
        # Set node attributes
        self._set_nodes_attributes(linked_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(unlinked_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(nodes_w_aliases, set_label=False, set_alias=True)
        self._set_nodes_attributes(linked_nodes_w_aliases, set_label=False, set_alias=True)

        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_event_mention(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'schema:mentions' edges between the given source node(s) and the extracted named entities of type 'other' of the corresponding article(s) representing the target node(s). """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "schema:mentions" relations for {len(source_nodes)} event nodes.')
        
        # Get corresponding event nodes
        source_nodes = [self._article_node2event(node) for node in source_nodes]
        source_nodes = [node for node in source_nodes if self.has_node(node)]

        target_nodes = [dataset['OTH_all'].loc[self._node_id2idx(dataset, self._event_node2article(node))].item() for node in source_nodes]
        edges = [(source_nodes[i], entity) for i in range(len(source_nodes)) for entity in target_nodes[i] if not len(target_nodes[i])==0]
        
        # Node labels
        linked_nodes = list(set([news_kg_util.qid2wikidata_resource(edge[1][1]) for edge in edges if edge[1][1]!='NIL']))
        linked_nodes_labels = self.get_wikidata_label([node for node in linked_nodes])
        linked_nodes_labels = [nlp_util.get_canonical_label(label) for label in linked_nodes_labels]
        linked_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[linked_nodes, linked_nodes_labels])}

        unlinked_nodes_labels = [nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='OTH')) for edge in edges if edge[1][1]=='NIL']
        unlinked_nodes = [news_kg_util.label2geneg_resource(label) for label in unlinked_nodes_labels]
        unlinked_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[unlinked_nodes, unlinked_nodes_labels])}

        # Node aliases
        nodes_aliases = [edge[1][0] for edge in edges]
        edges = [(edge[0], news_kg_util.qid2wikidata_resource(edge[1][1])) if edge[1][1]!='NIL' else (edge[0], news_kg_util.label2geneg_resource(nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='OTH')))) for edge in edges]
        target_nodes = [edge[1] for edge in edges]
        nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: nlp_util.normalize_unicodedata(item[1])} for item in zip(*[target_nodes, nodes_aliases])}
        linked_nodes_aliases = [self.wiki_attributes_map[node]['aliases'] for node in linked_nodes]
        linked_nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: item[1]} for item in zip(*[linked_nodes, linked_nodes_aliases]) if item[1]}

        # Add edges
        edges = [(edge[0], edge[1], rdf_util.PREDICATE_MENTIONS) for edge in list(set(edges))]
        self._add_edges(edges)
        
        # Set node attributes
        self._set_nodes_attributes(linked_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(unlinked_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(nodes_w_aliases, set_label=False, set_alias=True)
        self._set_nodes_attributes(linked_nodes_w_aliases, set_label=False, set_alias=True)

        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _add_event_mention_part(self, dataset: pd.DataFrame, source_nodes: List[str]=None) -> None:
        """ Adds edges of type 'schema:mentions' edges between the given source node(s) and the extracted named entities parts of types 'PER', 'LOC', 'ORG', and 'OTH', 
        of the corresponding article(s) representing the target node(s). 
        """
        source_nodes = source_nodes if not source_nodes is None else self.get_article_nodes()
        utils.get_logger().debug(f'GeNeG: Adding "schema:mentions" relations for {len(source_nodes)} event nodes.')

        # Get corresponding event nodes
        source_nodes = [self._article_node2event(node) for node in source_nodes]
        source_nodes = [node for node in source_nodes if self.has_node(node)]

        target_nodes = list()
        edges = list()
        entity_types = ['PER', 'LOC', 'ORG', 'OTH']
        for entity_type in entity_types:
            target_nodes_per_type = [dataset[entity_type + 'part_all'].loc[self._node_id2idx(dataset, self._event_node2article(node))].item() for node in source_nodes]
            target_nodes.extend(target_nodes_per_type)
            edges.extend([(source_nodes[i], entity) for i in range(len(source_nodes)) for entity in target_nodes_per_type[i] if not len(target_nodes_per_type[i])==0])
        
        # Node labels
        linked_nodes = list(set([news_kg_util.qid2wikidata_resource(edge[1][1]) for edge in edges if edge[1][1]!='NIL']))
        linked_nodes_labels = self.get_wikidata_label([node for node in linked_nodes])
        linked_nodes_labels = [nlp_util.get_canonical_label(label) for label in linked_nodes_labels]
        linked_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[linked_nodes, linked_nodes_labels])}

        unlinked_nodes_labels = [nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='OTH')) for edge in edges if edge[1][1]=='NIL']
        unlinked_nodes = [news_kg_util.label2geneg_resource(label) for label in unlinked_nodes_labels]
        unlinked_nodes_w_labels = {item[0]: {self.ATTRIBUTE_LABEL: item[1]} for item in zip(*[unlinked_nodes, unlinked_nodes_labels])}

        # Node aliases
        nodes_aliases = [edge[1][0] for edge in edges]
        edges = [(edge[0], news_kg_util.qid2wikidata_resource(edge[1][1])) if edge[1][1]!='NIL' else (edge[0], news_kg_util.label2geneg_resource(nlp_util.get_canonical_label(nlp_util.clean_label(edge[1][0], entity_type='OTH')))) for edge in edges]
        target_nodes = [edge[1] for edge in edges]
        nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: nlp_util.normalize_unicodedata(item[1])} for item in zip(*[target_nodes, nodes_aliases])}
        linked_nodes_aliases = [self.wiki_attributes_map[node]['aliases'] for node in linked_nodes]
        linked_nodes_w_aliases = {item[0]: {self.ATTRIBUTE_ALIAS: item[1]} for item in zip(*[linked_nodes, linked_nodes_aliases]) if item[1]}

        # Add edges
        edges = [(edge[0], edge[1], rdf_util.PREDICATE_MENTIONS) for edge in list(set(edges))]
        self._add_edges(edges)
        
        # Set node attributes
        self._set_nodes_attributes(linked_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(unlinked_nodes_w_labels, set_label=True, set_alias=False)
        self._set_nodes_attributes(nodes_w_aliases, set_label=False, set_alias=True)
        self._set_nodes_attributes(linked_nodes_w_aliases, set_label=False, set_alias=True)

        utils.get_logger().debug(f'GeNeG: New graph size of {len(self.nodes)} nodes and {len(self.edges)} edges.')

    def _get_overlap_ratio(self, sequence1: str, sequence2: str) -> float:
        """ Returns a measure of sequences' similarity between [0, 1], given by the ratio of number of matches to the total number of elements in both sequences. """ 
        s = SequenceMatcher(None, sequence1, sequence2) 
        return s.ratio()

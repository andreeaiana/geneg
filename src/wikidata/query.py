# -*- coding: utf-8 -*-

import src.news_kg.nlp as nlp_util
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.entity import WikidataItem


def get_attr_dict(qid: str) -> dict:
    """ Retrieves a dictionary of attributes from Wikidata for a given entity QID. """
    attr_dict = get_entity_dict_from_api(qid.split('/')[-1])
    return attr_dict


def get_entity_attributes(attr_dict: dict) -> tuple:
    """ Returns a tuple (label, aliases) containing the lists of labels and aliases for a Wikidata entity. 
        For labels, the German Wikidata label is selected, if it exists, otherwise the English one. 
        If the German label differs from the English one, the English label is added as an alias.
        For aliases, it stores a set of both German and English Wikidata aliases.
    """
    aliases = set()
    
    # Set the label of the entity to the German label, if it exists, othrwise to the English one
    if 'labels' in attr_dict:
        if 'de' in attr_dict['labels'].keys():
            label = attr_dict['labels']['de']['value'] 
            
            # If an English label also exists, and it is different from the German one, add it as an alias, otherwise initialize alias list
            if 'en' in attr_dict['labels'].keys():
                en_label = attr_dict['labels']['en']['value']
                if not label == en_label:
                    aliases.add(en_label)
        elif 'en' in attr_dict['labels'].keys():
            label = attr_dict['labels']['en']['value']
        else:
            item = WikidataItem(attr_dict)
            label = item.get_label()
            if label == '':
                label = None
    else:
        label = None

    # Add German and English aliases to the map, if they exist
    if 'aliases' in attr_dict:
        if attr_dict['aliases']:
            de_aliases = [nlp_util.normalize_unicodedata(alias['value']) for alias in attr_dict['aliases']['de'] if not nlp_util.is_emoji(alias['value'])] if 'de' in attr_dict['aliases'].keys() else list()
            en_aliases = [nlp_util.normalize_unicodedata(alias['value']) for alias in attr_dict['aliases']['en'] if not nlp_util.is_emoji(alias['value'])] if 'en' in attr_dict['aliases'].keys() else list()
            aliases.update(de_aliases + en_aliases)

    return (label, aliases)

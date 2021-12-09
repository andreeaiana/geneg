# -*- coding: utf-8 -*-

""" Utilities for ReNewRS resources and properties. """

import utils

NAMESPACE_GENEG_BASE = utils.get_config('geneg.namespace.base')
NAMESPACE_GENEG_PROPERTY = utils.get_config('geneg.namespace.property')
NAMESPACE_GENEG_RESOURCE = utils.get_config('geneg.namespace.resource')
NAMESPACE_WIKIDATA_RESOURCE = 'https://www.wikidata.org/wiki/'
NAMESPACE_WIKIDATA_PROPERTY = 'https://www.wikidata.org/wiki/Property:'


def is_geneg_resource(item: str) -> bool:
    return type(item)==str and item.startswith(NAMESPACE_GENEG_RESOURCE)


def id2geneg_resource(node_id: str) -> str:
    return NAMESPACE_GENEG_RESOURCE + node_id


def geneg_resource2id(geneg_resource: str) -> str:
    return geneg_resource[len(NAMESPACE_GENEG_RESOURCE):]


def label2geneg_resource(label: str) -> str:
    return NAMESPACE_GENEG_RESOURCE + label


def geneg_resource2label(geneg_resource: str) -> str:
    return geneg_resource[len(NAMESPACE_GENEG_RESOURCE):]


def is_geneg_property(item: str) -> bool:
    return item.startswith(NAMESPACE_GENEG_PROPERTY)


def label2geneg_property(label: str) -> str:
    return NAMESPACE_GENEG_PROPERTY + label


def geneg_property2label(geneg_property: str) -> str:
    return geneg_property[len(NAMESPACE_GENEG_PROPERTY):]


def qid2wikidata_resource(qid: str) -> str:
    return NAMESPACE_WIKIDATA_RESOURCE + qid


def wikidata_resource2qid(wikidata_resource: str) -> str:
    return wikidata_resource.split(NAMESPACE_WIKIDATA_RESOURCE)[-1]


def is_wikidata_resource(item: str) -> bool:
    return type(item)==str and item.startswith(NAMESPACE_WIKIDATA_RESOURCE)


def pid2wikidata_property(pid: str) -> str:
    return NAMESPACE_WIKIDATA_PROPERTY + pid

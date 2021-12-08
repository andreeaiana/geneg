# -*- coding: utf-8 -*-

""" Serialization of statements into RDF triples """

import urllib.parse


TYPE_RESOURCE = 'type_resource'
POSTFIXES = {
        int: 'http://www.w3.org/2001/XMLSchema#integer'
        }
RESOURCE_ENCODED_CHARS = ['\\', '\'', '"', '´', '`', '{', '}', '^', ' ']
LITERAL_ENCODED_CHARS = ['\\', '\'', '"']


def as_literal_triple(subj: str, pred: str, obj) -> str:
    """ Serialize a triple as a literal triple. """
    obj_type = type(obj)
    if obj_type == str:
        obj = _encode_literal_string(obj)
    return _as_triple(subj, pred, obj, obj_type)


def as_object_triple(subj: str, pred: str, obj: str) -> str:
    """ Serialize a triple as an object triple. """
    return _as_triple(subj, pred, obj, TYPE_RESOURCE)


def _as_triple(subj: str, pred: str, obj: str, obj_type) -> str:
    if obj_type == TYPE_RESOURCE:
        obj_as_string = _resource_to_string(obj)
    else:
        obj_as_string = f'"{obj}"'
        if obj_type in POSTFIXES:
            obj_as_string += f'^^{_resource_to_string(POSTFIXES[obj_type])}'
    return f'{_resource_to_string(subj)} {_resource_to_string(pred)} {obj_as_string} .\n'


def _resource_to_string(resource: str) -> str:
    for c in RESOURCE_ENCODED_CHARS:
        resource = resource.replace(c, urllib.parse.quote_plus(c))
    return resource if resource.startswith('_:') else f'<{resource}>'


def _encode_literal_string(literal: str) -> str:
    for c in LITERAL_ENCODED_CHARS:
        literal = literal.replace(c, f'\\{c}')
    return literal

# -*- coding: utf-8 -*-

""" Functionality for handling RDF """

import src.news_kg.util as news_kg_util


# PREDICATES
PREDICATE_URL = 'https://schema.org/url'
PREDICATE_PUBLISHER = 'https://schema.org/publisher'
PREDICATE_DATE_PUBLISHED = 'https://schema.org/datePublished'
PREDICATE_DATE_MODIFIED = 'https://schema.org/dateModified'
PREDICATE_AUTHOR = 'https://schema.org/author'
PREDICATE_ABOUT = 'https://schema.org/about'
PREDICATE_HEADLINE = 'https://schema.org/headline'
PREDICATE_ABSTRACT = 'https://schema.org/abstract'
PREDICATE_ARTICLE_BODY = 'https://schema.org/articleBody'
PREDICATE_IS_BASED_ON = 'https://schema.org/isBasedOn'
PREDICATE_KEYWORDS = 'https://schema.org/keywords'
PREDICATE_MENTIONS = 'https://schema.org/mentions'
PREDICATE_HAS_PLACE = 'https://semanticweb.cs.vu.nl/2009/11/sem/hasPlace'
PREDICATE_HAS_ACTOR = 'https://semanticweb.cs.vu.nl/2009/11/sem/hasActor'
PREDICATE_POLARITY = news_kg_util.label2geneg_property('polarity') 
PREDICATE_IN_FAVOR = news_kg_util.label2geneg_property('in_favor')
PREDICATE_AGAINST = news_kg_util.label2geneg_property('against')
PREDICATE_TYPE = 'https://www.w3.org/1999/02/22-rdf-syntax-ns#type'
PREDICATE_LABEL = 'http://www.w3.org/2000/01/rdf-schema#label'


# Classes
CLASS_EVENT = 'https://semanticweb.cs.vu.nl/2009/11/sem/Event'
CLASS_NEWS_ARTICLE = 'https://schema.org/NewsArticle'
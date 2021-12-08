# -*- coding: utf-8 -*-
""" Basic utilities for data loading. """

import os
import utils
import pickle
import pandas as pd


def load_dataset() -> pd.DataFrame:
    """ Retrieves the dataset of news articles from cache, if created, otherwise creates dataset for KG. """
    utils.get_logger().info('Data loading: Retrieving dataset of news articles..')
    global __DATASET__
    if '__DATASET__' not in globals():
        initiallizer = lambda: _merge_dataset_polarity_scores()
        __DATASET__ = utils.load_or_create_cache('dataset', initiallizer)
    utils.get_logger().info('Data loading: Loaded.\n')
    return __DATASET__
        

def _load_dataset_with_entities() -> pd.DataFrame:
    """ Loads dataset of news articles annotated with named entities. """
    utils.get_logger().debug('Data loading: Retrieving dataset annotated with named entities..')

    file = utils.get_dataset_file('files.dataset')
    if not os.path.isfile(file):
        raise FileNotFoundError(f'File {file} does not exist.')
    
    with open(file, 'rb') as f:
        dataset = pickle.load(f)
    utils.get_logger().debug(f'Data loading: Size of dataset with named entities {dataset.shape}.')

    return dataset


def _load_polarity_scores() -> pd.DataFrame:
    """ Loads the polarity scores of the news articles. """
    utils.get_logger().debug('Data loading: Retrieving polarity scores..')
    
    file = utils.get_polarity_scores_file('files.polarity_scores')
    if not os.path.isfile(file):
        raise FileNotFoundError(f'File {file} does not exist.')

    polarity_scores = pd.read_csv(file)
    polarity_scores.drop(columns=[col for col in list(polarity_scores.columns) if not col in ['provenance', 'sentiment_score']], inplace=True)
    utils.get_logger().debug(f'Data loading: Size of polarity scorees {polarity_scores.shape}')

    return polarity_scores


def _merge_dataset_polarity_scores() -> pd.DataFrame:
    """ Merges the dataset of news articles annotated with named entities with the pre-calculated polarity scores. """
    dataset = _load_dataset_with_entities()
    polarity_scores = _load_polarity_scores()

    utils.get_logger().debug('Data loading: Adding polarity scores to dataset..')
    return pd.merge(dataset, polarity_scores, how='left', on='provenance')


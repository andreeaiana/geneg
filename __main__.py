# -*- coding: utf-8 -*-
 
import utils
import traceback
from src import news_kg


if __name__ == '__main__':
    try:
        utils.get_logger().info('Starting extraction of base GeNeG.')
        news_kg.serialize_final_graph(graph_type='base')
        utils.get_logger().info('Finished constructing the base GeNeG.')

        utils.get_logger().info('Constructing entities graph from base GeNeG.')
        news_kg.serialize_final_graph(graph_type='entities')
        utils.get_logger().info('Finished constructing the entities GeNeG.')

        utils.get_logger().info('Constructing complete graph from base GeNeG.')
        news_kg.serialize_final_graph(graph_type='complete')
        utils.get_logger().info('Finished constructing the complete GeNeG.')

    except Exception as e:
        error_msg = traceback.format_exc()
        utils.get_logger().error(error_msg)
        raise e

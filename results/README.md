# Results

GeNeG is serialized in [gzipped](https://www.gzip.org/) files in [N-Triples](https://www.w3.org/TR/n-triples/) format. 

| File | Description | Base GeNeG | Entities GeNeG | Complete GeNeG |
|------|-------------|------------|----------------|----------------|
| geneg_*type*-metadata.nt.bz2     |    Metadata about the dataset, described using void vocabulary.         |   &check;     |          &check;      | &check;     |
| geneg_*type*-instances_types.nt.bz2     |     Class definitions of articles and events.        |     &check;       |         &check;       |        &check;        |
| geneg_*type*-instances_metadata_literals.nt.bz2   |   Relations between news article resurces and metadata literals (e.g. URL, publishing date, modification date, polarity score).        |     &check;      |                |       &check;         |
|   geneg_*type*-instances_metadata_resources.nt.bz2   |  Relations between news article resources and metadata entities (i.e. publishers, authors, keywords).           |  &check;          |         &check;       |        &check;        |
|   geneg_*type*-instances_content_relations.nt.bz2   |  Relations between news article resources and content components (e.g. titles, abstracts, article bodies).      |    &check;       |                |        &check;        |
|   geneg_*type*-instances_event_mapping.nt.bz2   |   Mapping of news article resources to events.          |    &check;        |         &check;       |          &check;      |
|    geneg_*type*-event_relations.nt.bz2  |   Relations between news events and entities mentioned (i.e. actors, places, mentions).        |    &check;        |         &check;      |     &check;           |
|   geneg_*type*-wiki_relations.nt.bz2   |    Relations between news event Wikidata entities and their *k*-hop entities neighbors from Wikidata.        |            |      &check;          |       &check;         |
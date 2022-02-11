# GeNeG
GeNeG is a knowledge graph constructed from news articles on the topic of refugees and migration, collected from German online media outlets. GeNeG contains rich textual and metadata information, as well as named entities extracted from the articles' content and metadata and linked to Wikidata. The graph is expanded with up to three-hop neighbors from Wikidata of the initial set of linked entities. 

### Features
GeNeG comes in three flavors:
- **Base GeNeG**: contains textual information, metadata, and entities extracted from the articles.
- **Entities GeNeG**: derived from the Base GeNeG by removing all literal nodes, it contains only resources and it is enriched with three-hop Wikidata neighbors of the entities extracted from the articles.
- **Complete GeNeG**: the combination of the Base and Entities GeNeG, it contains both literals and resources.

### Statistics
| Statistics          | Base GeNeG | Entities GeNeG | Complete GeNeG |
|---------------------|------------|----------------|----------------|
| #Nodes              | 54,332     | 844,935        | 868,164        |
| #Edges              | 209,369    | 6,615,972      | 6,679,564      |
| #Properties         | 18         | 1,263          | 1,273          |
| Avg.degree          | 3.85       | 7.83           | 7.69           |
| Resources frequency | 40.47      | 1.0            | 96.27          |
| Literals frequency  | 59.53      | 0.0            | 3.73           |
## Usage

### Basic Configuration Options
Application-specific parameters, logging, and file-related settings can be configured in `config.yaml`.


### Knowledge Graph Construction
Run the news knowledge graph construction
```
python3 .
```
All three versions of GeNeG will be constructed. GeNeG is serialized in N-Triple format. The resulting files are placed in the `results` folder.

## Data
A sample of  annotated news corpus and the polarity scores used to construct the knowledge graph are available in the `data/dataset` folder. Due to copyright policies, this sample does not contain the abstract and body of the articles.

A full version of the news corpus is available [upon request](mailto:andreea@informatik.uni-mannheim.de).

## Results
The [complete GeNeG](https://doi.org/10.5281/zenodo.6039372) is hosted on [Zenodo](https://zenodo.org/). All files are [gzipped](https://www.gzip.org/) and in [N-Triples format](https://www.w3.org/TR/n-triples/). 

A sample of the three versions of GeNeG can also be found in the `results` folder, together with a corresponding [description](results/README.md). Due to copyright policies, this sample does not contain the abstract and body of the articles.


## Requirements
This code is implemented in Python 3. The requirements can be installed from `requirements.txt`.

```
pip3 install -r requirements.txt
```

## License
The code is licensed under the MIT License. The data and knowledge graph files are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

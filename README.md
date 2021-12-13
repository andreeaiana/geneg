# GeNeG
GeNeG is a knowledge graph constructed from news articles on the topic of refugees and migration, collected from German online media outlets. GeNeG contains rich textual and metadata information, as well as named entities extracted from the articles' content and metadata and linked to Wikidata. The graph is expanded with up to three-hop neighbors from Wikidata of the initial set of linked entities. 

### Features
GeNeG comes in three flavours:
- **Base GeNeG**: contains testual information, metadata, and linked entities extracted from the articles.
- **Entities GeNeG**: derived from the Base GeNeG by removing all literal nodes, it contains only resources and it is enriched with three-hop Wikidata neighbors of the entities extracted from the articles.
- **Complete GeNeG**: the combination of the Base and Entities GeNeG, it contains both literals and resources.

### Statistics
| Statistics          | Base GeNeG | Entities GeNeG | Complete GeNeG |
|---------------------|------------|----------------|----------------|
| #Nodes              | 54,327     | 844,935        | 868,159        |
| #Edges              | 186,584    | 6,615,972      | 6,656,779      |
| #Properties         | 8          | 1,263          | 1,271          |
| Avg.degree          | 3.43       | 7.83           | 7.67           |
| Resources frequency | 57.25      | 1.0            | 96.28          |
| Literals frequency  | 42.75      | 0.0            | 3.72           |
## Usage

### Basic Configuration Options
Application-specific parameters, logging, and file-related settings can be configured in `config.yaml`.


### Knowledge graph construction
Run the news knowledge graph construction
```
python3 .
```
All three versions of GeNeG will be constructed. GeNeG is serialized in N-Triple format. The resulting files are placed in the `results` folder.

## Data
A sample of  annotated news corpus and the polarity scores used to construct the knowledge graph are available in the `data/dataset` folder. Due to copyright policies, this sample does not contain the abstract and body of the articles.

A full version of the news corpus is available [upon request](mailto: andreea@informatik.uni-mannheim.de).

## Results
The gzipped N-Triples files for all three versions of GeNeG can be found in the `results` folder, together with a corresponding [description](results/README.md). Due to copyright issues, these do not include information about the abstract and bodies of the news articles. 

The files correspnding to the full version of GeNeG are avaiable [upon request](mailto: andreea@informatik.uni-mannheim.de).


## Requirements
This code is implemented in Python 3. The requirements can be installed from `requirements.txt`.

```
pip3 install -r requirements.txt
```

## License
Licensed under the MIT License.

# Embedding dictionaries 
The embeddings used for the evaluation of this paper can be found in its corresponding repositories on Zenodo. Embeddings are stored as an NPZ file, which encodes a dictionary where the key is the SNOMED CT identifier and the value is the corresponding embedding. Take into consideration that the dimension of the embedding varies according to the encoder used.

These embeddings can be easily loaded into a Python dictionary using the _load_embeddings_ function from _embedding_models.embedding_model.py_.

- Embeddings from FastText using MIMIC-IV or SNOMED CT walks: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18220275.svg)](https://doi.org/10.5281/zenodo.18220275)
- Embeddings from base sentence BERT or fine-tuned sentence BERT on concept similarity: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18253463.svg)](https://doi.org/10.5281/zenodo.18253463)
- Embeddings from graph neural network for SNOMED CT concepts: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18253463.svg)](https://doi.org/10.5281/zenodo.18253463)
- Embeddings from LLM2vec for SNOMED CT concepts: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18455301.svg)](https://doi.org/10.5281/zenodo.18455301)

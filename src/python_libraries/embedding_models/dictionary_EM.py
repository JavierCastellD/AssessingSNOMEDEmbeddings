import numpy as np

from .embedding_model import EmbeddingModel, load_embeddings

class DictionaryEM(EmbeddingModel):
    '''Class that represents an embedding model whose embeddings were previously generated and saved as an npz
    file, such as those from the sentence transformers, GNNs, LLM2Vec or knowledge graph embeddings. The
    embeddings are obtained from the SNOMED CT identifiers, which are the keys of the dictionary.

    Attributes:
        model (dict):
            Dictionary that links SNOMED CT identifiers (as strings) to embeddings.
    '''
    def __init__(self, model_path : str):
        '''Method to load the dictionary of embeddings.

        Parameters:
            model_path (str):
                Path to the npz file that contains the embeddings.
        '''
        super().__init__(model_path)

    def train_model(self, corpora: list[str]):
        '''Method to train the model. It does nothing, as the embeddings were generated beforehand.

        Parameters:
            corpora (list):
                List of paths from which to train the model. It is ignored.

        Returns:
            None.
        '''
        return None

    def save_model(self, model_path: str):
        '''Method to save the dictionary of embeddings as an npz file.

        Parameters:
            model_path (str):
                Path to where the dictionary should be saved.
        '''
        ids = np.array(list(self.model.keys()))
        vectors = np.array(list(self.model.values()), dtype=np.float32)
        np.savez_compressed(model_path, ids=ids, vectors=vectors)

    def load_model(self, model_path : str):
        '''Method to load the dictionary of embeddings.

        Parameters:
            model_path (str):
                Path to the npz file that contains the embeddings.

        Returns:
            A dictionary that links SNOMED CT identifiers to embeddings.
        '''
        return load_embeddings(model_path)

    def get_embedding(self, word : str):
        '''Method to obtain the embedding of a SNOMED CT identifier.

        Parameters:
            word (str):
                SNOMED CT identifier of the concept or relationship.

        Returns:
            A numpy array with the embedding.
        '''
        return np.array(self.model[str(word)])

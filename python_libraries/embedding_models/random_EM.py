import numpy as np

from .embedding_model import EmbeddingModel

SEED = 1234

class RandomEM(EmbeddingModel):
    '''Class that represents a random embedding model. It returns embeddings with random float values
    between -1 and 1, and ignores the input it receives.
    
    Attributes:
        vector_size (int):
            Dimensions of the embeddings produced by the model. This is used when the model is trained.
    '''
    def __init__(self, vector_size : int = 200):
        '''Method to initialize the random embedder.
        
        Parameters:
            vector_size (int):
                Dimensions of the embeddings produced by the model.    
        '''
        self.vector_size = vector_size
        super().__init__('', '')

    def train_model(self, corpora: list[str]):
        '''Method to train the model. It does nothing.
        
        Parameters:
            corpora (list):
                List of paths from which to train the model. It is ignored.
        
        Returns:
            None.
        '''
        return None

    def save_model(self, model_path: str):
        '''Method to save the model. It does nothing.
        
        Parameters:
            model_path (str):
                Path to where FastText should be saved. It is ignored.
        '''
        pass

    def load_model(self, model_path : str):
        '''Method to load a FastText model.
        
        Parameters:
            model_path (str):
                Path from where the FastText model to be loaded can be found.
        '''
        return None

    def get_embedding(self, word : str):
        '''Method to obtain an embedding for a given string. The input is ignored and instead
        a random embedding of size vector_size is returned.
        
        Parameters:
            word (str):
                String that represents the word(s) from which to obtain the embedding.
        
        Returns:
            A numpy array of size vector size with random values between -1 and 1.
        '''
        return np.random.uniform(-1, 1, self.vector_size)
    
    def get_embedding_from_list(self, names_list : list[str], agg_average : bool = True):
        '''Method to obtain an embedding by aggregating the embedding of different words. If agg_average is 
        set to True, the aggregation is done by averaging the embeddings. Otherwise, the aggregation is done
        by a sum. The input will be ignored and instead a random embedding of size vector_size is returned.
        
        Parameters:
            names_list (list):
                List of Strings that represents the words from which to obtain the embedding.
            
            agg_average (bool):
                Whether to aggregate by averaging or summing.
        Returns:
            A numpy array of size vector size with random values between -1 and 1.
        '''
        return np.random.uniform(-1, 1, self.vector_size)
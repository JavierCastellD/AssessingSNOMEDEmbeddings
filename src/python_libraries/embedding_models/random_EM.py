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
        # Create a dictionary to store the random embeddings of already generated words
        self.vector_dict = {}  
        self.vector_size = vector_size
        # Random generator with a fixed seed, so that the embeddings are reproducible
        self.rng = np.random.default_rng(SEED)
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
                Path to where the model should be saved. It is ignored.
        '''
        pass

    def load_model(self, model_path : str):
        '''Method to load the model. It does nothing.
        
        Parameters:
            model_path (str):
                Path from where the model should be loaded. It is ignored.
        
        Returns:
            None.
        '''
        return None

    def get_embedding(self, word : str, use_store : bool = True):
        '''Method to obtain an embedding for a given string. The input is ignored and instead
        a random embedding of size vector_size is returned.
        
        Parameters:
            word (str):
                String that represents the word(s) from which to obtain the embedding.
            use_store (bool):
                Whether to always return the same embedding for the same word.
        
        Returns:
            A numpy array of size vector size with random values between -1 and 1.
        '''
        if not use_store:
            return self.rng.uniform(-1, 1, self.vector_size)
        
        if word not in self.vector_dict:
            self.vector_dict[word] = self.rng.uniform(-1, 1, self.vector_size)
            
        return self.vector_dict[word]
            
    
    def get_embedding_from_list(self, names_list : list[str], use_store : bool = True, agg_average : bool = True):
        '''Method to obtain an embedding by aggregating the embedding of different words. The input is 
        ignored and instead a random embedding of size vector_size is returned.
        
        Parameters:
            names_list (list):
                List of Strings that represents the words from which to obtain the embedding.
            use_store (bool):
                Whether to always return the same embedding for the same list of words.
            agg_average (bool):
                Whether to aggregate by averaging or summing. It is ignored.
        
        Returns:
            A numpy array of size vector size with random values between -1 and 1.
        '''
        if not use_store:
            return self.rng.uniform(-1, 1, self.vector_size)
        
        # The list is stored as a tuple, since lists cannot be used as keys
        key = tuple(names_list)
        if key not in self.vector_dict:
            self.vector_dict[key] = self.rng.uniform(-1, 1, self.vector_size)
            
        return self.vector_dict[key]

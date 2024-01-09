from abc import ABC, abstractmethod
import warnings

class EmbeddingModel(ABC):
    '''Abstract class that represents an embedding model, such as BERT, FastText, etc. 
    The methods to be implemented are train_model, save_model, load_model and get_embedding.
    
    Atributes:
        model:
            Language model prepared to be used to obtain embeddings from it.
    '''
    def __init__(self, model_path : str = None, corpora : list[str] = None):
        '''Loads the model from a certain path if it was provided. Otherwise, if a corpora was given, trains the model. 
        The method expects either model_path or corpora to be provided.
        
        Parameters:
            model_path (str):
                Path to the language model to be loaded.
            corpora (list):
                List that contains the path to the corpora from which the model will be trained.
        '''
        if model_path is not None:
            self.model = self.load_model(model_path)
        elif corpora is not None: 
            self.model = self.train_model(corpora)
        else:
            warnings.warn('Neither model path to load nor corpora to train were given.')

    @abstractmethod
    def train_model(self, corpora : list[str]):
        '''Method to train the model from a list of corpus.
        
        Parameters:
            corpora (list):
                List of paths from which to train the model.
        
        Returns:
            A trained language model.
        '''
        pass

    @abstractmethod
    def save_model(self, model_path : str):
        '''Method to save the model.
        
        Parameters:
            model_path (str):
                Path to where the model should be saved.
        '''
        pass

    @abstractmethod
    def load_model(self, model_path : str):
        '''Method to load the model.
        
        Parameters:
            model_path (str):
                Path from where the model to be loaded can be found.
        '''
        pass

    @abstractmethod
    def get_embedding(self, word : str):
        '''Method to obtain an embedding for a given string, which can be either a single word or multiple ones.
        
        Parameters:
            word (str):
                String that represents the word(s) from which to obtain the embedding.
        
        Returns:
            An embedding.
        '''
        pass

    def get_embedding_from_list(self, names_list : list, agg_average : bool = True):
        '''Method to obtain an embedding by aggregating the embedding of different words. If agg_average is 
        set to True, the aggregation is done by averaging the embeddings. Otherwise, the aggregation is done
        by a sum.
        
        Parameters:
            names_list (list):
                List of Strings that represents the words from which to obtain the embedding.
            
            agg_average (bool):
                Whether to aggregate by averaging or summing.
        Returns:
            An embedding.
        '''
        vectors = []

        if len(names_list) == 0:
            return []

        for name in names_list:
            vectors.append(self.get_embedding(name))
        
        if agg_average:
            return sum(vectors)/len(vectors)
        else:
            return sum(vectors)
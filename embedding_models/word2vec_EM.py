import nltk
import warnings
import numpy as np
from embedding_models.embedding_model import EmbeddingModel
from gensim.models.word2vec import Word2Vec
from preprocessing.preprocess_em import PreprocessEM
from preprocessing.preprocess_gensim_sct import PreprocessGensimSCT

SEED = 1234

class Word2VecEM(EmbeddingModel):
    '''Class that represents a Word2Vec embedding model. The Word2Vec model is described here: https://arxiv.org/abs/1301.3781.
    
    Attributes:
        model (Word2Vec):
            Gensim implementation of Word2Vec.
        vector_size (int):
            Dimensions of the embeddings produced by the model. This is used when the model is trained.
        language (str):
            Language of the corpora from which to train the model.
        preprocess_pipeline (PreprocessEM):
            PreprocessEM object to preprocess the text used for training the model and creating the embedding.

    '''
    def __init__(self, vector_size : int = 200, language : str = 'english', model_path: str = None, corpora: list[str] = None,
                 preprocess_pipeline : PreprocessEM = PreprocessGensimSCT()):
        '''Method to load the Word2Vec model. If model_path is given as parameter, the model will be loaded. Otherwise, the model
        will be trained from the corpora parameter. The method expects to receive at least one of those parameters.
        
        Parameters:
            vector_size (int):
                Dimensions of the embeddings produced by the model.
            language (str):
                Language of the corpora from which the model is trained.
            model_path (str):
                Path to the model.
            corpora (list):
                List of corpus from which to train the model.        
            preprocess_pipeline (PreprocessEM):
                PreprocessEM object to preprocess the text used for training the model and creating the embedding.      
        '''
        self.vector_size = vector_size
        self.language = language
        self.preprocess_pipeline = preprocess_pipeline
        super().__init__(model_path, corpora)

    def train_model(self, corpora: list[str]):
        '''Method to train a Word2Vec model from a list of corpus. Each corpus is preprocessed using
        preprocessed_text.
        
        Parameters:
            corpora (list):
                List of paths from which to train the model.
        
        Returns:
            A trained Word2Vec model.
        '''
        sentences = self.preprocess_pipeline.preprocess_text(corpora)

        ft_model = Word2Vec(sentences=sentences, sg=1, seed=SEED, vector_size=self.vector_size,
                            window=5, min_count=1)

        return ft_model

    def save_model(self, model_path: str):
        '''Method to save the Word2Vec model.
        
        Parameters:
            model_path (str):
                Path to where Word2Vec should be saved.
        '''
        self.model.save(model_path)

    def load_model(self, model_path : str):
        '''Method to load a Word2Vec model.
        
        Parameters:
            model_path (str):
                Path from where the Word2Vec model to be loaded can be found.
        '''
        return Word2Vec.load(model_path)
    
    def get_embedding(self, word : str):
        '''Method to obtain an embedding for a given string, which can be either a single word or multiple ones.
        Before obtaining the embedding, the string is preprocessed. This preprocessing should be the same as the one
        used for the corpora. If multiple words are in the input string, the average of their embeddings is returned.
        
        Parameters:
            word (str):
                String that represents the word(s) from which to obtain the embedding.
        
        Returns:
            An embedding.
        '''
        name = self.preprocess_pipeline.preprocess_sentence(word)
        words_in_name = nltk.word_tokenize(name, language=self.language)
        vectors = []

        for word in words_in_name:
            # We have to check if the word is part of the vocabulary
            if word in self.model.vocab:
                vectors.append(self.model.wv.get_vector(word))

        if len(vectors) > 0:
            v = sum(vectors)/len(vectors)
        else:
            warnings.warn("No word from", word, "could be found in the vocabulary for the trained Word2Vec model.")
            v = np.zeros(self.vector_size)
            
        return v
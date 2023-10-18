from embedding_model import EmbeddingModel
from gensim.models.fasttext import FastText
import re
import numpy as np
import nltk

SEED = 1234

def preprocess_sentence(sentence : str):
    '''Method to preprocess a sentence. The sentence is lowered, special symbols are separated (such as : or -),
    values between parenthesis are removed (this is because of SNOMED CT), and double spaces are deleted.
    
    Parameters:
        sentence (str):
            String that represents the sentence to be preprocessed.
    
    Returns:
        A preprocessed string. 
    '''
    # Lower the text
    s = sentence.lower()

    # Separate certain symbols
    symbols = ['(',')','.','[',']',':','-','/']
    for symb in symbols:
        s = s.replace(symb, ' ' + symb + ' ')

    # Remove words between parenthesis
    s = re.sub('\(.+\)', ' ', s)

    # Remove double spaces
    s = re.sub(' +', ' ', s)
    s = s.strip()

    return s

# TODO: Podríamos permitir al usuario que definiera sus propios métodos
#  para preprocesar y que los pasara como parámetros, de manera que estos
#  fueran solo la versión por defecto
def preprocess_text(list_sentences : list[str], language : str):
    '''Method to preprocess a text. Each sentence is preprocessed using the preprocess_sentence method. 
    Each sentence is tokenized into words after preprocessing, as expected for gensim's implementation
    of FastText.
    
    Parameters:
        list_sentences (list):
            List of sentences to be preprocessed.
        
        language (str):
            Language of the corpus.

    Returns:
        A list of tokenized and preprocessed sentences.    
    '''
    sentences = []

    for sentence in list_sentences:
        s = preprocess_sentence(sentence)

        # Tokenize the sentence into words
        s = nltk.word_tokenize(s, language=language)

        sentences.append(s)

    return sentences

class FastTextEM(EmbeddingModel):
    '''Class that represents a FastText embedding model. The FastText model is described here: https://arxiv.org/abs/1607.04606.
    
    Attributes:
        model (FastText):
            Gensim implementation of FastText.
        vector_size (int):
            Dimensions of the embeddings produced by the model. This is used when the model is trained.
        language (str):
            Language of the corpora from which to train the model.
    '''
    def __init__(self, vector_size : int = 200, language : str = 'english', model_path: str = None, corpora: list[str] = None):
        '''Method to load the FastText model. If model_path is given as parameter, the model will be loaded. Otherwise, the model
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
        '''
        self.vector_size = vector_size
        self.language = language
        super().__init__(model_path, corpora)

    def train_model(self, corpora: list[str]):
        '''Method to train a FastText model from a list of corpus. Each corpus is preprocessed using
        preprocessed_text.
        
        Parameters:
            corpora (list):
                List of paths from which to train the model.
        
        Returns:
            A trained FastText model.
        '''
        sentences_corpus = []
        for corpus in corpora:
            with open(corpus, encoding='utf8') as corpus_file:
                sentences_corpus += corpus_file.readlines()
        
        sentences = preprocess_text(sentences_corpus, self.language)

        ft_model = FastText(sentences=sentences, sg=1, seed=SEED, vector_size=self.vector_size,
                                          window=5, min_count=1)

        return ft_model

    def save_model(self, model_path: str):
        '''Method to save the FastText model.
        
        Parameters:
            model_path (str):
                Path to where FastText should be saved.
        '''
        self.model.save(model_path)

    def load_model(self, model_path : str):
        '''Method to load a FastText model.
        
        Parameters:
            model_path (str):
                Path from where the FastText model to be loaded can be found.
        '''
        return FastText.load(model_path)

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
        name = preprocess_sentence(word)
        words_in_name = nltk.word_tokenize(name, language=self.language)
        vectors = []

        for word in words_in_name:
            vectors.append(self.model.wv.get_vector(word))

        if len(vectors) > 0:
            v = sum(vectors)/len(vectors)
        else:
            v = np.zeros(self.vector_size)
            
        return v
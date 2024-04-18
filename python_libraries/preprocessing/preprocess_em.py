from abc import ABC, abstractmethod

class PreprocessEM(ABC):
    '''Abstract class that represents a preprocessing pipeline for sentences and texts. This preprocessing is 
    thought to be used for an embedding model. The methods to be implemented are preprocess_sentence and 
    preprocess_text.
    
    Atributes:
        language (str):
            Language of the text. This attribute can then be used in the preprocessing methods.
    '''

    def __init__(self, language : str = 'english'):
        '''Sets the language of the preprocessing pipeline.

        Parameters:
            language (str):
                Language of the text to be preprocessed.
        '''
        self.language = language

    @abstractmethod
    def preprocess_sentence(self, sentence : str) -> str:
        '''Method to preprocess a sentence. This is usually used as preprocessing before getting an
        embedding as well as for preprocessing the text.
        
        Parameters:
            sentence (str):
                String that represents a sentence.

        Returns:
            A preprocessed string.
        '''
        pass


    @abstractmethod
    def preprocess_text(self, corpora : list[str], text : str = None):
        '''Method to preprocess the texts extracted from the list of corpora.
        If text is not None, this method can be used to preprocessed a loaded text.
        
        Parameters:
            corpora (list):
                List of paths from which to load the texts to be preprocessed.
            text (str):
                Text to be preprocessed, if it is not needed to load the text from a file.

        Returns:
            The text preprocessed. 
        '''
        pass


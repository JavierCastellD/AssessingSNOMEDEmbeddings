from preprocess_em import PreprocessEM
import nltk
import re

class PreprocessGensimSCT(PreprocessEM):
    '''Class that represents the preprocessing pipeline for gensim's implementation of Word2Vec and FastText for
    texts obtained from random walks on SNOMED CT.
    
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

    def preprocess_sentence(self, sentence : str) -> str:
        '''Method to preprocess a sentence. The sentence is lowered, special symbols are separated 
        (such as : or -), values between parenthesis are removed (this is because of SNOMED CT), and 
        double spaces are deleted.
        
        Parameters:
            sentence (str):
                String that represents a sentence.

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


    def preprocess_text(self, corpora : list[str], text : str = None):
        '''Method to preprocess the texts extracted from the list of corpora.
        If text is not None, this method can be used to preprocessed a loaded text.
        Each sentence is tokenized into words after preprocessing, as expected for gensim's implementation
        of Word2Vec and FastText.
        
        Parameters:
            corpora (list):
                List of paths from which to load the texts to be preprocessed.
            text (str):
                Text to be preprocessed, if it is not needed to load the text from a file.

        Returns:
            A list of tokenized and preprocessed sentences.    
        '''
        sentences_corpus = []
        for corpus in corpora:
            with open(corpus, encoding='utf8') as corpus_file:
                sentences_corpus += corpus_file.readlines()

        for sentence in sentences_corpus:
            s = self.preprocess_sentence(sentence)

            # Tokenize the sentence into words
            s = nltk.word_tokenize(s, language=self.language)

            sentences_corpus.append(s)

        return sentences_corpus


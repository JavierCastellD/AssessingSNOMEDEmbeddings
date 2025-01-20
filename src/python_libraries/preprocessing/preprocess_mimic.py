import nltk
import re
from re import Match

from .preprocess_em import PreprocessEM

def remove_break(match : Match):
    """Auxiliary function used for only substituting break lines for spaces from the regex expression.
    
    Parameters:
        match (Match):
            Match object detected from re.sub method.
    Returns:
        A string.
    """
    return ' ' + match.group(0)[2:]

class PreprocessGensimMIMIC(PreprocessEM):
    '''Class that represents the preprocessing pipeline for gensim's implementation of Word2Vec and FastText 
    for texts from MIMIC.
    
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
        (such as : or -), and double spaces are deleted.
        
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
        tokens = []
        for corpus in corpora:
            with open(corpus, encoding='utf8') as corpus_file:
                text = corpus_file.read()

            ### REMOVING SPECIFIC FIELDS ###
            # Remove the pertinent results 
            text = re.sub('Pertinent Results:(\n|.)*___ \d+:\d+.*___','', text)

            # Remove the part prior to allergies (empty information about name...)
            text = re.sub('\s+(.|\n)+ \n(?=Allergies:\s*)', '', text)

            # Remove the information about attending if it's empty
            text = re.sub('Attending.+___.', '', text)

            ### FIXING ERRORS IN FORMATING ###
            # Remove lines breaks inside text, except for enumerations
            text = re.sub(' \n(?!\d+\.).+?', remove_break, text)

            # Remove anonymized words
            text = re.sub('___', '', text)

            # Remove consecutive spaces
            text = re.sub(' +', ' ', text)

            # Remove multiple line breaks
            text = re.sub('\s+\n', ' \n', text)

            # Remove spaces after line break
            text = re.sub('\n ', '\n', text)

            # Remove ending spaces
            text = text.strip()

            # Split the processed text into sentences and tokenize it
            for sentence in text.split('\n'):
                s = self.preprocess_sentence(sentence)
                
                s = nltk.word_tokenize(s, language=self.language)
                tokens.append(s)

        return tokens


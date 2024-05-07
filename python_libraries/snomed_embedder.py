from collections import Counter
import warnings

import faiss
import numpy as np

from .embedding_models.embedding_model import EmbeddingModel
from .snomed import Snomed, IS_A_ID

ARBITRARY_LARGE_NUMBER = 1000

def list_contains_different_rel(relationship_list : list[(int, int)], different_rel : int):
    """Method that returns whether a list of tuples (relationship_id, concept_id) contains at least one instance of a relationship that is different from different_rel.
    
    Parameters:
        relationship_list (list[(int, int)]):
            A list of tuples.
        different_rel (int):
            ID a SNOMED CT relationship.
    
    Returns:
        True if the list contains a relationship that is different from different_rel. Otherwise returns false.
    """
    for relationship_id, _ in relationship_list:
        if relationship_id != different_rel:
            return True

    return False

def order_by_frequency(strings : list[str]):
    """Method that orders a list of strings by the frequency of the strings contained in it. Returns a new ordered list that contains no duplicates.
    
    Parameters:
        strings (list[str]):
               List of strings. 

    Returns:
        A list that is ordered by the frequency of the strings and that contains no duplicates.
    """
    # Count occurrences of each string
    string_count = Counter(strings)
    
    # Sort strings by their frequencies in descending order
    ordered_strings = sorted(string_count, key=lambda x: string_count[x], reverse=True)
    
    return ordered_strings

class SnomedEmbedder:
    """Class that performs operations on SNOMED CT using embeddings.

    Attributes:
        snomed (Snomed):
            Snomed object that contains the information about SNOMED CT.
        
        embedding_model (EmbeddingModel):
            Embedding model used for generating embeddings of each concept.
    
        embedding_values (list):
            List of embeddings, one per each SNOMED CT concept. The embedding at position i corresponds to the concept denoted by the ID
            at position i in embedding_sct_index.
        
        embedding_sct_index (list):
            List of SNOMED CT concept IDs.

        sct2index (dict):
            Dictionary that associates each SNOMED CT concept with its corresponding index position in the other lists.
    """
    def __init__(self, snomed : Snomed, embedding_model : EmbeddingModel, embedding_dictionary : dict = None):
        """Prepares the class by storing the SNOMED CT model, the embedding model, and prepares the dictionary needed 
        for some of the functionality. If no dictionary is given to the concept, one is generated by iterating over
        the concepts, obtaining their descriptions and creating the embedding for a concept as an average of the embedding of
        the descriptions.
        
        Parameters:
            snomed (Snomed):
                Snomed object that contains the information about SNOMED CT.
            embedding_model (EmbeddingModel):
                Represents an embedding model.
            embedding_dictionary (dict):
                Dictionary whose keys are IDs of SNOMED CT concepts and the values are their corresponding embeddings.
        """
        
        self.snomed = snomed
        self.embedding_model = embedding_model

        # If no embedding dictionary for SNOMED CT was given, we create our own
        if embedding_dictionary is None:
            embedding_dictionary = dict()

            for concept_id in snomed.get_sct_concepts(metadata=False):
                descriptions = snomed.get_descriptions(concept_id)
                
                embedding_dictionary[concept_id] = embedding_model.get_embedding_from_list(descriptions)
        
        # We keep a list of the embeddings of each SNOMED CT concept, as well as a list with their corresponding IDs
        self.embedding_values = list(embedding_dictionary.values())
        self.embedding_sct_index = list(embedding_dictionary.keys())

        self.sct2index = {}
        for i in range(len(self.embedding_sct_index)):
            concept_id = self.embedding_sct_index[i]
            self.sct2index[i] = concept_id

        # We use Faiss for fast similarity search, so we need to normalize the embeddings first
        faiss_embeddings = np.array(self.embedding_values).astype(np.float32)
        faiss.normalize_L2(faiss_embeddings)

        # Because we want to use cosine similarity, we need to define the metric as inner product
        self.faiss_index = faiss.index_factory(200, "Flat", faiss.METRIC_INNER_PRODUCT)
        self.faiss_index.ntotal

        self.faiss_index.add(faiss_embeddings)

    
    def get_most_similar_concept(self, word : str, n : int = 1):
        """Method that returns the most similar concept to the string that receives as a parameter. This is
        done by performing cosine similarity between embeddings.
        
        Paramters:
            word (str):
                String of text from which to obtain the most similar concept.
            n (int):
                Number of similar concepts to retrieve. The default value is 1.
        Returns:
            Returns a list of tuples (SCT-ID, sim_value), where the first element is the ID of the SNOMED concept, and the second
            is the similarity score.
        """
        # Obtain the embedding for the word
        vector = self.embedding_model.get_embedding(word)

        # Normalize the vector
        q_vector = np.array([vector]).astype(np.float32)
        faiss.normalize_L2(q_vector)

        # Perform the search
        sim_values, index = self.faiss_index.search(q_vector, n)

        # Obtain the corresponding concept_ids from SNOMED
        sim_list = [(self.embedding_sct_index[i], sim_val) for sim_val, i in zip(sim_values[0], index[0])]

        # Return a list of tuples (concept_id, similarity)
        return sim_list
        
    def get_postcoordinated_expression(self, clinical_term : str):
        """Method that returns the postcoordinated expression for a given clinical term. This is done by using embedding similarity 
        and analogies, as explained in 10.1016/j.jbi.2023.104297.
        
        Parameters:
            clinical_term (str):
                Name of the clinical term from which to generate the postcoordination.

        Returns:
            A dictionary with the keys: clinical_term, semantic_type, and relations, which contains the name, semantic type, and relations of the postcoordinated concept.
            The relations are stored in a list of dictionaries, whose keys are: relation_name, relation_id, target_concept_name, and target_concept_id.
        """
        # First, get the semantic type of the expression
        sem_type = self.identify_semantic_type(clinical_term)[0]

        # Then, get the reference concept
        reference_concept_id = self.identify_reference_concept(clinical_term, sem_type, n=1)[0]
        
        # We get the relationships of the reference concept
        reference_relations = self.snomed.get_related_concepts(reference_concept_id)

        relations = []
        for relation_id, _ in reference_relations:
            target_concept_id = self.identify_target_concept(clinical_term, reference_concept_id, relation_id)[0]

            relations.append({'relation_name' : self.snomed.get_fsn(relation_id),
                              'relation_id' : relation_id,
                              'target_concept_name' : self.snomed.get_fsn(target_concept_id),
                              'target_concept_id' : target_concept_id})

        return {'clinical_term' : clinical_term,
                'semantic_type' : sem_type,
                'relations' : relations}

    def identify_semantic_type(self, clinical_term : str, n : int = 5):
        """Method that finds the semantic types that might be assigned for a given clinical term. This is done by
        finding the five most similar concepts of SNOMED, taking their semantic types, and returning the most frequent ones among them. 
        
        Parameters:
            clinical_term (str):
                Name of the clinical term.
            n (int):
                Number of similar concepts to retrieve to decide the semantic type. The default value is 5.

        Returns:
            A list of semantic types ordered by frequency among the most similar concepts.
        """
        # Obtain the five most similar concepts
        sim_concepts = [concept_id for concept_id, _ in self.get_most_similar_concept(clinical_term, n)]

        # Obtain the semantic types for those concepts
        sem_types = [self.snomed.get_semantic_type(concept_id) for concept_id in sim_concepts]

        # Returns a list of possible semantic types
        return order_by_frequency(sem_types)

    def identify_reference_concept(self, clinical_term : str, semantic_type : str, n : int = 5):
        """Method that finds the possible reference concept for a given clinical term. This is done by looking at the
        most similar concepts that share a semantic type and that at least have one relationship different from IS_A.
        
        Parameters:
            clinical_term (str):
                Name of the clinical term.
            semantic_type (str):
                Name of the semantic type.
            n (int):
                Number of reference concepts to retrieve.
        Returns:
            A list of reference concepts ids.
        """
        # Obtain the embedding for the word
        clinical_term_embedding = self.embedding_model.get_embedding(clinical_term)
        
        # Normalize the vector
        q_vector = np.array([clinical_term_embedding]).astype(np.float32)
        faiss.normalize_L2(q_vector)

        # Perform the search
        _, index = self.faiss_index.search(q_vector, ARBITRARY_LARGE_NUMBER)

        reference_concepts = []

        for i in index[0]:
            # Obtain the concept id
            concept_id = self.embedding_sct_index[i]

            # Check if it shares the semantic type and if it contains a relationship other than IS_A
            if self.snomed.get_semantic_type(concept_id) == semantic_type and list_contains_different_rel(self.snomed.get_related_concepts(concept_id), IS_A_ID):
                reference_concepts.append(concept_id)

                # If we have enough reference concepts, we return it
                if len(reference_concepts) == n:
                    return reference_concepts
        
        # If we could not find enough reference concepts among the first ARBITRARY_LARGE_NUMBER of concepts, we return those
        # that comply with the requirements
        if len(reference_concepts) != 0:
            warnings.warn('Not enough reference concepts of that semantic type with a relationship other than is a could be found among 1000 most similar concepts. Those concepts found have been returned.')
            return reference_concepts

        # This should never happen if the semantic type has been selected by choosing the most frequent among the most similar concepts
        warnings.warn('No reference concept of semantic type could be found among 1000 most similar concepts. The most similar concept has been returned instead.')
        return [self.embedding_sct_index[index[0][0]]]

    def identify_target_concept(self, clinical_term : str, reference_concept_id : int, relation_id : int, n : int = 5, apply_filtering : bool = True):
        """Method to obtain the corresponding target concept for a certain relation and clinical term [clinical term, relation, target concept].
        This is done by using analogies and a reference concept, and then checking for the most similar concept among SNOMED CT to the analogy embedding.
        By default it tries to use filtering rules based on SNOMED CT logical model, although this requires for the rules files to be given when creating
        the SnomedEmbedder object.

        Parameters:
            clinical_term (str):
                Name of the clinical term.
            reference_concept_id (int):
                ID of the reference concept.
            relation_id (int):
                ID of the relationship.
            n (int):
                Number of target concepts to retrieve.
            apply_filtering (bool):
                Whether to apply filtering rules using SNOMED CT model. Defaults to true.
        
        Returns:
            A list of possible target concepts ids.   
        """
        # Obtain the embedding for the word
        clinical_term_embedding = self.embedding_model.get_embedding(clinical_term)
        
        # Obtain the embedding for the reference concept
        reference_embedding = self.embedding_values[self.sct2index[reference_concept_id]]

        # Obtain the embedding for the target concept given the reference and the relation
        related_concepts = self.snomed.get_related_concepts(reference_concept_id, filter_rels=[relation_id])

        if len(related_concepts) == 0:
            warnings.warn('No target concept found for relation', relation_id, 'and concept', reference_concept_id)
            return None
        
        target_concept_id = related_concepts[0][1]

        target_concept_embedding = self.embedding_values[self.sct2index[target_concept_id]]

        # Obtain the embedding of the analogy
        analogy_embedding = np.array(target_concept_embedding) - np.array(reference_embedding) + np.array(clinical_term_embedding)

        # Normalize the vector
        q_vector = np.array([analogy_embedding]).astype(np.float32)
        faiss.normalize_L2(q_vector)
        
        # Perform the search
        _, index = self.faiss_index.search(q_vector, n)

        concept_ids = [self.embedding_sct_index[i] for i in index[0]]

        # TODO: Falta añadir el filtrado por reglas

        return concept_ids
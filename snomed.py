import pandas as pd
from embedding_models.embedding_model import EmbeddingModel
from collections.abc import Iterable
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import re

# SCT-Codes
FULLY_SPECIFIED_NAME_ID = 900000000000003001
IS_A_ID = 116680003
ROOT_CONCEPT = 138875005
METADATA_ROOT = 900000000000441003
TOP_CONCEPTS = [123037004, 404684003, 308916002, 272379006, 363787002,
                410607006, 373873005, 78621006, 260787004, 71388002,
                362981000, 419891008, 243796009, 48176007, 370115009,
                123038009, 254291000, 105590001, 900000000000441003]

class Snomed:
    """Represents the SNOMED CT ontology-based terminology.

    Attributes:
        concepts (dict):
            A dictionary that maps each SCT-ID to a dictionary with the following keys: FSN, description, relations,
            relationsAux, definition, and semantic_type. Relations is a list of tuples (tail concept SCT-ID, relationship
            SCT-ID) that represents the relationships of the concept. RelationsAux is similar, but contains the inverse
            of the is-a relationships.

        metadata (dict):
            A dictionary that maps each SCT-ID from a metadata concept to a dictionary with the same keys as the
            concepts' attribute.

        model (EmbeddingModel):
            Embedding model used for generating embeddings of each concept. This can be None.

        embeddings (dict):
            A dictionary that maps each SCT-ID to an embedding. This can be None if there is no model.
    """
    def __init__(self, con_path: str, rel_path: str, desc_path: str, def_path: str = None, embedding_model : EmbeddingModel = None):
        """Loads SNOMED-CT from certain files.

        Parameters:
            con_path (str):
               Path to the concepts file from the international release.
            rel_path (str):
                Path to the relationship file from the international release.
            desc_path (str):
                Path to the descriptions file from a national or the international release.
            def_path (str):
                Path to the definitions file from a national or the international release. This is optional.
            embedding_model (EmbeddingModel):
                Represents an embedding model, and it should be a subclass of EmbeddingModel. This is optional.
        """

        concepts_pd = pd.read_csv(con_path, delimiter='\t')

        # We check the status of the concepts in the international release to avoid differences
        # with the possible national releases
        concept_status = dict()
        for CID, active in zip(concepts_pd['id'], concepts_pd['active']):
            concept_status[CID] = active

        descriptions_pd = pd.read_csv(desc_path, delimiter='\t')

        self.concepts = dict()
        for CID, active, typeID, description in zip(descriptions_pd['conceptId'],
                                                    descriptions_pd['active'],
                                                    descriptions_pd['typeId'],
                                                    descriptions_pd['term']):

            # If the concept is active
            if active == 1 and (CID in concept_status and concept_status[CID] == 1):
                if CID in self.concepts:
                    if typeID == FULLY_SPECIFIED_NAME_ID:
                        match = re.search('(.+)(\(.+\))', description)
                        self.concepts[CID]['FSN'] = match.group(1).strip() 
                        self.concepts[CID]['semantic_type'] = match.group(2).strip()[1:-1]
                        if match.group(1).strip() not in self.concepts[CID]['description']:
                            self.concepts[CID]['description'].append(match.group(1).strip())
                    elif description not in self.concepts[CID]['description'] and description == description:
                        self.concepts[CID]['description'].append(description)
                else:
                    if typeID == FULLY_SPECIFIED_NAME_ID:
                        match = re.search('(.+)(\(.+\))', description)
                        self.concepts[CID] = {'FSN': match.group(1).strip(),
                                              'description': [match.group(1).strip()],
                                              'relations': [],
                                              'relationsAux': [], 'definition': '',
                                              'semantic_type': match.group(2).strip()[1:-1]}
                    else:
                        # This is to prevent a nan value
                        if description == description:
                            self.concepts[CID] = {'FSN': '', 'description': [description], 'relations': [],
                                                'relationsAux': [], 'definition': '', 'semantic_type': ''}

        # The file of definitions is optional
        if def_path is not None:
            definition_pd = pd.read_csv(def_path, delimiter='\t')
            for active, CID, definition in zip(definition_pd['active'],
                                               definition_pd['conceptId'],
                                               definition_pd['term']):
                if active == 1 and CID in self.concepts:
                    self.concepts[CID]['definition'] = definition

        relations_pd = pd.read_csv(rel_path, delimiter='\t')
        for active, sourceID, destID, typeID in zip(relations_pd['active'],
                                                    relations_pd['sourceId'],
                                                    relations_pd['destinationId'],
                                                    relations_pd['typeId']):
            if active == 1 and sourceID in self.concepts and destID in self.concepts:
                self.concepts[sourceID]['relations'].append([destID, typeID])
                if typeID == IS_A_ID:
                    self.concepts[destID]['relationsAux'].append(sourceID)

        # This is to extract the metadata
        unexplored_metadata = [METADATA_ROOT]
        self.metadata = dict()

        while len(unexplored_metadata) > 0:
            sourceID = unexplored_metadata.pop(0)

            if sourceID not in self.metadata:
                for destID in self.concepts[sourceID]['relationsAux']:
                    unexplored_metadata.append(destID)

                self.metadata[sourceID] = self.concepts.pop(sourceID)
        
        # Loading the embedding model and creating embeddings
        if embedding_model is not None:
            self.model = embedding_model
            self.embeddings = dict()

            for conceptID, concept in self.concepts.items():
                vectors = []
                for name in concept['description']:
                    vectors.append(self.model.get_embedding(name))
                
                self.embeddings[conceptID] = sum(vectors)/len(vectors)

        else:
            self.model = None
            self.embeddings = None

    def get_fsn(self, sct_id : int):
        """Method that returns the full specified name (FSN) of a concept given its ID.
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
        Returns:
            A string that represents the FSN. It returns an empty string if the concept is not in SNOMED.
        """
        if sct_id in self.concepts:
            return self.concepts[sct_id]['FSN']
        elif sct_id in self.metadata:
            return self.metadata[sct_id]['FSN']
        warnings.warn("Concept", sct_id, "was not found in this version of SNOMED CT.")
        return ''
    
    def get_descriptions(self, sct_id : int):
        """Method that returns the descriptions of a concept given its ID.
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
        Returns:
            A list of string that contains the description of the concept. It returns an empty list
            if the concept is not in SNOMED.
        """
        if sct_id in self.concepts:
            return self.concepts[sct_id]['description']
        elif sct_id in self.metadata:
            return self.metadata[sct_id]['description']
        warnings.warn("Concept", sct_id, "was not found in this version of SNOMED CT.")
        return []

    def get_semantic_type(self, sct_id : int):
        """Method that returns the semantic type, that is what appears between parenthesis in the concept's FSN.
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
        Returns:
            A string that represents the semantic type. It returns an empty string if the concept is not in SNOMED.
        """
        if sct_id in self.concepts:
            return self.concepts[sct_id]['semantic_type']
        elif sct_id in self.metadata:
            return self.metadata[sct_id]['semantic_type']
        warnings.warn("Concept", sct_id, "was not found in this version of SNOMED CT.")
        return ''
    
    def get_related_concepts(self, sct_id : int, filter_rels : list[int] = None):
        """Method that returns which SNOMED CT concepts are related to the concept given as a parameter. Concepts
        that are the object part of a relationship with the concept parameter are considered related concepts. If
        filter_rels is set to a list of integers containing valid relationships, only concepts that are part
        of those types of relationships will be returned.
        
        Parameters:
            sct_id (int):
                Integer that represents an ID of SNOMED CT.
            filter_rels (list):
                List that contains which relationships from SNOMED CT we are interested in.        
        
        Returns:
            A list containing tuples (relationship_ID, sct_ID) for each related concept. If there are no related concepts,
            an empty list is returned instead.
        """
        related_concepts = []
        if sct_id in self.concepts:
            related_concepts = [(rel_id, object_id) for object_id, rel_id in self.concepts[sct_id]['relations'] 
                                                    if filter_rels is None or rel_id in filter_rels]
        elif sct_id in self.metadata:
            related_concepts = [(rel_id, object_id) for object_id, rel_id in self.metadata[sct_id]['relations'] 
                                                    if filter_rels is None or rel_id in filter_rels]
        else:
            warnings.warn("Concept", sct_id, "was not found in this version of SNOMED CT.")

        return related_concepts

    def get_top_level_concept(self, sct_id : int):
        """Method that returns to which of the 19 top level hierarchies the concept belongs to.
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
        Returns:
            The SCT-ID of the top level concept.
        """
        if sct_id in TOP_CONCEPTS:
            return sct_id
        
        if sct_id == ROOT_CONCEPT or sct_id == METADATA_ROOT:
            return sct_id

        if sct_id in self.concepts:
            parent_id = [destID for destID, typeID in self.concepts[sct_id]['relations'] if typeID == IS_A_ID][0]
        elif sct_id in self.metadata:
            parent_id = [destID for destID, typeID in self.metadata[sct_id]['relations'] if typeID == IS_A_ID][0]
        else:
            warnings.warn("Concept", sct_id, "was not found in this version of SNOMED CT.")
            return None
        return self.get_top_level_concept(parent_id)

    def get_top_concept_list(self, sct_id : int, top_list : Iterable):
        """Method that returns to which of the concepts in the top_list the sct_id is related to in the is-a hierarchy.
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
            top_list (iterable):
                Collection of SCT-IDs.
        Returns:
            A list with the SCT-ID of the concepts from the top_list to which the concept is related to. 
            If no concept in top_list is related with sct_id, an empty list is returned instead.
        """
        if sct_id == ROOT_CONCEPT or sct_id == METADATA_ROOT:
            return []

        if sct_id in top_list:
            elements_from_top_list = [sct_id]
        else:
            elements_from_top_list = []
        
        if sct_id in self.concepts:
            parent_ids = [destID for destID, typeID in self.concepts[sct_id]['relations'] if typeID == IS_A_ID]
        elif sct_id in self.metadata:
            parent_ids = [destID for destID, typeID in self.metadata[sct_id]['relations'] if typeID == IS_A_ID]
        else:
            warnings.warn("Concept", sct_id, "was not found in this version of SNOMED CT.")
            return None
            
        for parent_id in parent_ids:
            elements_from_top_list += self.get_top_concept_list(parent_id, top_list)
            
        return list(set(elements_from_top_list))

    def get_depth(self, sct_id : int):
        """Method that returns how deep a concept is in the is_a hierarchy, being the the root concept 
        138875005 | SNOMED CT Concept (SNOMED RT+CTV3) of depth 1.
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
        Returns:
            An integer representing the depth of the concept. If the concept is not present in SNOMED, it will
            return -1.
        """
        if sct_id == ROOT_CONCEPT:
            return 1
        
        if sct_id in self.concepts:
            parent_id = [destID for destID, typeID in self.concepts[sct_id]['relations'] if typeID == IS_A_ID][0]
        elif sct_id in self.metadata:
            parent_id = [destID for destID, typeID in self.metadata[sct_id]['relations'] if typeID == IS_A_ID][0]
        else:
            return -1
        return self.get_depth(parent_id) + 1

    def get_sct_concepts(self, concepts : bool = True, metadata : bool = True):
        """Method that returns the concepts in SNOMED CT. If concepts is set to true, non-metadata concepts
        will be returned. If metadata is set to true, metadata concepts will be returned as well.
        
        Paramters:
            concepts (bool):
                Whether to return non-metadata concepts or not.
            metadata (bool):
                Whether to return metadata concepts or not.
        
        Returns:
            A list of integers representing SCT_IDs. If both concepts and metadata are set to false,
            an empty list is returned instead.
        """
        concepts_list = []

        if concepts:
            concepts_list += list(self.concepts.keys())
        
        if metadata:
            concepts_list += list(self.metadata.keys())

        return concepts_list
    
    def get_children_of(self, sct_id : int):
        """Method that returns the concepts in SNOMED CT that are children of sct_id. This method travels through the 
        whole hiearchy, so not only direct children are returned.
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
        
        Returns:
            A list of integers representing SCT_IDs.
        """
        if sct_id in self.concepts:
            children_ids = self.concepts[sct_id]['relationsAux']
        elif sct_id in self.metadata:
            children_ids = self.metadata[sct_id]['relationsAux']
        else:
            warnings.warn("Concept", sct_id, "was not found in this version of SNOMED CT.")
            return []
        
        children = [sct_id]
        for children_id in children_ids:
            children += self.get_children_of(children_id)

        return list(set(children))
    
    def is_leaf_concept(self, sct_id : int):
        """Method that returns if a concept in SNOMED CT is a leaf concept or not. A leaf concept
        is one which has no child with an Is-a relationship (116680003).
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
        
        Returns:
            True if the concept is a leaf concept. Otherwise, False is returned.
        """
        if sct_id in self.concepts:
            return len(self.concepts[sct_id]['relationsAux']) == 0
        
        if sct_id in self.metadata:
            return len(self.metadata[sct_id]['relationsAux']) == 0
        
        warnings.warn("Concept", sct_id, "was not found in this version of SNOMED CT.")
        return False

    def is_child_of(self, sct_id_A : int, sct_id_B : int):
        """Method that returns if concept denoted by sct_id_A is a child of the concept denoted by sct_id_B. This relation does not have to be direct, as the
        method only checks if sct_id_B is part of any of the concepts in the is_a hierarchy of sct_id_A created by going towards the root concept.
        
        Parameters:
            sct_id_A (int):
                ID of the SNOMED CT concept that might be the child.
            sct_id_B (int):
                ID of the SNOMED CT concept that might be the parent.
        
        Returns:
            True if sct_id_A is child of sct_id_B. False otherwise.
        """
        if sct_id_A in self.concepts:
            parent_ids = [destID for destID, typeID in self.concepts[sct_id_A]['relations'] if typeID == IS_A_ID]
        elif sct_id_A in self.metadata:
            parent_ids = [destID for destID, typeID in self.metadata[sct_id_A]['relations'] if typeID == IS_A_ID]
        else:
            return False
        
        if sct_id_B in parent_ids:
            return True
        
        for parent_id in parent_ids:
            if self.is_child_of(parent_id, sct_id_B):
                return True
        
        return False

    # Methods that need an embedding model to work
    def get_most_similar_concept(self, word : str, n : int = 1):
        '''Method that returns the most similar concept to the string that receives as a parameter. This is
        done by performing cosine similarity between embeddings.
        
        Paramters:
            word (str):
                String of text from which to obtain the most similar concept.
            n (int):
                Number of similar concepts to retrieve. The default value is 1.
        Returns:
            A tuple (SCT-ID, sim_value), where the first element is the ID of the SNOMED concept, and the second
            is the similarity score. If n is greater than 1, it returns a list of tuples with the n most similar 
            concepts instead. If no embedding model was assigned, an empty list is returned.
        '''
        if self.model is None:
            warnings.warn('No embedding model was assigned to Snomed.')
            return []

        vector = self.model.get_embedding(word)

        dictionary_embeddings = list(self.embeddings.values())
        dictionary_conceptids = list(self.embeddings.keys())

        # Obtain the similarities between this vector and the embeddings
        similarities = cosine_similarity(X=vector.reshape(1, -1), Y=dictionary_embeddings)[0]

        # We join together the concepts_ids with the similarities and order it to get the most
        # similar concept
        sim_list = list(zip(dictionary_conceptids, similarities))
        sim_list.sort(key=lambda x : x[1], reverse=True)

        # Return a tuple (conceptID, similarity)
        if n <= 1:
            return sim_list[0]
        else:
            return sim_list[:n]
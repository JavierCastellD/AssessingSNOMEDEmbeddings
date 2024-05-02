import os
import warnings

import numpy as np
import torch
from torch_geometric.data import HeteroData, Dataset

from python_libraries.embedding_models.embedding_model import EmbeddingModel
from python_libraries.snomed import Snomed

class SnomedData(Dataset):
    """Class that representents a Heterogeneous Graph derived from SNOMED CT so that it can be used for GNN tasks.
    Semantic tags or top level concepts are used as classes, whereas each concept is considered an entity/individual.
    Initial feature vectors are defined by using an EmbeddingModel on the different descriptions/names of each concept.

    Attributes:
        snomed (Snomed):
            Snomed object that represents the information of the ontology/terminology SNOMED CT.
        
        embedding_model (EmbeddingModel):
            EmbeddingModel object that represents the embedding model used to create the initial feature vectors.
            
        data (HeteroData):
            HeteroData that contains the information of the knowledge graph.

    """
    def __init__(self, root : str, snomed : Snomed, embedding_model : EmbeddingModel = None,
                 class_as_semantic_tag : bool = False):
        """
        Parameters:
            root (str):
                Path to the folder where the dataset should be stored. This folder should be split into
                raw_dir (downloaded dataset) and processed_dir (processed dataset).
            snomed (Snomed):
                Snomed object that represents an instance of SNOMED CT.
            embedding_model (EmbeddingModel):
                EmbeddingModel used to create initial feature vectors for each concept in SNOMED CT.
            class_as_semantic_tag (bool):
                Whether to use semantic tags as classes for the concepts. If set to False, top level
                concepts will be considered as classes.
        """ 
        # Store the SNOMED CT information
        self.snomed = snomed
        
        # Store the embedding model information
        # TODO: If set to None, a different way of creating the initial feature vector should be used 
        self.embedding_model = embedding_model

        # Set if semantic tags or top level concepts will be used as "classes"
        self._class_as_semantic_tag = class_as_semantic_tag

        super(SnomedData, self).__init__(root, transform=None, pre_transform=None)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return 'no_raw_file'

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        return 'snomed_data.pt'


    def download(self):
        pass
    
    def process(self):
        """Process the dataset into a Data object from Pytorch Geometric. This method
        transforms the SNOMED CT graph into a HeteroData object.
        """
        # Preprocess the information in SNOMED CT to create the Heterogeneous Graph
        self.data = self.__preprocess_snomed()

        # Save the data into the processed folder
        torch.save(self.data, self.processed_paths[0])


    def __preprocess_snomed(self):
        """Method that handles everything related to preprocessing SNOMED CT. It extracts information about each
        concept's 'class', uses each concept's names as feature vectors by creating an embedding, and stores the
        information about the edges after mapping each concept to an index.
        
        Returns:
            An HeteroData containing SNOMED CT.
        """
        # Extract the information about feature vectors and edges from SNOMED
        concepts_feature_vectors, concepts_relations_edges = self.__extract_info_snomed()

        # Create the HeteroData object
        data = HeteroData()

        # Add the feature vectors to the object
        for class_name, class_features in concepts_feature_vectors.items():
            data[class_name].x = torch.tensor(np.array(class_features), dtype=torch.float)
            data[class_name].node_id = torch.arange(len(class_features))

        # Add the edges
        for domain_class, rel_range_edge in concepts_relations_edges.items():
            for rel, range_edge in rel_range_edge.items():
                for range_class, edges in range_edge.items():
                    # Transform the edges into a Pytorch tensor and use
                    # transpose and contiguous to shape it as expected
                    mapped_edges = torch.tensor(edges, dtype=torch.long)
                    edge_index = mapped_edges.t().contiguous()

                    data[domain_class, rel, range_class].edge_index = edge_index

        return data

    def __extract_info_snomed(self):
        """Extracts the information from SNOMED CT and stores into two dictionaries: one containing
        the information about each concept's feature vector, and another one containing the edge indexes.
        
        Returns:
            Two dictionaries. The first one has classes as keys and contains the feature vectors to be store in the HeteroData.
            The second one contains the information about edges and is a three level dictionary: domain_class, relation, range_class,
            and the values are lists of tuples [subject_index, object_index].
        """
        # Dictionary to store the feature vector per class
        concepts_feature_vectors = {}

        # Dictionary to store the relation edges
        concepts_relations_edges = {}

        # Dictionary to map the concepts into index
        mappings = {}

        # We first iterate through the concepts to generate the mappings and feature vectors
        for concept_id in self.snomed.get_sct_concepts():
            # Obtain the class ID and FSN the concept id belongs to
            if self._class_as_semantic_tag:
                subject_hierarchy_fsn = self.snomed.get_semantic_tag(concept_id)
            else:
                subject_hierarchy_id = self.snomed.get_top_level_concept(concept_id) 
                subject_hierarchy_fsn = self.snomed.get_fsn(subject_hierarchy_id)

            # We generate the feature vector for the given concept
            subject_feature_vector = self.__get_feature_vector(concept_id)

            # We store the feature vector inside their class dictionary 
            if subject_hierarchy_fsn not in concepts_feature_vectors:
                concepts_feature_vectors[subject_hierarchy_fsn] = []
            concepts_feature_vectors[subject_hierarchy_fsn].append(subject_feature_vector)
                
            # As we iterate through the concepts, we create the mappings
            if subject_hierarchy_fsn not in mappings:
                mappings[subject_hierarchy_fsn] = {concept_id : 0}
            else:
                if concept_id not in mappings[subject_hierarchy_fsn]:
                    mappings[subject_hierarchy_fsn][concept_id] = len(mappings[subject_hierarchy_fsn])

        # Once we have the mappings and feature vectors created, we can
        # store the information about the edges. This is done separately
        # to ensure that the index of a concept's ID is the same as the position of
        # its corresponding feature vector
        for concept_id in self.snomed.get_sct_concepts():
            # Obtain the FSN of the subject concept's class
            if self._class_as_semantic_tag:
                subject_hierarchy_fsn = self.snomed.get_semantic_tag(concept_id)
            else:
                subject_hierarchy_id = self.snomed.get_top_level_concept(concept_id) 
                subject_hierarchy_fsn = self.snomed.get_fsn(subject_hierarchy_id)

            # Iterate through the concept's relationships
            for rel_id, object_id in self.snomed.get_related_concepts(concept_id):
                # Obtain the FSN of the object concept's class
                if self._class_as_semantic_tag:
                    object_hierarchy_fsn = self.snomed.get_semantic_tag(object_id)
                else:
                    object_hierarchy_id = self.snomed.get_top_level_concept(object_id) 
                    object_hierarchy_fsn = self.snomed.get_fsn(object_hierarchy_id)

                # Obtain the FSN of the relation
                rel_fsn = self.snomed.get_fsn(rel_id)

                # Obtain the index of the subject and object concept
                subject_index = mappings[subject_hierarchy_fsn][concept_id]
                object_index = mappings[object_hierarchy_fsn][object_id]

                # Add the relationship to the dictionary
                if subject_hierarchy_fsn in concepts_relations_edges:
                    if rel_fsn in concepts_relations_edges[subject_hierarchy_fsn]:
                        if object_hierarchy_fsn in concepts_relations_edges[subject_hierarchy_fsn][rel_fsn]:
                            concepts_relations_edges[subject_hierarchy_fsn][rel_fsn][object_hierarchy_fsn].append([subject_index, object_index])
                        else:
                            concepts_relations_edges[subject_hierarchy_fsn][rel_fsn][object_hierarchy_fsn] = [[subject_index, object_index]]
                    else:
                        concepts_relations_edges[subject_hierarchy_fsn][rel_fsn] = {object_hierarchy_fsn : [[subject_index, object_index]]}
                else:
                    concepts_relations_edges[subject_hierarchy_fsn] = {rel_fsn : { object_hierarchy_fsn : [[subject_index, object_index]]}}

        return concepts_feature_vectors, concepts_relations_edges


    def __get_feature_vector(self, sct_id : int):
        """Method to generate the initial feature vector for a given concept. If this class contains an EmbeddingModel,
        it uses it to create an embedding using the descriptions of the concept.

        Parameters:
            sct_id (int):
                Identifier of a concept in SNOMED CT.

        Returns:
            A list containing the initial feature vector.
        """
        if self.embedding_model is not None:
            names = self.snomed.get_descriptions(sct_id)
            vectors = []
            for name in names:
                vectors.append(self.embedding_model.get_embedding(name))
            
            if len(vectors) > 0:
                return sum(vectors)/len(vectors)
            warnings.warn("Returning empty feature vector for concept " + str(sct_id) + ".")
            return []
        else:
            warnings.warn("No embedding model was given for SnomedData. An empty feature vector will be generated. WIP. ")
            return []
        
    def len(self):
        return self.data.shape[0]
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt')) 

        return data
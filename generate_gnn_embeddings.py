import random

from python_libraries.snomed import Snomed, IS_A_ID
from snomed_gnn_data import preprocess_fsn, SnomedData

N_NEGATIVE_SAMPLES = 3
SIM_VALUE_IS_A = 0.75
SIM_VALUE_RELATED = 0.5
SIM_VALUE_NOT_RELATED = 0

def generate_train_tuples(snomed : Snomed, snomed_data : SnomedData):
    # List of tuples (top_level_concept_A, index_A, top_level_B, index_B)
    gnn_train_data = []

    # Label information
    label = []

    snomed_concepts = snomed.get_sct_concepts()

    # We iterate through each SNOMED CT concept
    for sct_id in snomed_concepts:
        # For each one we obtain the top level concept
        top_level_id_A = snomed.get_top_level_concept(sct_id)
        top_level_A = preprocess_fsn(snomed.get_fsn(top_level_id_A))

        # And the index position in the GNN
        index_A = snomed_data.mappings[top_level_A][sct_id]

        related_objs = []
        # Then, for each of that concept's related concepts
        for rel_id, obj_id in snomed.get_related_concepts(sct_id):
            # We obtain the top level concept
            top_level_id_B = snomed.get_top_level_concept(obj_id)
            top_level_B = preprocess_fsn(snomed.get_fsn(top_level_id_B))

            # Obtain the index position in the GNN
            index_B = snomed_data.mappings[top_level_B][obj_id]

            # And we store it, applying a different similarity value
            # depending on the relation
            gnn_train_data.append((top_level_A, index_A, top_level_B, index_B))

            # If they are part of a child-parent relation, we give a 0.75
            if rel_id == IS_A_ID:
                label.append(SIM_VALUE_IS_A)
            # Otherwise we give them a 0.5
            else:
                label.append(SIM_VALUE_RELATED)

            related_objs.append(obj_id)

        # We also create negative instances by searching for non directly related concepts
        for i in range(N_NEGATIVE_SAMPLES):
            random_sct_id = random.choice(snomed_concepts)
            while(random_sct_id in related_objs): random_sct_id = random.choice(snomed_concepts)

            # We obtain the top level concept
            top_level_id_B = snomed.get_top_level_concept(random_sct_id)
            top_level_B = preprocess_fsn(snomed.get_fsn(top_level_id_B))
            
            # Obtain the index position in the GNN
            index_B = snomed_data.mappings[top_level_B][random_sct_id]

            # And we store it, applying a 0 in similarity value
            gnn_train_data.append((top_level_A, index_A, top_level_B, index_B))
            label.append(SIM_VALUE_NOT_RELATED)

    return gnn_train_data, label

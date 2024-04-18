import random

from python_libraries.snomed import Snomed, IS_A_ID

def list_contains_other_id(id_list : list, id : int):
    """Method that returns whether a list contains an ID different from id.
    
    Parameters:
        id_list (list):
            List of integers, each one representing a ID.
    
        id (int):
            Integer that represents an ID.

    Returns:
        True if there is an ID different from id in the list. False otherwise.
    """
    if len(id_list) == 0:
        return False
    
    for it_id in id_list:
        if it_id != id:
            return True
    
    return False

# Load SNOMED CT
concept_path = "./snomed_data/conceptInternational_20221031.txt"
relationship_path = "./snomed_data/relationshipInternational_20221031.txt"
description_path = "./snomed_data/descriptionInternational_20221031.txt"

snomed = Snomed(concept_path, relationship_path, description_path)

# Define train/dev/test percentages splits
TRAIN_PROPORTION = 0.8
TEST_PROPORTION = 0.1

multiple_rel_concepts = []

# Obtain those concepts for which we have at least one relationship that is not IS_A_ID
for concept_id in snomed.get_sct_concepts(metadata=False):
    relations = [rel_id for rel_id, _ in snomed.get_related_concepts(concept_id)]
    
    if list_contains_other_id(relations, IS_A_ID):
        multiple_rel_concepts.append(concept_id)

n_train = round(len(multiple_rel_concepts) * TRAIN_PROPORTION)
n_test = round(len(multiple_rel_concepts) * TEST_PROPORTION)

# Shuffle the concepts
random.shuffle(multiple_rel_concepts)

# Obtain the train, dev, and test list of sct_ids
train_ids = multiple_rel_concepts[:n_train]
dev_ids = multiple_rel_concepts[n_train:-n_test]
test_ids = multiple_rel_concepts[-n_test:]

with open('train_concepts.txt', 'w') as f:
    for line in train_ids:
        f.write(f"{line}\n")

with open('dev_concepts.txt', 'w') as f:
    for line in dev_ids:
        f.write(f"{line}\n")

with open('test_concepts.txt', 'w') as f:
    for line in test_ids:
        f.write(f"{line}\n")
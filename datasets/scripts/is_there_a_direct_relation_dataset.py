# Load info to create datasets
from python_libraries.snomed import Snomed
import pandas as pd 
import random

IS_A_ID = 116680003

train_concepts = []
dev_concepts = []
test_concepts = []

with open('train_concepts.txt', 'r') as concepts_file:
    for line in concepts_file.readlines():
        train_concepts.append(int(line))

with open('dev_concepts.txt', 'r') as concepts_file:
    for line in concepts_file.readlines():
        dev_concepts.append(int(line))

with open('test_concepts.txt', 'r') as concepts_file:
    for line in concepts_file.readlines():
        test_concepts.append(int(line))

concept_path = "./snomed_data/conceptInternational_20240101.txt"
relationship_path = "./snomed_data/relationshipInternational_20240101.txt"
description_path = "./snomed_data/descriptionInternational_20240101.txt"

snomed = Snomed(concept_path, relationship_path, description_path)

# Create is there a direct relation dataset
concept_list = test_concepts
concepts_b_list = train_concepts + dev_concepts + test_concepts

subject_concepts = []
object_concepts = []
relation_status = []

subject_fsns = []
object_fsns = []

for concept in concept_list:
    subject_fsn = snomed.get_fsn(concept)
    relation_tuples = snomed.get_related_concepts(concept)

    concepts_direct_related = set([object_id for _, object_id in relation_tuples])

    for rel_id, object_id in relation_tuples:
        # We keep out of the dataset IS_A hierarchical relation
        if rel_id != IS_A_ID:
            # Add the positive example
            # Append IDs
            subject_concepts.append(concept)
            object_concepts.append(object_id)
            relation_status.append('YES')

            object_fsn = snomed.get_fsn(object_id)

            # Append FSNs
            subject_fsns.append(subject_fsn)
            object_fsns.append(object_fsn)

            # Add a negative example
            index = random.randint(0, len(concepts_b_list) - 1)
            
            while (concepts_b_list[index] in concepts_direct_related):
                index = random.randint(0, len(concepts_b_list) - 1)

            subject_concepts.append(concept)
            object_concepts.append(concepts_b_list[index])
            relation_status.append('NO')

            object_fsn = snomed.get_fsn(concepts_b_list[index])

            # Append FSNs
            subject_fsns.append(subject_fsn)
            object_fsns.append(object_fsn)



df = pd.DataFrame({'subject_id' : subject_concepts,
                   'object_id' : object_concepts,
                   'relation_status' : relation_status,
                   'subject_fsn' : subject_fsns,
                   'object_fsn' : object_fsns})

df.to_csv('dataset_is_there_relation_test.csv', index=False)
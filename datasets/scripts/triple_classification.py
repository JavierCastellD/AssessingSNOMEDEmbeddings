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

# Create triple classification dataset
concept_list = train_concepts
concepts_b_list = train_concepts

subject_concepts = []
relation_ids = []
object_concepts = []

triple_status = []

subject_fsns = []
relation_fsns = []
object_fsns = []

for concept in concept_list:
    subject_fsn = snomed.get_fsn(concept)
    relation_tuples = snomed.get_related_concepts(concept)

    for rel_id, object_id in relation_tuples:
        # We keep out of the dataset IS_A hierarchical relation
        if rel_id != IS_A_ID:
            concepts_related_by_rel = set([obj for rel, obj in relation_tuples if rel == rel_id])

            # Add the positive example
            # Append IDs
            subject_concepts.append(concept)
            relation_ids.append(rel_id)
            object_concepts.append(object_id)

            triple_status.append('YES')

            object_fsn = snomed.get_fsn(object_id)
            rel_fsn = snomed.get_fsn(rel_id)

            # Append FSNs
            subject_fsns.append(subject_fsn)
            relation_fsns.append(rel_fsn)
            object_fsns.append(object_fsn)

            # Add a negative example
            index = random.randint(0, len(concepts_b_list) - 1)
            
            while (concepts_b_list[index] in concepts_related_by_rel):
                index = random.randint(0, len(concepts_b_list) - 1)

            subject_concepts.append(concept)
            relation_ids.append(rel_id)
            object_concepts.append(concepts_b_list[index])

            triple_status.append('NO')

            object_fsn = snomed.get_fsn(concepts_b_list[index])

            # Append FSNs
            subject_fsns.append(subject_fsn)
            relation_fsns.append(rel_fsn)
            object_fsns.append(object_fsn)


df = pd.DataFrame({'subject_id' : subject_concepts,
                   'relation_id' : relation_ids,
                   'object_id' : object_concepts,
                   'triple_status' : triple_status,
                   'subject_fsn' : subject_fsns,
                   'relation_fsn' : relation_fsns,
                   'object_fsn' : object_fsns})

df.to_csv('dataset_triple_classification_train.csv', index=False)
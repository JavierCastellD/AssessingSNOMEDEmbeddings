# Load info to create datasets
from snomed import Snomed
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

# Create is son of direct relation dataset
concept_list = test_concepts
concepts_b_list = train_concepts + dev_concepts + test_concepts

subject_concepts = []
object_concepts = []
relation_status = []

subject_fsns = []
object_fsns = []

# For negative samples we ensure that they comply with the SNOMED logical model
concepts_per_sem_type = {}

for concept in concepts_b_list:
    sem_type = snomed.get_semantic_type(concept)
    if sem_type != '': # To prevent errors with concepts that do not have FSN according in the SCT files
        if sem_type in concepts_per_sem_type:
            concepts_per_sem_type[sem_type].append(concept)
        else:
            concepts_per_sem_type[sem_type] = [concept]

for concept in concept_list:
    subject_fsn = snomed.get_fsn(concept)
    relation_tuples = snomed.get_related_concepts(concept)

    parent_concepts = set([object_id for rel_id, object_id in relation_tuples if rel_id == IS_A_ID])

    for object_id in parent_concepts:
        obj_sem_type = snomed.get_semantic_type(object_id)

        # To prevent errors with concepts that do not have FSN according in the SCT files or with concepts such as "special concept" or similar ones with special semantic types
        if obj_sem_type in concepts_per_sem_type and len(concepts_per_sem_type[obj_sem_type]) > 1:
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
            index = random.randint(0, len(concepts_per_sem_type[obj_sem_type]) - 1)
            
            while (concepts_per_sem_type[obj_sem_type][index] in parent_concepts):
                index = random.randint(0, len(concepts_per_sem_type[obj_sem_type]) - 1)

            subject_concepts.append(concept)
            object_concepts.append(concepts_per_sem_type[obj_sem_type][index])
            relation_status.append('NO')

            object_fsn = snomed.get_fsn(concepts_per_sem_type[obj_sem_type][index])

            # Append FSNs
            subject_fsns.append(subject_fsn)
            object_fsns.append(object_fsn)



df = pd.DataFrame({'subject_id' : subject_concepts,
                   'object_id' : object_concepts,
                   'is_son_of' : relation_status,
                   'subject_fsn' : subject_fsns,
                   'object_fsn' : object_fsns})

df.to_csv('dataset_is_son_of_v2_test.csv', index=False)
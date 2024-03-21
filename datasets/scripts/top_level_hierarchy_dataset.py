# Load info to create datasets
from snomed import Snomed
import pandas as pd 

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

# Create top level hierarchy classification dataset
concept_list = test_concepts

# To ensure we are not dealing with special concepts
concepts_per_sem_type = {}

for concept in concept_list:
    sem_type = snomed.get_semantic_type(concept)
    if sem_type != '': # To prevent errors with concepts that do not have FSN according to the SCT files
        if sem_type in concepts_per_sem_type:
            concepts_per_sem_type[sem_type].append(concept)
        else:
            concepts_per_sem_type[sem_type] = [concept]

top_levels = []
fsns = []
aux_concept_list = []

for concept in concept_list:
    sem_type = snomed.get_semantic_type(concept)

    if sem_type != '' and len(concepts_per_sem_type[sem_type]) > 1:
        top_level = snomed.get_top_level_concept(concept)
        top_levels.append(snomed.get_fsn(top_level))

        fsn = snomed.get_fsn(concept)
        fsns.append(fsn)

        aux_concept_list.append(concept)

df = pd.DataFrame({'concept_id' : aux_concept_list,
                   'fsn' : fsns,
                   'top_level' : top_levels})

df.to_csv('dataset_top_level_hierarchy_test.csv', index=False)
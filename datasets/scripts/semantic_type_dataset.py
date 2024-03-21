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

# Create semantic type dataset
concept_list = dev_concepts

# To ensure we are not dealing with special concepts
concepts_per_sem_type = {}

for concept in concept_list:
    sem_type = snomed.get_semantic_type(concept)
    if sem_type != '': # To prevent errors with concepts that do not have FSN according to the SCT files
        if sem_type in concepts_per_sem_type:
            concepts_per_sem_type[sem_type].append(concept)
        else:
            concepts_per_sem_type[sem_type] = [concept]


concept_ids = []
semantic_types = []
fsns = []

for concept in concept_list:
    sem_type = snomed.get_semantic_type(concept)

    # This is to prevent certain concepts, such as 'special concepts', from appearing
    if sem_type != '' and len(concepts_per_sem_type[sem_type]) > 1:
        semantic_types.append(sem_type)
        concept_ids.append(concept)
        fsn = snomed.get_fsn(concept)
        fsns.append(fsn)

df = pd.DataFrame({'concept_id' : concept_ids,
                   'fsn' : fsns,
                   'semantic_type' : semantic_types})

df.to_csv('dataset_semantic_type_dev.csv', index=False)
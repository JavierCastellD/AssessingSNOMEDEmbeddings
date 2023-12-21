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

snomed = Snomed('snomed_data/conceptInternational_20221031.txt', 'snomed_data/relationshipInternational_20221031.txt', 'snomed_data/descriptionInternational_20221031.txt')

# Create semantic type dataset
concept_list = dev_concepts

concept_ids = []
semantic_types = []
fsns = []

for concept in concept_list:
    sem_type = snomed.get_semantic_tag(concept)
    if sem_type != '':
        semantic_types.append(sem_type)
        concept_ids.append(concept)
        fsn = snomed.get_fsn(concept)
        fsns.append(fsn)

df = pd.DataFrame({'concept_id' : concept_ids,
                   'fsn' : fsns,
                   'semantic_type' : semantic_types})

df.to_csv('dataset_semantic_type_dev.csv', index=False)
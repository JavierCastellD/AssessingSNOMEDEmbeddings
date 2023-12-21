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

# Create top level hierarchy classification dataset
concept_list = test_concepts

top_levels = []
fsns = []

for concept in concept_list:
    top_level = snomed.get_top_level_concept(concept)
    top_levels.append(snomed.get_fsn(top_level))

    fsn = snomed.get_fsn(concept)
    fsns.append(fsn)

df = pd.DataFrame({'concept_id' : concept_list,
                   'fsn' : fsns,
                   'top_level' : top_levels})

df.to_csv('dataset_top_level_hierarchy_test.csv', index=False)
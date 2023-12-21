# Load info to create datasets
from snomed import Snomed
import pandas as pd 

IS_A_ID = 116680003
RELATION_THRESHOLD = 0.2

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

# Create relation prediction dataset
concept_list = dev_concepts

relation_types = {}
total = 0
for concept in train_concepts + test_concepts + dev_concepts:
    for rel_id, object_id in snomed.get_related_concepts(concept):
        if rel_id != IS_A_ID:
            # Count the number of relations, as we will filter out 
            # those with a frequency below a threshold
            total += 1
            if rel_id in relation_types:
                relation_types[rel_id] += 1
            else:
                relation_types[rel_id] = 1

# We will only consider those relationships with a frequency over 5%
rels_to_keep = []
rels_to_remove = []
for rel_id, count in relation_types.items():
    if round(count/total*100, 2) >= RELATION_THRESHOLD:
        rels_to_keep.append(rel_id)
    else:
        rels_to_remove.append(rel_id)


subject_concepts = []
object_concepts = []
relations = []

subject_fsns = []
object_fsns = []
relation_fsns = []

for concept in concept_list:
    subject_fsn = snomed.get_fsn(concept)
    for rel_id, object_id in snomed.get_related_concepts(concept):
        # We keep out of the dataset IS_A hierarchical relation
        # and only consider some relationships according to its frequency
        if rel_id != IS_A_ID and rel_id in rels_to_keep:
            # Append IDs
            subject_concepts.append(concept)
            object_concepts.append(object_id)
            relations.append(rel_id)

            object_fsn = snomed.get_fsn(object_id)
            rel_fsn = snomed.get_fsn(rel_id)

            # Append FSNs
            subject_fsns.append(subject_fsn)
            object_fsns.append(object_fsn)
            relation_fsns.append(rel_fsn)


df = pd.DataFrame({'subject_id' : subject_concepts,
                   'object_id' : object_concepts,
                   'relation_id' : relations,
                   'subject_fsn' : subject_fsns,
                   'object_fsn' : object_fsns,
                   'relation_fsn' : relation_fsns})

df.to_csv('dataset_relation_prediction_dev.csv', index=False)
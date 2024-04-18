# Script to train an embedding model in walk corpus
from python_libraries.snomed import Snomed
from python_libraries.snomed_walks import create_walk_corpus

# Load SNOMED CT
concept_path = "./snomed_data/conceptInternational_20221031.txt"
relationship_path = "./snomed_data/relationshipInternational_20221031.txt"
description_path = "./snomed_data/descriptionInternational_20221031.txt"

snomed = Snomed(concept_path, relationship_path, description_path)

concept_ids = []
with open('train_concepts.txt', 'r') as file:
    for line in file.readlines():
        concept_ids.append(int(line))

with open('dev_concepts.txt', 'r') as file:
    for line in file.readlines():
        concept_ids.append(int(line))

create_walk_corpus(snomed, 'corpus_train_dev_20221031.txt', concept_ids, uri_depth = 2, word_depth = 2)
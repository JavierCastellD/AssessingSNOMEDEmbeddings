import random

from python_libraries.snomed import Snomed

# Load SNOMED CT
CONCEPTS_PATH = "snomed_data/conceptInternational_20240101.txt"
RELATIONS_PATH = "snomed_data/relationshipInternational_20240101.txt"
DESCRIPTIONS_PATH = "snomed_data/descriptionInternational_20240101.txt"
snomed = Snomed(CONCEPTS_PATH, RELATIONS_PATH, DESCRIPTIONS_PATH)

# Define train/dev/test percentages splits
TRAIN_PROPORTION = 0.8
TEST_PROPORTION = 0.1

# Obtain all the non-metadata concepts
concepts = snomed.get_sct_concepts(metadata=False)

n_train = round(len(concepts) * TRAIN_PROPORTION)
n_test = round(len(concepts) * TEST_PROPORTION)

# Shuffle the concepts
random.shuffle(concepts)

# Obtain the train, dev, and test list of sct_ids
train_ids = concepts[:n_train]
dev_ids = concepts[n_train:-n_test]
test_ids = concepts[-n_test:]

with open('train_concepts.txt', 'w') as f:
    for line in train_ids:
        f.write(f"{line}\n")

with open('dev_concepts.txt', 'w') as f:
    for line in dev_ids:
        f.write(f"{line}\n")

with open('test_concepts.txt', 'w') as f:
    for line in test_ids:
        f.write(f"{line}\n")
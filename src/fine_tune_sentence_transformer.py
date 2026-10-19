from itertools import combinations
import random

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from python_libraries.snomed import IS_A_ID, Snomed

# Fine-tuning of a pre-trained sentence transformer on the task of concept similarity. Each sentence
# represents a SNOMED CT concept, and the pairs of sentences are given a similarity value depending on
# the relationship between both concepts:
# 1. Synonyms, i.e., two descriptions of the same concept, are given a 1
# 2. Parent-child pairs are given a 0.75
# 3. Concepts linked by any other relationship are given a 0.5
# 4. Unrelated concepts, chosen randomly, are given a 0
#
# Only the training concepts are used as the first concept of each pair, so that the relationships of the
# test concepts are not part of the supervision.
#
# Usage:
#   python src/fine_tune_sentence_transformer.py

N_NEGATIVE_SAMPLES = 3
N_EPOCH = 1
BATCH_SIZE = 32

BASE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
TRAIN_CONCEPTS_PATH = 'datasets/train_concepts.txt'
MODEL_PATH = 'models/full_st_minilm_' + str(N_NEGATIVE_SAMPLES) + '_neg_' + str(N_EPOCH) + '_ep_sct_train.model'
CHECKPOINT_PATH = 'models/st_minilm_' + str(N_NEGATIVE_SAMPLES) + '_neg_' + str(N_EPOCH) + '_ep_sct'

# Load SNOMED CT
snomed = Snomed('snomed_data/conceptInternational_20240101.txt', 'snomed_data/relationshipInternational_20240101.txt', 'snomed_data/descriptionInternational_20240101.txt')

# Load the base model
sentence_transformer = SentenceTransformer(BASE_MODEL)

# Obtain the concepts in scope
concepts_in_scope = []
print('Reading file...')
with open(TRAIN_CONCEPTS_PATH, 'r') as file:
    for line in file.readlines():
        concepts_in_scope.append(int(line))

# Create the train dataset
# A dataset for a sentence-transformer is made up of pairs of sentences and a value between 0 and 1
# representing how similar are both sentences 
kb_sft_examples = []

for sct_id in concepts_in_scope:
    descriptions = snomed.get_descriptions(sct_id)
    fsn = snomed.get_fsn(sct_id)
    
    # Synonyms are given a 1.0 in similarity, as they represent the same concept
    for syn1, syn2 in combinations(descriptions, 2):
        kb_sft_examples.append(InputExample(texts=[syn1, syn2], label=1.0))

    related_objs = []
    for rel_id, obj_id in snomed.get_related_concepts(sct_id):
        fsn_obj = snomed.get_fsn(obj_id)
        
        # Related concepts are given a 0.75 if they are part of a child-parent relationship
        if rel_id == IS_A_ID:
            kb_sft_examples.append(InputExample(texts=[fsn, fsn_obj], label=0.75))
        # Or a 0.5 otherwise
        else:
            kb_sft_examples.append(InputExample(texts=[fsn, fsn_obj], label=0.5))

        related_objs.append(obj_id)

    # For each concept we also select a number of random concepts that are unrelated
    # and are given a 0 in similarity
    for i in range(N_NEGATIVE_SAMPLES):
        random_sct_id = random.choice(concepts_in_scope)
        while(random_sct_id in related_objs): random_sct_id = random.choice(concepts_in_scope)

        random_sct_id_fsn = snomed.get_fsn(random_sct_id)
        kb_sft_examples.append(InputExample(texts=[fsn, random_sct_id_fsn], label=0.0))


# Create the dataloader
kb_sft_dataloader = DataLoader(kb_sft_examples, shuffle=True, batch_size=BATCH_SIZE)

# Define the loss
kb_sft_loss = losses.CosineSimilarityLoss(sentence_transformer)

# Train the model
sentence_transformer.fit(train_objectives=[(kb_sft_dataloader, kb_sft_loss)],
                         epochs=N_EPOCH,
                         checkpoint_path=CHECKPOINT_PATH)
                         
# Save the model
sentence_transformer.save(MODEL_PATH)

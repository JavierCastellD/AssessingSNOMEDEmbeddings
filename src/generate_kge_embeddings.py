import sys

import numpy as np
import torch
from torch.optim import Adam
from pykeen.models import TransE, DistMult
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory

from python_libraries.snomed import Snomed
from python_libraries.embedding_models.embedding_model import save_embeddings

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else 'TransE'
EMBEDDING_DIM = 200
NUM_EPOCHS = 100
BATCH_SIZE = 4096
LEARNING_RATE = 0.001
SEED = 1234

CONCEPTS_PATH = 'snomed_data/conceptInternational_20240101.txt'
RELATIONS_PATH = 'snomed_data/relationshipInternational_20240101.txt'
DESCRIPTIONS_PATH = 'snomed_data/descriptionInternational_20240101.txt'
TEST_CONCEPTS_PATH = 'datasets/test_concepts.txt'
OUTPUT_PREFIX = f'kge_{MODEL_NAME.lower()}_{EMBEDDING_DIM}_no_test_edges'
CONCEPTS_OUTPUT_PATH = OUTPUT_PREFIX + '_sct_dict.npz'
RELATIONS_OUTPUT_PATH = OUTPUT_PREFIX + '_sct_metadata_dict.npz'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load SNOMED CT and the test concepts
snomed = Snomed(CONCEPTS_PATH, RELATIONS_PATH, DESCRIPTIONS_PATH)
with open(TEST_CONCEPTS_PATH, 'r') as file:
    test_concepts = set(int(line) for line in file.readlines())

# Build the triples, excluding those whose subject is a test concept
concepts = list(snomed.get_sct_concepts())
triples, relation_ids = [], set()
excluded = 0
for subject in concepts:
    for rel_id, obj in snomed.get_related_concepts(subject):
        if subject in test_concepts:
            excluded += 1
            continue
        triples.append((str(subject), str(rel_id), str(obj)))
        relation_ids.add(rel_id)

print(f'Triples used: {len(triples)} | excluded (test subjects): {excluded} | relation types: {len(relation_ids)}')

# Every concept is an entity, even if it does not appear in any remaining triple
entities = set(str(c) for c in concepts) | {o for _, _, o in triples}
entity_to_id = {e: i for i, e in enumerate(sorted(entities))}
relation_to_id = {str(r): i for i, r in enumerate(sorted(relation_ids))}

triples_factory = TriplesFactory.from_labeled_triples(
    triples=np.array(triples, dtype=str),
    entity_to_id=entity_to_id,
    relation_to_id=relation_to_id,
)

# Concepts without any training triple keep an untrained embedding
in_triples = {s for s, _, _ in triples} | {o for _, _, o in triples}
test_without_triples = sum(str(c) not in in_triples for c in test_concepts)
print(f'Test concepts without any training triple (untrained embedding): {test_without_triples} '
      f'({100 * test_without_triples / len(test_concepts):.1f}%)')

# Model: same training setup for both, so that they only differ in the scoring function
if MODEL_NAME == 'TransE':
    model = TransE(triples_factory=triples_factory, embedding_dim=EMBEDDING_DIM, scoring_fct_norm=1,
                   loss='MarginRankingLoss', random_seed=SEED)
elif MODEL_NAME == 'DistMult':
    model = DistMult(triples_factory=triples_factory, embedding_dim=EMBEDDING_DIM,
                     loss='MarginRankingLoss', random_seed=SEED)
else:
    raise ValueError(f'Unknown model: {MODEL_NAME}')
model = model.to(device)

optimizer = Adam(params=model.get_grad_params(), lr=LEARNING_RATE)
training_loop = SLCWATrainingLoop(model=model, triples_factory=triples_factory, optimizer=optimizer)
training_loop.train(triples_factory=triples_factory, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

# Export concept and relation embeddings
model.eval()
with torch.no_grad():
    entity_embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()
    relation_embeddings = model.relation_representations[0](indices=None).detach().cpu().numpy()

concept_dictionary = {entity: entity_embeddings[idx] for entity, idx in entity_to_id.items()}
relation_dictionary = {relation: relation_embeddings[idx] for relation, idx in relation_to_id.items()}

save_embeddings(embedding_dictionary=concept_dictionary, filename=CONCEPTS_OUTPUT_PATH)
save_embeddings(embedding_dictionary=relation_dictionary, filename=RELATIONS_OUTPUT_PATH)
print(f'Saved {len(concept_dictionary)} concept embeddings to {CONCEPTS_OUTPUT_PATH}')
print(f'Saved {len(relation_dictionary)} relation embeddings to {RELATIONS_OUTPUT_PATH}')

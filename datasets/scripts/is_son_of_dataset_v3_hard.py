import random
import sys
sys.path.append('../..') 

import pandas as pd

from src.python_libraries.snomed import Snomed, IS_A_ID

SEED = 1234
NEGATIVE_TYPES = ['sibling', 'grandparent', 'reversed', 'cousin', 'random']
MAX_ATTEMPTS = 20
RESTRICT_RANGE = False

random.seed(SEED)

def read_ids(path):
    with open(path, 'r') as f:
        return [int(line) for line in f.readlines()]

train_concepts = read_ids('train_concepts.txt')
dev_concepts = read_ids('dev_concepts.txt')
test_concepts = read_ids('test_concepts.txt')

snomed = Snomed("../../snomed_data/conceptInternational_20240101.txt",
                "../../snomed_data/relationshipInternational_20240101.txt",
                "../../snomed_data/descriptionInternational_20240101.txt")

# Same range rule as the previous versions: the concepts that can appear as the second
# concept of a pair depend on the split
if RESTRICT_RANGE:
    SPLITS = {
        'train': (train_concepts, set(train_concepts)),
        'dev':   (dev_concepts, set(train_concepts + dev_concepts)),
        'test':  (test_concepts, set(train_concepts + dev_concepts + test_concepts)),
    }
else:
    all_concepts = set(snomed.get_sct_concepts(metadata=False))
    SPLITS = {split: (subjects, all_concepts) for split, subjects in
              [('train', train_concepts), ('dev', dev_concepts), ('test', test_concepts)]}

def parents(c):
    return {obj for rel, obj in snomed.get_related_concepts(c) if rel == IS_A_ID}

def children(c):
    # 'relationsAux' stores the direct children (inverse is-a) of each non-metadata concept
    return set(snomed.concepts[c]['relationsAux']) if c in snomed.concepts else set()

def valid(candidate, subject, subject_parents, allowed):
    return (candidate != subject and candidate not in subject_parents and candidate in allowed
            and snomed.get_semantic_type(candidate) != '')

def pick(candidates, subject, subject_parents, allowed):
    options = [c for c in candidates if valid(c, subject, subject_parents, allowed)]
    return random.choice(options) if options else None

def sample_negative(neg_type, subject, parent, subject_parents, allowed, concepts_per_sem_type):
    if neg_type == 'sibling':
        return pick(children(parent), subject, subject_parents, allowed)
    if neg_type == 'grandparent':
        return pick(parents(parent), subject, subject_parents, allowed)
    if neg_type == 'reversed':
        return pick(children(subject), subject, subject_parents, allowed)
    if neg_type == 'cousin':
        grandparents = list(parents(parent))
        for _ in range(MAX_ATTEMPTS if grandparents else 0):
            aunts = list(children(random.choice(grandparents)) - {parent})
            if aunts:
                # Exclude the parent's own children, so that a concept that is also a sibling
                # (multiple inheritance) is not labelled as a cousin
                candidate = pick(children(random.choice(aunts)) - children(parent), subject, subject_parents, allowed)
                if candidate is not None:
                    return candidate
        return None
    if neg_type == 'random':
        pool = concepts_per_sem_type.get(snomed.get_semantic_type(parent), [])
        for _ in range(MAX_ATTEMPTS * 50 if len(pool) > 1 else 0):
            candidate = random.choice(pool)
            if valid(candidate, subject, subject_parents, allowed):
                return candidate
        return None
    raise ValueError(neg_type)

for split, (subjects, allowed) in SPLITS.items():
    # Pool for the 'random' type: same as v2 (concepts of each semantic type within the allowed range)
    concepts_per_sem_type = {}
    for c in allowed:
        sem_type = snomed.get_semantic_type(c)
        if sem_type != '':
            concepts_per_sem_type.setdefault(sem_type, []).append(c)

    rows = []
    type_index = 0
    for subject in subjects:
        subject_parents = parents(subject)
        for parent in subject_parents:
            parent_sem_type = snomed.get_semantic_type(parent)
            # Same filter as v2 for the positive instances
            if parent_sem_type not in concepts_per_sem_type or len(concepts_per_sem_type[parent_sem_type]) <= 1:
                continue

            # Cycle through the types; fall back to the next one if no candidate exists
            # ('random' is a target type in its own right, and otherwise always the last fallback)
            order = NEGATIVE_TYPES[type_index % len(NEGATIVE_TYPES):] + NEGATIVE_TYPES[:type_index % len(NEGATIVE_TYPES)]
            if order[0] != 'random':
                order.remove('random')
                order.append('random')
            type_index += 1

            negative, used_type = None, None
            for neg_type in order:
                negative = sample_negative(neg_type, subject, parent, subject_parents, allowed, concepts_per_sem_type)
                if negative is not None:
                    used_type = neg_type
                    break
            if negative is None:
                continue  # Skip the positive too, so the dataset stays balanced

            rows.append({'subject_id': subject, 'object_id': parent, 'is_son_of': 'YES', 'negative_type': 'positive',
                         'subject_fsn': snomed.get_fsn(subject), 'object_fsn': snomed.get_fsn(parent)})
            rows.append({'subject_id': subject, 'object_id': negative, 'is_son_of': 'NO', 'negative_type': used_type,
                         'subject_fsn': snomed.get_fsn(subject), 'object_fsn': snomed.get_fsn(negative)})

    df = pd.DataFrame(rows)
    df.to_csv(f'../datasets filtered top/datasets_is_son_of/dataset_is_son_of_v3_hard_{split}.csv', index=False)
    print(split, len(df), df['negative_type'].value_counts().to_dict())

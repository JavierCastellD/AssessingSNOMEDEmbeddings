import warnings
import numpy as np
import pandas as pd
from snomed import Snomed
from embedding_models.fasttext_EM import FastTextEM
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# The current tasks that we are considering that require using classification are:
# 1. Is son of, i.e., given two concepts, whether the first one is a direct son of the second one
# 2. Is there relation, i.e., given two concepts, whether there is any kind of non-is-a relationship between them
# 3. Semantic type classification, i.e., given a concept, identify its semantic type
# 4. Top level hierarchy classification, i.e., given a concept, identify the hierarchy to which it belongs
# 5. Triple classification, i.e., given a triple <subject, relation, object>, whether it is valid or not

# Identify the task
task = 1

# Load SNOMED CT
con_path = "snomed_data/conceptInternational_20221031.txt"
rel_path = "snomed_data/relationshipInternational_20221031.txt"
desc_path = "snomed_data/descriptionInternational_20221031.txt"
snomed = Snomed(con_path, rel_path, desc_path)

# Load the embedding model
embedding_model = FastTextEM("TODO")

# Load the dataset
if task == 1: # is son of
    dataset_train = pd.read_csv('datasets/datasets_is_son_of/dataset_is_son_of_v2_train.csv')
    dataset_dev = pd.read_csv('datasets/datasets_is_son_of/dataset_is_son_of_v2_dev.csv')

    target_column = 'is_son_of'

elif task == 2: # is there relation
    dataset_train = pd.read_csv('datasets/datasets_is_there_relation/dataset_is_there_relation_v2_train.csv')
    dataset_dev = pd.read_csv('datasets/datasets_is_there_relation/dataset_is_there_relation_v2_dev.csv')

    target_column = 'relation_status'

elif task == 3: # semantic type classification
    dataset_train = pd.read_csv('datasets/datasets_semantic_type/dataset_semantic_type_train.csv')
    dataset_dev = pd.read_csv('datasets/datasets_semantic_type/dataset_semantic_type_dev.csv')

    target_column = 'semantic_type'
elif task == 4: # top level hierarchy classification
    dataset_train = pd.read_csv('datasets/datasets_top_level_hierarchy/dataset_top_level_hierarchy_train.csv')
    dataset_dev = pd.read_csv('datasets/datasets_top_level_hierarchy/dataset_top_level_hierarchy_dev.csv')

    target_column = 'top_level'
elif task == 5: # triple classification
    dataset_train = pd.read_csv('datasets/datasets_triple_classification/dataset_triple_classification_v2_train.csv')
    dataset_dev = pd.read_csv('datasets/datasets_triple_classification/dataset_triple_classification_v2_dev.csv')

    target_column = 'triple_status'
else:
    warnings.warn('No correct task chosen.')

# Transform the dataset into the input data
le = LabelEncoder()
X_train = []
y_train = le.fit_transform(dataset_train[target_column])

X_test = []
y_test = le.transform(dataset_dev[target_column])
if task == 1 or task == 2:
    for conceptA_id, conceptB_id in zip(dataset_train['subject_id'], dataset_train['object_id']):
        conceptA_names = snomed.get_descriptions(conceptA_id)
        conceptB_names = snomed.get_descriptions(conceptB_id)

        conceptA_emb = embedding_model.get_embedding_from_list(conceptA_names)
        conceptB_emb = embedding_model.get_embedding_from_list(conceptB_names)

        X_train.append(np.concatenate([conceptA_emb, conceptB_emb]))

    for conceptA_id, conceptB_id in zip(dataset_dev['subject_id'], dataset_dev['object_id']):
        conceptA_names = snomed.get_descriptions(conceptA_id)
        conceptB_names = snomed.get_descriptions(conceptB_id)

        conceptA_emb = embedding_model.get_embedding_from_list(conceptA_names)
        conceptB_emb = embedding_model.get_embedding_from_list(conceptB_names)

        X_test.append(np.concatenate([conceptA_emb, conceptB_emb]))
elif task == 3 or task == 4:
    for concept_id in dataset_train['concept_id']:
        concept_names = snomed.get_descriptions(concept_id)
        concept_emb = embedding_model.get_embedding_from_list(concept_names)

        X_train.append(concept_emb)

    for concept_id in dataset_dev['concept_id']:
        concept_names = snomed.get_descriptions(concept_id)
        concept_emb = embedding_model.get_embedding_from_list(concept_names)

        X_test.append(concept_emb)
elif task == 5:
    for subject_id, relation_id, object_id in zip(dataset_train['subject_id'], dataset_train['relation_id'], dataset_train['object_id']):
        subject_names = snomed.get_descriptions(subject_id)
        object_names = snomed.get_descriptions(object_id)
        relation_name = snomed.get_fsn(relation_id)

        subject_emb = embedding_model.get_embedding_from_list(subject_names)
        object_emb = embedding_model.get_embedding_from_list(object_names)
        relation_emb = embedding_model.get_embedding(relation_name)
        # relation_emb = embedding_model.get_embedding(str(relation_id))

        X_train.append(np.concatenate([subject_emb, relation_emb, object_emb]))
    
    for subject_id, relation_id, object_id in zip(dataset_dev['subject_id'], dataset_dev['relation_id'], dataset_dev['object_id']):
        subject_names = snomed.get_descriptions(subject_id)
        object_names = snomed.get_descriptions(object_id)
        relation_name = snomed.get_fsn(relation_id)

        subject_emb = embedding_model.get_embedding_from_list(subject_names)
        object_emb = embedding_model.get_embedding_from_list(object_names)
        relation_emb = embedding_model.get_embedding(relation_name)
        # relation_emb = embedding_model.get_embedding(str(relation_id))

        X_test.append(np.concatenate([subject_emb, relation_emb, object_emb]))

# Use GridSearchCV for hyperparameter optimization for RandomForest
clf_gs = GridSearchCV(RandomForestClassifier(random_state=42), cv=5, verbose=1,
                      param_grid={'n_estimators' : [10, 50, 100, 200, 500], 'max_depth' : [5, 10, 20, 40, 50, 100, 200],
                                  'min_samples_split' : [2, 4], 'min_samples_leaf' : [1, 3], 'max_features' : ['sqrt', 'log2']},
                      scoring='accuracy')

# Train the classifier
print('Training')
clf_gs.fit(X_train, y_train)

# Test it on the test (dev) set
y_pred = clf_gs.predict(X_test)

# Print out the results
print('Accuracy:', round(accuracy_score(y_test, y_pred) * 100, 2))
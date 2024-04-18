import json
import os.path
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from python_libraries.embedding_models.fasttext_EM import FastTextEM
from python_libraries.embedding_models.sentencetransformer_EM import SentenceTransformerEM
from python_libraries.nn_embedding_classifier import EmbeddingClassifier
from python_libraries.snomed import Snomed

# The current tasks that we are considering that require using classification are:
# 1. Is son of, i.e., given two concepts, whether the first one is a direct son of the second one
# 2. Is there relation, i.e., given two concepts, whether there is any kind of non-is-a relationship between them
# 3. Semantic type classification, i.e., given a concept, identify its semantic type
# 4. Top level hierarchy classification, i.e., given a concept, identify the hierarchy to which it belongs
# 5. Triple classification, i.e., given a triple <subject, relation, object>, whether it is valid or not

# Identify the task
task = 1

# Load SNOMED CT
con_path = "snomed_data/conceptInternational_20240101.txt"
rel_path = "snomed_data/relationshipInternational_20240101.txt"
desc_path = "snomed_data/descriptionInternational_20240101.txt"
snomed = Snomed(con_path, rel_path, desc_path)

# Load the embedding model
embedding_model = FastTextEM(model_path="models/ft_2_2_20221031.model")

# Load the dictionary if we are using SBERT
concept_dictionary = None
if isinstance(embedding_model, SentenceTransformerEM):
    dict_test = open('concepts_dictionaries/full_train_mini_lm_3_1_sct_dict.json')
    concept_dictionary = json.load(dict_test)
    dict_test.close()

# Load the dataset
if task == 1: # is son of
    dataset_train = pd.read_csv('datasets/datasets_is_son_of/dataset_is_son_of_v2_train.csv')
    dataset_dev = pd.read_csv('datasets/datasets_is_son_of/dataset_is_son_of_v2_dev.csv')

    target_column = 'is_son_of'
    n_embeddings = 2
elif task == 2: # is there relation
    dataset_train = pd.read_csv('datasets/datasets_is_there_relation/dataset_is_there_relation_v2_train.csv')
    dataset_dev = pd.read_csv('datasets/datasets_is_there_relation/dataset_is_there_relation_v2_dev.csv')

    target_column = 'relation_status'
    n_embeddings = 2
elif task == 3: # semantic type classification
    dataset_train = pd.read_csv('datasets/datasets_semantic_type/dataset_semantic_type_train.csv')
    dataset_dev = pd.read_csv('datasets/datasets_semantic_type/dataset_semantic_type_dev.csv')

    target_column = 'semantic_type'
    n_embeddings = 1
elif task == 4: # top level hierarchy classification
    dataset_train = pd.read_csv('datasets/datasets_top_level_hierarchy/dataset_top_level_hierarchy_train.csv')
    dataset_dev = pd.read_csv('datasets/datasets_top_level_hierarchy/dataset_top_level_hierarchy_dev.csv')

    target_column = 'top_level'
    n_embeddings = 1
elif task == 5: # triple classification
    dataset_train = pd.read_csv('datasets/datasets_triple_classification/dataset_triple_classification_v2_train.csv')
    dataset_dev = pd.read_csv('datasets/datasets_triple_classification/dataset_triple_classification_v2_dev.csv')

    target_column = 'triple_status'
    n_embeddings = 3
else:
    warnings.warn('No correct task chosen.')
    exit()

# Transform the dataset into the input data
ohe = OneHotEncoder(sparse_output=False)
X_train = []
y_train = ohe.fit_transform(dataset_train[target_column].to_numpy().reshape(-1, 1))

X_test = []
y_test = ohe.fit_transform(dataset_dev[target_column].to_numpy().reshape(-1, 1))
if task == 1 or task == 2:
    for conceptA_id, conceptB_id in zip(dataset_train['subject_id'], dataset_train['object_id']):
        if concept_dictionary is not None:
            conceptA_emb = np.array(concept_dictionary[str(conceptA_id)])
            conceptB_emb = np.array(concept_dictionary[str(conceptB_id)])
        else:
            conceptA_names = snomed.get_descriptions(conceptA_id)
            conceptB_names = snomed.get_descriptions(conceptB_id)
    
            conceptA_emb = embedding_model.get_embedding_from_list(conceptA_names)
            conceptB_emb = embedding_model.get_embedding_from_list(conceptB_names)

        X_train.append(np.concatenate([conceptA_emb, conceptB_emb]))

    for conceptA_id, conceptB_id in zip(dataset_dev['subject_id'], dataset_dev['object_id']):
        if concept_dictionary is not None:
            conceptA_emb = np.array(concept_dictionary[str(conceptA_id)])
            conceptB_emb = np.array(concept_dictionary[str(conceptB_id)])
        else:
            conceptA_names = snomed.get_descriptions(conceptA_id)
            conceptB_names = snomed.get_descriptions(conceptB_id)

            conceptA_emb = embedding_model.get_embedding_from_list(conceptA_names)
            conceptB_emb = embedding_model.get_embedding_from_list(conceptB_names)

        X_test.append(np.concatenate([conceptA_emb, conceptB_emb]))
elif task == 3 or task == 4:
    for concept_id in dataset_train['concept_id']:
        if concept_dictionary is not None:
            concept_emb = np.array(concept_dictionary[str(concept_id)])
        else:
            concept_names = snomed.get_descriptions(concept_id)
            concept_emb = embedding_model.get_embedding_from_list(concept_names)

        X_train.append(np.array(concept_emb))

    for concept_id in dataset_dev['concept_id']:
        if concept_dictionary is not None:
            concept_emb = concept_dictionary[str(concept_id)]
        else:
            concept_names = snomed.get_descriptions(concept_id)
            concept_emb = embedding_model.get_embedding_from_list(concept_names)

        X_test.append(np.array(concept_emb))
elif task == 5:
    for subject_id, relation_id, object_id in zip(dataset_train['subject_id'], dataset_train['relation_id'], dataset_train['object_id']):
        if concept_dictionary is not None:
            relation_name = snomed.get_fsn(relation_id)
        
            subject_emb = np.array(concept_dictionary[str(subject_id)])
            object_emb = np.array(concept_dictionary[str(object_id)])
            relation_emb = embedding_model.get_embedding(relation_name)
        else:
            subject_names = snomed.get_descriptions(subject_id)
            object_names = snomed.get_descriptions(object_id)
            relation_name = snomed.get_fsn(relation_id)
    
            subject_emb = embedding_model.get_embedding_from_list(subject_names)
            object_emb = embedding_model.get_embedding_from_list(object_names)
            relation_emb = embedding_model.get_embedding(relation_name)
            # relation_emb = embedding_model.get_embedding(str(relation_id))

        X_train.append(np.concatenate([subject_emb, relation_emb, object_emb]))
    
    for subject_id, relation_id, object_id in zip(dataset_dev['subject_id'], dataset_dev['relation_id'], dataset_dev['object_id']):
        if concept_dictionary is not None:
            relation_name = snomed.get_fsn(relation_id)
        
            subject_emb = np.array(concept_dictionary[str(subject_id)])
            object_emb = np.array(concept_dictionary[str(object_id)])
            relation_emb = embedding_model.get_embedding(relation_name)
        else:
            subject_names = snomed.get_descriptions(subject_id)
            object_names = snomed.get_descriptions(object_id)
            relation_name = snomed.get_fsn(relation_id)
    
            subject_emb = embedding_model.get_embedding_from_list(subject_names)
            object_emb = embedding_model.get_embedding_from_list(object_names)
            relation_emb = embedding_model.get_embedding(relation_name)
            # relation_emb = embedding_model.get_embedding(str(relation_id))

        X_test.append(np.concatenate([subject_emb, relation_emb, object_emb]))


# Transform it into numpy array
X_train = np.array(X_train)
X_test = np.array(X_test)

# Define the neural network hyperparameters
batch_size = 128
epochs = 50
dense_layers = [300, 200, 100]
activation_function = 'relu'
kernel_initializer = 'glorot_uniform'
dropout_layers = None
batch_normalization = None
regularization_type = None
l1_regularization = 0.01
l2_regularization = 0.01
optimizer_function = 'adam'
learning_rate = 0.001

number_of_classes = y_train.shape[1]

log_output_file_name = 'FT_Task_' + target_column + '_nn_results.csv'

# We will only need to write the header if the file did not exist
header_needed = not os.path.isfile(log_output_file_name)

with open(log_output_file_name, 'a') as log_output:

    if header_needed:
        log_output.write('Batch size;Epochs;Dense layers;Activation function;Kernel initializer;Dropout layers;Batch normalization;Regularization type;Optimizer function;Learning rate;Accuracy;Loss;Val_accuracy;Val_loss\n')

    for batch_size in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        # Create the NN
        nn = EmbeddingClassifier(number_of_classes, n_embeddings, batch_size, epochs, dense_layers,
                                dropout_layers, activation_function, kernel_initializer,
                                batch_normalization, regularization_type, l1_regularization,
                                l2_regularization, optimizer_function, learning_rate)

        # Train the model
        history = nn.train_model(X_train, y_train, X_test, y_test, verbose=2)

        # Transform it into a dataframe
        results = pd.DataFrame(history.history)

        # Obtain the metrics
        accuracy = round(results.categorical_accuracy.values[-1:][0] * 100, 2)
        val_accuracy = round(results.val_categorical_accuracy.values[-1:][0]*100, 2)
        loss = round(results.loss.values[-1:][0], 4)
        val_loss = round(results.val_loss.values[-1:][0], 4)

        # Write to log
        log_output.write(str(batch_size) + ';' + str(epochs) + ';' + str(dense_layers) + ';' + str(activation_function) + ';' + str(kernel_initializer) + ';' + str(dropout_layers) + ';' + str(batch_normalization) + ';' + str(regularization_type) + ';' + str(optimizer_function) + ';' + str(learning_rate) + ';' + str(accuracy) + ';' + str(loss) + ';' + str(val_accuracy) + ';' + str(val_loss) + '\n')
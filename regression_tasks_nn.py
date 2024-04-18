import json
import os.path

import numpy as np
import pandas as pd

from python_libraries.embedding_models.fasttext_EM import FastTextEM
from python_libraries.embedding_models.sentencetransformer_EM import SentenceTransformerEM
from python_libraries.nn_embedding_predictor import EmbeddingPredictor
from python_libraries.snomed import Snomed


# The current tasks that we are considering that require using multi-regression are:
# 1. Relation prediction, i.e., given a subject and object, predict which relation exists between them
# 2. Analogy/Object prediction, i.e., given a subject and a relationship, predict the object of that triple

# If the task is relation prediction, it should be set to true
# If the task is analogy prediction, it should be set to false
relation_prediction = True

# Load SNOMED CT
con_path = "snomed_data/conceptInternational_20240101.txt"
rel_path = "snomed_data/relationshipInternational_20240101.txt"
desc_path = "snomed_data/descriptionInternational_20240101.txt"
snomed = Snomed(con_path, rel_path, desc_path)

# Load the model
embedding_model = FastTextEM(model_path="models/ft_2_2_20221031.model")

# Load the dictionary if we are using SBERT
concept_dictionary = None
if isinstance(embedding_model, SentenceTransformerEM):
    dict_test = open('concepts_dictionaries/full_train_mini_lm_3_1_sct_dict.json')
    concept_dictionary = json.load(dict_test)
    dict_test.close()

# Load the dataset
if relation_prediction:
    dataset_train = pd.read_csv('datasets/datasets_relation_prediction/dataset_relation_prediction_train.csv')
    dataset_dev = pd.read_csv('datasets/datasets_relation_prediction/dataset_relation_prediction_dev.csv')
else:
    dataset_train = pd.read_csv('datasets/datasets_analogy_prediction/dataset_analogy_prediction_train.csv')
    dataset_dev = pd.read_csv('datasets/datasets_analogy_prediction/dataset_analogy_prediction_dev.csv')

# Create the input and target
X_train = []
y_train = []

X_test = []
y_test = []
y_test_ids = []

if relation_prediction:
    task = 'relation_prediction'
    for subject_id, object_id, relation_id in zip(dataset_train['subject_id'], dataset_train['object_id'], dataset_train['relation_id']):
        if concept_dictionary is not None:
            subject_embedding = np.array(concept_dictionary[str(subject_id)])
            object_embedding = np.array(concept_dictionary[str(object_id)])
        else:
            subject_descriptions = snomed.get_descriptions(subject_id)
            object_descriptions = snomed.get_descriptions(object_id)

            subject_embedding = embedding_model.get_embedding_from_list(subject_descriptions)
            object_embedding = embedding_model.get_embedding_from_list(object_descriptions)

        X_train.append(np.concatenate([subject_embedding, object_embedding]))
        
        relation_name = snomed.get_fsn(relation_id)
        relation_embedding = embedding_model.get_embedding(relation_name)
        # relation_embedding = embedding_model.get_embedding(str(relation_id))
        y_train.append(relation_embedding)
    
    for subject_id, object_id, relation_id in zip(dataset_dev['subject_id'], dataset_dev['object_id'], dataset_dev['relation_id']):
        if concept_dictionary is not None:
            subject_embedding = np.array(concept_dictionary[str(subject_id)])
            object_embedding = np.array(concept_dictionary[str(object_id)])
        else:
            subject_descriptions = snomed.get_descriptions(subject_id)
            object_descriptions = snomed.get_descriptions(object_id)

            subject_embedding = embedding_model.get_embedding_from_list(subject_descriptions)
            object_embedding = embedding_model.get_embedding_from_list(object_descriptions)

        X_test.append(np.concatenate([subject_embedding, object_embedding]))

        relation_name = snomed.get_fsn(relation_id)
        relation_embedding = embedding_model.get_embedding(relation_name)
        # relation_embedding = embedding_model.get_embedding(str(relation_id))
        y_test.append(relation_embedding)
        y_test_ids.append(relation_id)
else:
    task = 'analogy_prediction'
    for subject_id, relation_id, object_id  in zip(dataset_train['subject_id'], dataset_train['relation_id'], dataset_train['object_id']):
        if concept_dictionary is not None:
            subject_embedding = np.array(concept_dictionary[str(subject_id)])
            object_embedding = np.array(concept_dictionary[str(object_id)])
        else:
            subject_descriptions = snomed.get_descriptions(subject_id)
            subject_embedding = embedding_model.get_embedding_from_list(subject_descriptions)

            object_descriptions = snomed.get_descriptions(object_id)
            object_embedding = embedding_model.get_embedding_from_list(object_descriptions)

        relation_name = snomed.get_fsn(relation_id)
        relation_embedding = embedding_model.get_embedding(relation_name)
        # relation_embedding = embedding_model.get_embedding(str(relation_id))

        X_train.append(np.concatenate([subject_embedding, relation_embedding]))
        y_train.append(object_embedding)
    
    for subject_id, relation_id, object_id in zip(dataset_dev['subject_id'], dataset_dev['relation_id'], dataset_dev['object_id']):
        if concept_dictionary is not None:
            subject_embedding = np.array(concept_dictionary[str(subject_id)])
            object_embedding = np.array(concept_dictionary[str(object_id)])
        else:
            subject_descriptions = snomed.get_descriptions(subject_id)
            subject_embedding = embedding_model.get_embedding_from_list(subject_descriptions)

            object_descriptions = snomed.get_descriptions(object_id)
            object_embedding = embedding_model.get_embedding_from_list(object_descriptions)

        relation_name = snomed.get_fsn(relation_id)
        relation_embedding = embedding_model.get_embedding(relation_name)
        # relation_embedding = embedding_model.get_embedding(str(relation_id))

        X_test.append(np.concatenate([subject_embedding, relation_embedding]))
        y_test.append(object_embedding)
        y_test_ids.append(object_id)

# Transform it into numpy array
X_train = np.array(X_train)
X_test = np.array(X_test)

# Obtain the target space
target_space = {}

if relation_prediction:
    for target_id, target_name in zip(dataset_train['relation_id'].unique(), dataset_train['relation_fsn'].unique()):
        target_space[target_id] = embedding_model.get_embedding(target_name)
        # TODO: If we are working with free text embeddings, we should use the name
        # TODO: If we are working with SCT embeddings, we either use ID or name, we have to test it
        #target_space[target_id] = embedding_model.get_embedding(str(target_id))
else:
    for concept_id in snomed.get_sct_concepts(metadata=False):
        descriptions = snomed.get_descriptions(concept_id)
        target_space[concept_id] = embedding_model.get_embedding_from_list(descriptions)

# Define the neural network hyperparameters
batch_size = 256
epochs = 100
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

log_output_file_name = 'FT_Task_' + task + '_nn_results.csv'

# We will only need to write the header if the file did not exist
header_needed = not os.path.isfile(log_output_file_name)

with open(log_output_file_name, 'a') as log_output:

    if header_needed:
        log_output.write('Batch size;Epochs;Dense layers;Activation function;Kernel initializer;Dropout layers;Batch normalization;Regularization type;Optimizer function;Learning rate;Loss;Val_loss;Accuracy;Top 3;Top 5;Top 10\n')


    # Create the NN
    nn = EmbeddingPredictor(target_space, batch_size, epochs, dense_layers, dropout_layers, activation_function, 
                            kernel_initializer, batch_normalization, regularization_type, l1_regularization, 
                            l2_regularization, optimizer_function, learning_rate,
                            relation_prediction=relation_prediction)

    # Train the model
    history = nn.train_model(X_train, y_train, X_test, y_test, verbose=2)

    # Obtain the predictions
    predictions = nn.predict(X_test)

    # Calculate the loss
    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    # Calculate the accuracy and top3, top5, and top10 hits
    accuracy = 0
    top3 = 0
    top5 = 0
    top10 = 0

    for top10_prediction, rel_id in zip(predictions, y_test_ids):
        predicted_ids = [predicted_id for _, predicted_id in top10_prediction]
        
        # Find if any of the top 10 predictions for the given row is correct
        # and in which position among them
        correct_prediction_pos = 0
        for predicted_id in predicted_ids:
            if predicted_id == rel_id:
                break
            correct_prediction_pos += 1
        
        if correct_prediction_pos == 0:
            accuracy += 1
            top3 += 1
            top5 += 1
            top10 += 1
        elif correct_prediction_pos < 3:
            top3 += 1
            top5 += 1
            top10 += 1
        elif correct_prediction_pos < 5:
            top5 += 1
            top10 += 1
        elif correct_prediction_pos < 10:
            top10 += 1

    accuracy = round(accuracy/len(predictions)*100, 2)
    top3 = round(top3/len(predictions)*100, 2)
    top5 = round(top5/len(predictions)*100, 2)
    top10 = round(top10/len(predictions)*100, 2)

    # Write to log
    log_output.write(str(batch_size) + ';' + str(epochs) + ';' + str(dense_layers) + ';' + str(activation_function) + ';' + str(kernel_initializer) + ';' + str(dropout_layers) + ';' + str(batch_normalization) + ';' + str(regularization_type) + ';' + str(optimizer_function) + ';' + str(learning_rate) + ';' + str(loss) + ';' + str(val_loss) + ';' + str(accuracy) + ';' + str(top3) + ';' + str(top5) + ';' + str(top10) + '\n')
import os
import os.path

import numpy as np
import pandas as pd

from python_libraries.embedding_models.embedding_model import load_embeddings
from python_libraries.embedding_models.fasttext_EM import FastTextEM
from python_libraries.embedding_models.sentencetransformer_EM import SentenceTransformerEM
from python_libraries.nn_embedding_predictor import EmbeddingPredictor
from python_libraries.snomed import Snomed


# The current tasks that we are considering that require using multi-regression are:
# 1. Relation prediction, i.e., given a subject and object, predict which relation exists between them
# 2. Analogy/Object prediction, i.e., given a subject and a relationship, predict the object of that triple

# If the task is relation prediction, it should be set to true
# If the task is analogy prediction, it should be set to false
relation_prediction = False

# Load SNOMED CT
CONCEPTS_PATH = "snomed_data/conceptInternational_20240101.txt"
RELATIONS_PATH = "snomed_data/relationshipInternational_20240101.txt"
DESCRIPTIONS_PATH = "snomed_data/descriptionInternational_20240101.txt"
snomed = Snomed(CONCEPTS_PATH, RELATIONS_PATH, DESCRIPTIONS_PATH)

# Load the model
model_log_name = 'results/MIMIC_10_Task_'
embedding_model = FastTextEM(model_path="models/ft_train_MIMIC_10.model")

# Load the dictionary if we are using SBERT
concept_dictionary = None
if isinstance(embedding_model, SentenceTransformerEM):
    concept_dictionary = load_embeddings('concepts_dictionaries/full_train_mini_lm_3_1_sct_dict.npz')

# Relation cache to speed up relation embedding inference
relation_cache = {}

# Define the neural network hyperparameters
batch_size = 1024 # 524
epochs = 50
dense_layers = [2000, 2000, 2000, 2000] # [300, 200, 100]
activation_function = 'relu'
kernel_initializer = 'glorot_uniform'
dropout_layers = None
batch_normalization = None
regularization_type = None
l1_regularization = 0.01
l2_regularization = 0.01
optimizer_function = 'adamax' # adam
learning_rate = 0.001

# Load the dataset
if relation_prediction:
    dataset_train = pd.read_csv('datasets/datasets_relation_prediction/dataset_relation_prediction_train.csv')
    dataset_train_2 = pd.read_csv('datasets/datasets_relation_prediction/dataset_relation_prediction_dev.csv')
    dataset_train = pd.concat([dataset_train, dataset_train_2])
    dataset_dev = pd.read_csv('datasets/datasets_relation_prediction/dataset_relation_prediction_test.csv')
else:
    dataset_train = pd.read_csv('datasets/datasets_analogy_prediction/dataset_analogy_prediction_train.csv')
    dataset_train_2 = pd.read_csv('datasets/datasets_analogy_prediction/dataset_analogy_prediction_dev.csv')
    dataset_train = pd.concat([dataset_train, dataset_train_2])    
    dataset_dev = pd.read_csv('datasets/datasets_analogy_prediction/dataset_analogy_prediction_test.csv')

# Create the input and target
X_train = []
y_train = []
y_train_ids = []

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
        
        if relation_id not in relation_cache:
            relation_name = snomed.get_fsn(relation_id)
            if isinstance(embedding_model, FastTextEM):
                relation_embedding = embedding_model.get_embedding(str(relation_id))
            else:
                relation_embedding = embedding_model.get_embedding(relation_name)
            
            relation_cache[relation_id] = relation_embedding
        
        relation_embedding = relation_cache[relation_id]
        
        y_train.append(relation_embedding)
        y_train_ids.append(relation_id)
    
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

        if relation_id not in relation_cache:
            relation_name = snomed.get_fsn(relation_id)
            if isinstance(embedding_model, FastTextEM):
                relation_embedding = embedding_model.get_embedding(str(relation_id))
            else:
                relation_embedding = embedding_model.get_embedding(relation_name)
            
            relation_cache[relation_id] = relation_embedding
        
        relation_embedding = relation_cache[relation_id]

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

        if relation_id not in relation_cache:
            relation_name = snomed.get_fsn(relation_id)
            if isinstance(embedding_model, FastTextEM):
                relation_embedding = embedding_model.get_embedding(str(relation_id))
            else:
                relation_embedding = embedding_model.get_embedding(relation_name)
            
            relation_cache[relation_id] = relation_embedding
        
        relation_embedding = relation_cache[relation_id]

        X_train.append(np.concatenate([subject_embedding, relation_embedding]))
        y_train.append(object_embedding)
        y_train_ids.append(object_id)
    
    for subject_id, relation_id, object_id in zip(dataset_dev['subject_id'], dataset_dev['relation_id'], dataset_dev['object_id']):
        if concept_dictionary is not None:
            subject_embedding = np.array(concept_dictionary[str(subject_id)])
            object_embedding = np.array(concept_dictionary[str(object_id)])
        else:
            subject_descriptions = snomed.get_descriptions(subject_id)
            subject_embedding = embedding_model.get_embedding_from_list(subject_descriptions)

            object_descriptions = snomed.get_descriptions(object_id)
            object_embedding = embedding_model.get_embedding_from_list(object_descriptions)

        if relation_id not in relation_cache:
            relation_name = snomed.get_fsn(relation_id)
            if isinstance(embedding_model, FastTextEM):
                relation_embedding = embedding_model.get_embedding(str(relation_id))
            else:
                relation_embedding = embedding_model.get_embedding(relation_name)
            
            relation_cache[relation_id] = relation_embedding
        
        relation_embedding = relation_cache[relation_id]

        X_test.append(np.concatenate([subject_embedding, relation_embedding]))
        y_test.append(object_embedding)
        y_test_ids.append(object_id)

# Transform it into numpy array
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)

y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# Obtain the target space
target_space = {}

if relation_prediction:
    for target_id, target_name in zip(dataset_train['relation_id'].unique(), dataset_train['relation_fsn'].unique()):
        if target_id not in relation_cache:
            if isinstance(embedding_model, FastTextEM):
                target_embedding = embedding_model.get_embedding(str(target_id))
            else:
                target_embedding = embedding_model.get_embedding(target_name)
            
            relation_cache[target_id] = target_embedding

        target_space[target_id] = relation_cache[target_id]
else:
    for concept_id in snomed.get_sct_concepts(metadata=False):
        if concept_dictionary is not None:
            target_embedding = concept_dictionary[str(concept_id)]
        else:
            descriptions = snomed.get_descriptions(concept_id)
            target_embedding = embedding_model.get_embedding_from_list(descriptions)

        target_space[concept_id] = target_embedding

embedding_size = len(list(target_space.values())[0])

log_output_file_name = model_log_name + task + '_nn_results.csv'

# We will only need to write the header if the file did not exist
header_needed = not os.path.isfile(log_output_file_name)

with open(log_output_file_name, 'a') as log_output:

    if header_needed:
        log_output.write('Batch size;Epochs;Dense layers;Activation function;Kernel initializer;Dropout layers;Batch normalization;Regularization type;Optimizer function;Learning rate;Loss;Train accuracy;Train top 3;Train top 5;Train top 10;Val_loss;Accuracy;Top 3;Top 5;Top 10\n')
        
    # Create the NN
    nn = EmbeddingPredictor(target_space, batch_size, epochs, dense_layers, dropout_layers, activation_function, 
                            kernel_initializer, batch_normalization, regularization_type, l1_regularization, 
                            l2_regularization, optimizer_function, learning_rate,
                            relation_prediction=relation_prediction, embedding_size=embedding_size)

    # Train the model
    history = nn.train_model(X_train, y_train, X_test, y_test, verbose=2)

    # Obtain the train predictions
    train_predictions = nn.predict_faiss(X_train)

    # Obtain the predictions
    predictions = nn.predict_faiss(X_test)

    # Calculate the loss
    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    # Calculate the accuracy and top3, top5, and top10 hits
    train_accuracy = 0
    train_top3 = 0
    train_top5 = 0
    train_top10 = 0

    for top10_prediction, rel_id in zip(train_predictions, y_train_ids):
        predicted_ids = [predicted_id for _, predicted_id in top10_prediction]
        
        # Find if any of the top 10 predictions for the given row is correct
        # and in which position among them
        correct_prediction_pos = 0
        for predicted_id in predicted_ids:
            if predicted_id == rel_id:
                break
            correct_prediction_pos += 1
        
        if correct_prediction_pos == 0:
            train_accuracy += 1
            train_top3 += 1
            train_top5 += 1
            train_top10 += 1
        elif correct_prediction_pos < 3:
            train_top3 += 1
            train_top5 += 1
            train_top10 += 1
        elif correct_prediction_pos < 5:
            train_top5 += 1
            train_top10 += 1
        elif correct_prediction_pos < 10:
            train_top10 += 1

    train_accuracy = round(train_accuracy/len(train_predictions)*100, 2)
    train_top3 = round(train_top3/len(train_predictions)*100, 2)
    train_top5 = round(train_top5/len(train_predictions)*100, 2)
    train_top10 = round(train_top10/len(train_predictions)*100, 2)


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
    log_output.write(str(batch_size) + ';' + str(epochs) + ';' + str(dense_layers) + ';' + str(activation_function) + ';' + str(kernel_initializer) + ';' + str(dropout_layers) + ';' + str(batch_normalization) + ';' + str(regularization_type) + ';' + str(optimizer_function) + ';' + str(learning_rate) + ';' + str(loss) + ';' + str(train_accuracy) + ';' + str(train_top3) + ';' + str(train_top5) + ';' + str(train_top10) + ';' + str(val_loss) + ';' + str(accuracy) + ';' + str(top3) + ';' + str(top5) + ';' + str(top10) + '\n')
    log_output.flush()
    os.fsync(log_output.fileno())
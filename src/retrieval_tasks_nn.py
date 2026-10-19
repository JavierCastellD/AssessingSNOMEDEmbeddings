import argparse
import os
import os.path

import numpy as np
from tensorflow.keras import backend as K

from python_libraries.nn_embedding_predictor import EmbeddingPredictor
from python_libraries.probing.config import ENCODERS, PROBES
from python_libraries.probing.tasks import RETRIEVAL_TASKS, create_input, get_embedding, load_dataset, load_approach_models
from python_libraries.snomed import Snomed

# The current tasks that we are considering that require retrieval are:
# 1. Relation prediction, i.e., given a subject and object, predict which relation exists between them
# 2. Object prediction, i.e., given a subject and a relationship, predict the object of that triple
#
# The probe is trained as a regressor that predicts the embedding of the missing element, which is then used to
# retrieve the most similar relationships or concepts.
#
# Usage:
#   python src/retrieval_tasks_nn.py --encoder <approach> [--tasks <tasks>] [--probe main|simple] [--iterations N]
# Example:
#   python src/retrieval_tasks_nn.py --encoder fine_tuned_st --tasks relation_prediction --iterations 10

# Paths to SNOMED CT
CONCEPTS_PATH = "snomed_data/conceptInternational_20240101.txt"
RELATIONS_PATH = "snomed_data/relationshipInternational_20240101.txt"
DESCRIPTIONS_PATH = "snomed_data/descriptionInternational_20240101.txt"

LOG_HEADER = 'Batch size;Epochs;Dense layers;Activation function;Kernel initializer;Dropout layers;Batch normalization;Regularization type;Optimizer function;Learning rate;Loss;Train accuracy;Train top 3;Train top 5;Train top 10;Val_loss;Accuracy;Top 3;Top 5;Top 10\n'

def compute_hits(predictions : list, true_ids : list) -> tuple[float, float, float, float]:
    '''Function to calculate the accuracy (Hits@1), and the Hits@3, Hits@5 and Hits@10 of the predictions.

    Parameters:
        predictions (list):
            List with the top 10 predictions for each sample, ordered from most to least similar. Each
            prediction can be either an identifier or a tuple (similarity value, identifier).
        true_ids (list):
            List with the correct identifier of each sample.

    Returns:
        A tuple with the accuracy, top 3, top 5 and top 10, as percentages.
    '''
    accuracy = 0
    top3 = 0
    top5 = 0
    top10 = 0

    for top10_prediction, true_id in zip(predictions, true_ids):
        predicted_ids = [prediction[1] if isinstance(prediction, tuple) else prediction for prediction in top10_prediction]

        # Find if any of the top 10 predictions for the given row is correct
        # and in which position among them
        correct_prediction_pos = 0
        for predicted_id in predicted_ids:
            if predicted_id == true_id:
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

    return accuracy, top3, top5, top10

# Read the arguments
parser = argparse.ArgumentParser(description='Probing of SNOMED CT embeddings on the retrieval tasks.')
parser.add_argument('--encoder', required=True, choices=ENCODERS.keys(), help='Embedding approach to evaluate.')
parser.add_argument('--tasks', nargs='+', default=list(RETRIEVAL_TASKS.keys()), choices=RETRIEVAL_TASKS.keys(), help='Tasks to evaluate.')
parser.add_argument('--probe', default='main', choices=PROBES.keys(), help='Probe to use.')
parser.add_argument('--iterations', type=int, default=10, help='Number of times each probe is trained and evaluated.')
parser.add_argument('--results_path', default='results/', help='Folder where the results are stored.')
args = parser.parse_args()

os.makedirs(args.results_path, exist_ok=True)

# Load SNOMED CT
snomed = Snomed(CONCEPTS_PATH, RELATIONS_PATH, DESCRIPTIONS_PATH)

# Load the models of the chosen approach
encoder_config = ENCODERS[args.encoder]
approach_models = load_approach_models(encoder_config)

model_log_name = args.results_path + encoder_config['log_name'] + '_Task_'

# Define the neural network hyperparameters
probe = PROBES[args.probe]

for task_name in args.tasks:
    print('Task:', task_name, flush=True)
    task = RETRIEVAL_TASKS[task_name]
    relation_prediction = task_name == 'relation_prediction'

    # Load the dataset
    dataset_train, dataset_test = load_dataset(task)

    # Create the input and target
    X_train = create_input(dataset_train, task['inputs'], approach_models, snomed)
    X_test = create_input(dataset_test, task['inputs'], approach_models, snomed)

    y_train_ids = list(dataset_train[task['target']])
    y_test_ids = list(dataset_test[task['target']])

    y_train = np.array([get_embedding(approach_models, sct_id, snomed, is_relation=relation_prediction) for sct_id in y_train_ids], dtype=np.float32)
    y_test = np.array([get_embedding(approach_models, sct_id, snomed, is_relation=relation_prediction) for sct_id in y_test_ids], dtype=np.float32)

    print('Input data ready', flush=True)

    # Obtain the target space
    target_space = {}

    if relation_prediction:
        for relation_id in dataset_train['relation_id'].unique():
            target_space[relation_id] = get_embedding(approach_models, relation_id, snomed, is_relation=True)
    else:
        for concept_id in snomed.get_sct_concepts(metadata=False):
            target_space[concept_id] = get_embedding(approach_models, concept_id, snomed)

    embedding_size = len(list(target_space.values())[0])

    log_output_file_name = model_log_name + task['log_name'] + '_nn_results.csv'

    for iteration in range(args.iterations):
        print('Iteration', iteration, 'for task', task_name, flush=True)

        # We will only need to write the header if the file did not exist
        header_needed = not os.path.isfile(log_output_file_name)

        with open(log_output_file_name, 'a') as log_output:
            if header_needed:
                log_output.write(LOG_HEADER)

            # Create the NN
            nn = EmbeddingPredictor(target_space, probe['batch_size'], probe['epochs'], probe['dense_layers'], probe['dropout_layers'],
                                    probe['activation_function'], probe['kernel_initializer'], probe['batch_normalization'],
                                    probe['regularization_type'], probe['l1_regularization'], probe['l2_regularization'],
                                    probe['optimizer_function'], probe['learning_rate'],
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
            train_accuracy, train_top3, train_top5, train_top10 = compute_hits(train_predictions, y_train_ids)
            accuracy, top3, top5, top10 = compute_hits(predictions, y_test_ids)

            # Write to log
            log_output.write(str(probe['batch_size']) + ';' + str(probe['epochs']) + ';' + str(probe['dense_layers']) + ';' + str(probe['activation_function']) + ';' + str(probe['kernel_initializer']) + ';' + str(probe['dropout_layers']) + ';' + str(probe['batch_normalization']) + ';' + str(probe['regularization_type']) + ';' + str(probe['optimizer_function']) + ';' + str(probe['learning_rate']) + ';' + str(loss) + ';' + str(train_accuracy) + ';' + str(train_top3) + ';' + str(train_top5) + ';' + str(train_top10) + ';' + str(val_loss) + ';' + str(accuracy) + ';' + str(top3) + ';' + str(top5) + ';' + str(top10) + '\n')
            log_output.flush()
            os.fsync(log_output.fileno())

        # Free memory before the next iteration
        del nn
        K.clear_session()

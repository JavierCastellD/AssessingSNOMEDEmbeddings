import argparse
import os
import os.path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K

from python_libraries.nn_embedding_classifier import EmbeddingClassifier
from python_libraries.probing.config import ENCODERS, PROBES
from python_libraries.probing.tasks import CLASSIFICATION_TASKS, create_input, load_dataset, load_approach_models
from python_libraries.snomed import Snomed

# The current tasks that we are considering that require using classification are:
# 1. Is child of, i.e., given two concepts, whether the first one is a direct child of the second one
# 2. Is there relation, i.e., given two concepts, whether there is any kind of non-is-a relationship between them
# 3. Semantic type classification, i.e., given a concept, identify its semantic type
# 4. Top level hierarchy classification, i.e., given a concept, identify the hierarchy to which it belongs
# 5. Triple classification, i.e., given a triple <subject, relation, object>, whether it is valid or not
#
# Usage:
#   python src/classification_tasks_nn.py --encoder <approach> [--tasks <tasks>] [--probe main|simple] [--iterations N]
# Example:
#   python src/classification_tasks_nn.py --encoder gnn_sim --tasks semantic_type top_level_hierarchy --iterations 10

# Paths to SNOMED CT
CONCEPTS_PATH = "snomed_data/conceptInternational_20240101.txt"
RELATIONS_PATH = "snomed_data/relationshipInternational_20240101.txt"
DESCRIPTIONS_PATH = "snomed_data/descriptionInternational_20240101.txt"

LOG_HEADER = 'Batch size;Epochs;Dense layers;Activation function;Kernel initializer;Dropout layers;Batch normalization;Regularization type;Optimizer function;Learning rate;Accuracy;Loss;Val_accuracy;Val_loss;Test_accuracy;Test_macro_F1;Test_balanced_accuracy\n'

# Read the arguments
parser = argparse.ArgumentParser(description='Probing of SNOMED CT embeddings on the classification tasks.')
parser.add_argument('--encoder', required=True, choices=ENCODERS.keys(), help='Embedding approach to evaluate.')
parser.add_argument('--tasks', nargs='+', default=list(CLASSIFICATION_TASKS.keys()), choices=CLASSIFICATION_TASKS.keys(), help='Tasks to evaluate.')
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
    task = CLASSIFICATION_TASKS[task_name]
    target_column = task['target']
    n_embeddings = len(task['inputs'])

    # Load the dataset
    dataset_train, dataset_test = load_dataset(task)

    # Transform the dataset into the input data
    ohe = OneHotEncoder(sparse_output=False)
    y_train = ohe.fit_transform(dataset_train[target_column].to_numpy().reshape(-1, 1))
    y_test = ohe.transform(dataset_test[target_column].to_numpy().reshape(-1, 1))

    X_train = create_input(dataset_train, task['inputs'], approach_models, snomed)
    X_test = create_input(dataset_test, task['inputs'], approach_models, snomed)

    print('Input data ready', flush=True)

    embedding_size = X_train.shape[1] // n_embeddings
    number_of_classes = y_train.shape[1]

    log_output_file_name = model_log_name + target_column + '_nn_results.csv'

    for iteration in range(args.iterations):
        print('Iteration', iteration, 'for task', target_column, flush=True)

        # We will only need to write the header if the file did not exist
        header_needed = not os.path.isfile(log_output_file_name)

        with open(log_output_file_name, 'a') as log_output:
            if header_needed:
                log_output.write(LOG_HEADER)

            # Create the NN
            nn = EmbeddingClassifier(number_of_classes, n_embeddings, probe['batch_size'], probe['epochs'], probe['dense_layers'],
                                     probe['dropout_layers'], probe['activation_function'], probe['kernel_initializer'],
                                     probe['batch_normalization'], probe['regularization_type'], probe['l1_regularization'],
                                     probe['l2_regularization'], probe['optimizer_function'], probe['learning_rate'],
                                     embedding_size=embedding_size)

            print(nn.model.summary())

            # Train the model
            history = nn.train_model(X_train, y_train, X_test, y_test, verbose=2)

            # Transform it into a dataframe
            results = pd.DataFrame(history.history)

            # Obtain the metrics
            accuracy = round(results.categorical_accuracy.values[-1:][0] * 100, 2)
            val_accuracy = round(results.val_categorical_accuracy.values[-1:][0] * 100, 2)
            loss = round(results.loss.values[-1:][0], 4)
            val_loss = round(results.val_loss.values[-1:][0], 4)

            # Obtain the test metrics from the predictions
            y_true = np.argmax(y_test, axis=1)
            y_pred = np.argmax(nn.predict(X_test, verbose=0), axis=1)
            all_labels = list(range(number_of_classes))

            test_accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)
            test_macro_f1 = round(f1_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0) * 100, 2)
            test_balanced_accuracy = round(balanced_accuracy_score(y_true, y_pred) * 100, 2)

            # Write to log
            log_output.write(str(probe['batch_size']) + ';' + str(probe['epochs']) + ';' + str(probe['dense_layers']) + ';' + str(probe['activation_function']) + ';' + str(probe['kernel_initializer']) + ';' + str(probe['dropout_layers']) + ';' + str(probe['batch_normalization']) + ';' + str(probe['regularization_type']) + ';' + str(probe['optimizer_function']) + ';' + str(probe['learning_rate']) + ';' + str(accuracy) + ';' + str(loss) + ';' + str(val_accuracy) + ';' + str(val_loss) + ';' + str(test_accuracy) + ';' + str(test_macro_f1) + ';' + str(test_balanced_accuracy) + '\n')
            log_output.flush()
            os.fsync(log_output.fileno())

        # Free memory before the next iteration
        del nn
        K.clear_session()

from nn_embedding_predictor import EmbeddingPredictor
from embedding_models.fasttext_EM import FastTextEM
from snomed import Snomed
import pandas as pd

# If the task is relation prediction, it should be set to true
# If the task is analogy prediction, it should be set to false
relation_prediction = True

# Load SNOMED
# TODO: Creo que solo haría falta para cuando estemos con analogy prediction
con_path = "snomed_data/conceptInternational_20221031.txt"
rel_path = "snomed_data/relationshipInternational_20221031.txt"
desc_path = "snomed_data/descriptionInternational_20221031.txt"
snomed = Snomed(con_path, rel_path, desc_path)

# Load the model 
# TODO: Esto será cargar el modelo de embeddings
embedding_model = FastTextEM(model_path="TODO")

# Load the dataset
dataset_train = pd.read_csv('datasets/datasets_relation_prediction/dataset_relation_prediction_train.csv')
dataset_dev = pd.read_csv('datasets/datasets_relation_prediction/dataset_relation_prediction_dev.csv')

# Obtain the target space
target_space = {}

for target_id in dataset_train['relation_id'].unique():
    target_space[target_id] = None # TODO: Habría que asignar el embedding aquí
    # TODO: Para las relaciones generamos el embedding a partir del ID o a partir del nombre? -> Probamos ambos y vemos

# Define the neural network hyperparameters
batch_size = 256
epochs = 100
dense_layers = [100, 75, 25]
activation_function = 'relu'
kernel_initializer = 'glorot_uniform'
dropout_layers = None
batch_normalization = None
regularization_type = None
l1_regularization = 0.01
l2_regularization = 0.01


# Create the NN
nn = EmbeddingPredictor(None, batch_size, epochs, dense_layers, dropout_layers, activation_function, 
                        kernel_initializer, batch_normalization, regularization_type, l1_regularization, 
                        l2_regularization, relation_prediction=relation_prediction)
import numpy as np
import pandas as pd

from python_libraries.snomed import Snomed

DATASETS_PATH = 'datasets/'

# The current tasks that we are considering that require using classification are:
# 1. Is child of, i.e., given two concepts, whether the first one is a direct child of the second one
# 2. Is there relation, i.e., given two concepts, whether there is any kind of non-is-a relationship between them
# 3. Semantic type classification, i.e., given a concept, identify its semantic type
# 4. Top level hierarchy classification, i.e., given a concept, identify the hierarchy to which it belongs
# 5. Triple classification, i.e., given a triple <subject, relation, object>, whether it is valid or not
#
# For each task, 'inputs' are the columns whose embeddings are concatenated, in that order, to create the input
# of the probe. The 'relation_id' column is embedded as a relationship, and the rest as concepts.
CLASSIFICATION_TASKS = {
    'is_child_of' : {
        'dataset' : 'datasets_is_son_of/dataset_is_son_of_v2',
        'target' : 'is_son_of',
        'inputs' : ['subject_id', 'object_id']
    },
    'is_there_relation' : {
        'dataset' : 'datasets_is_there_relation/dataset_is_there_relation_v2',
        'target' : 'relation_status',
        'inputs' : ['subject_id', 'object_id']
    },
    'semantic_type' : {
        'dataset' : 'datasets_semantic_type/dataset_semantic_type',
        'target' : 'semantic_type',
        'inputs' : ['concept_id']
    },
    'top_level_hierarchy' : {
        'dataset' : 'datasets_top_level_hierarchy/dataset_top_level_hierarchy',
        'target' : 'top_level',
        'inputs' : ['concept_id']
    },
    'is_triple_correct' : {
        'dataset' : 'datasets_triple_classification/dataset_triple_classification_v2',
        'target' : 'triple_status',
        'inputs' : ['subject_id', 'relation_id', 'object_id']
    }
}

# The current tasks that we are considering that require retrieval are:
# 1. Relation prediction, i.e., given a subject and object, predict which relation exists between them
# 2. Object prediction, i.e., given a subject and a relationship, predict the object of that triple
#
# The probe predicts the embedding of the 'target' column, which is then used to retrieve the most similar
# relationships or concepts. 'log_name' is the name of the task used in the results files.
RETRIEVAL_TASKS = {
    'relation_prediction' : {
        'dataset' : 'datasets_relation_prediction/dataset_relation_prediction',
        'target' : 'relation_id',
        'inputs' : ['subject_id', 'object_id'],
        'log_name' : 'relation_prediction'
    },
    'object_prediction' : {
        'dataset' : 'datasets_analogy_prediction/dataset_analogy_prediction',
        'target' : 'object_id',
        'inputs' : ['subject_id', 'relation_id'],
        'log_name' : 'analogy_prediction'
    }
}


def load_dataset(task : dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Function to load the dataset of a task. The training and development partitions are used to train the
    probe, while the test partition is used to evaluate it.

    Parameters:
        task (dict):
            Definition of the task, as in CLASSIFICATION_TASKS or RETRIEVAL_TASKS.

    Returns:
        A tuple with the training and test datasets.
    '''
    dataset_path = DATASETS_PATH + task['dataset']

    dataset_train = pd.read_csv(dataset_path + '_train.csv')
    dataset_train_2 = pd.read_csv(dataset_path + '_dev.csv')
    dataset_train = pd.concat([dataset_train, dataset_train_2], ignore_index=True)

    dataset_test = pd.read_csv(dataset_path + '_test.csv')

    return dataset_train, dataset_test


def load_approach_models(encoder_config : dict) -> dict:
    '''Function to load the models used to obtain the embeddings of the concepts and relationships of an
    approach. If both use the same model, it is only loaded once.
    
    Parameters:
        encoder_config (dict):
            Configuration of the approach, as defined in config.ENCODERS. Each model is defined as (type, path), 
            where type can be 'fasttext', 'sentence_transformer', 'dictionary' or 'random'. For 'random', 
            the path is the dimension of the embeddings.
    
    Returns:
        A dictionary with the keys 'concepts' and 'relations', whose values are tuples (EmbeddingModel, text).
    '''
    approach_models = {}
    loaded_models = {}

    for key in ['concepts', 'relations']:
        model, text = encoder_config[key]

        # Only load the model if it has not been loaded yet
        if model not in loaded_models:
            model_type, model_path = model

            if model_type == 'fasttext':
                from python_libraries.embedding_models.fasttext_EM import FastTextEM
                loaded_models[model] = FastTextEM(model_path=model_path)
            elif model_type == 'sentence_transformer':
                from python_libraries.embedding_models.sentencetransformer_EM import SentenceTransformerEM
                loaded_models[model] = SentenceTransformerEM(model_path=model_path)
            elif model_type == 'dictionary':
                from python_libraries.embedding_models.dictionary_EM import DictionaryEM
                loaded_models[model] = DictionaryEM(model_path=model_path)
            elif model_type == 'random':
                from python_libraries.embedding_models.random_EM import RandomEM
                loaded_models[model] = RandomEM(vector_size=model_path)
            else:
                raise ValueError('Unknown model type: ' + str(model_type))

        approach_models[key] = (loaded_models[model], text)

    return approach_models


def get_embedding(approach_models : dict, sct_id : int, snomed : Snomed, is_relation : bool = False) -> np.ndarray:
    '''Function to obtain the embedding of a SNOMED CT concept or relationship.
    
    Parameters:
        approach_models (dict):
            Models used to obtain the embeddings, as returned by load_approach_models.
        sct_id (int):
            Identifier of the concept or relationship in SNOMED CT.
        snomed (Snomed):
            SNOMED CT, used to obtain the descriptions and FSNs.
        is_relation (bool):
            Whether sct_id is a relationship, so that the relations model is used.
    
    Returns:
        A numpy array with the embedding.
    '''
    embedding_model, text = approach_models['relations' if is_relation else 'concepts']

    if text == 'descriptions':
        descriptions = snomed.get_descriptions(sct_id)
        embedding = embedding_model.get_embedding_from_list(descriptions)
    elif text == 'fsn':
        fsn = snomed.get_fsn(sct_id)
        embedding = embedding_model.get_embedding(fsn)
    elif text == 'id':
        embedding = embedding_model.get_embedding(str(sct_id))
    else:
        raise ValueError('Unknown text type: ' + str(text))

    return np.array(embedding)


def create_input(dataset : pd.DataFrame, columns : list[str], approach_models : dict, snomed : Snomed) -> np.ndarray:
    '''Function to create the input of the probe, by concatenating the embeddings of the given columns 
    for each sample of the dataset. The 'relation_id' column is embedded as a relationship, and the rest 
    as concepts.
    
    Parameters:
        dataset (DataFrame):
            Dataset of the task.
        columns (list):
            Columns whose embeddings are concatenated, in that order.
        approach_models (dict):
            Models used to obtain the embeddings, as returned by load_approach_models.
        snomed (Snomed):
            SNOMED CT, used to obtain the descriptions and FSNs.
    
    Returns:
        A numpy array with one row per sample.
    '''
    X = []
    for sample in zip(*[dataset[column] for column in columns]):
        embeddings = []
        for column, sct_id in zip(columns, sample):
            embeddings.append(get_embedding(approach_models, sct_id, snomed, is_relation=(column == 'relation_id')))
        X.append(np.concatenate(embeddings))

    return np.array(X, dtype=np.float32)

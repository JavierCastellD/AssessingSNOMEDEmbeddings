# Configuration of the embedding approaches and probes used in the probing experiments.
#
# Each approach defines:
#   - log_name: name used in the results files
#   - concepts: how the concept embeddings are obtained, as (model, text)
#   - relations: how the relationship embeddings are obtained, as (model, text)
#
# The model is defined as (type, path), where type can be 'fasttext', 'sentence_transformer', 'dictionary'
# (embeddings previously generated and saved as an npz file) or 'random' (in this case, the path is the
# dimension of the embeddings).
#
# The text is what is given to the model to obtain the embedding:
#   - 'descriptions': all the descriptions of the concept, whose embeddings are averaged
#   - 'fsn': the FSN of the concept or relationship
#   - 'id': the SNOMED CT identifier (used for the dictionaries and FastText trained on SNOMED CT walks)

MODELS_PATH = 'models/'
DICTIONARIES_PATH = 'concepts_dictionaries_2024/'

RANDOM_MODEL = ('random', 200)
BASE_ST_MODEL = ('sentence_transformer', 'sentence-transformers/all-MiniLM-L6-v2')
FINE_TUNED_ST_MODEL = ('sentence_transformer', MODELS_PATH + 'full_st_minilm_3_neg_1_ep_sct_train.model')
WALKS_MODEL = ('fasttext', MODELS_PATH + 'ft_train_dev_20240101_1_1.model')

ENCODERS = {
    # Random embeddings baseline
    'random' : {
        'log_name' : 'Random_EM',
        'concepts' : (RANDOM_MODEL, 'fsn'),
        'relations' : (RANDOM_MODEL, 'fsn')
    },

    # FastText trained on MIMIC-IV
    'mimic_10' : {
        'log_name' : 'MIMIC_10',
        'concepts' : (('fasttext', MODELS_PATH + 'ft_train_MIMIC_10.model'), 'descriptions'),
        'relations' : (('fasttext', MODELS_PATH + 'ft_train_MIMIC_10.model'), 'fsn')
    },
    'mimic_25' : {
        'log_name' : 'MIMIC_25',
        'concepts' : (('fasttext', MODELS_PATH + 'ft_train_MIMIC_25.model'), 'descriptions'),
        'relations' : (('fasttext', MODELS_PATH + 'ft_train_MIMIC_25.model'), 'fsn')
    },
    'mimic_50' : {
        'log_name' : 'MIMIC_50',
        'concepts' : (('fasttext', MODELS_PATH + 'ft_train_MIMIC_50.model'), 'descriptions'),
        'relations' : (('fasttext', MODELS_PATH + 'ft_train_MIMIC_50.model'), 'fsn')
    },
    'mimic_75' : {
        'log_name' : 'MIMIC_75',
        'concepts' : (('fasttext', MODELS_PATH + 'ft_train_MIMIC_75.model'), 'descriptions'),
        'relations' : (('fasttext', MODELS_PATH + 'ft_train_MIMIC_75.model'), 'fsn')
    },
    'mimic_100' : {
        'log_name' : 'MIMIC_100',
        'concepts' : (('fasttext', MODELS_PATH + 'ft_train_MIMIC_100.model'), 'descriptions'),
        'relations' : (('fasttext', MODELS_PATH + 'ft_train_MIMIC_100.model'), 'fsn')
    },

    # FastText trained on SNOMED CT walks
    'walks' : {
        'log_name' : 'FT_walks',
        'concepts' : (WALKS_MODEL, 'descriptions'),
        'relations' : (WALKS_MODEL, 'id')
    },

    # Sentence transformers
    'base_st' : {
        'log_name' : 'Base_SBERT',
        'concepts' : (('dictionary', DICTIONARIES_PATH + 'full_base_mini_lm_sct_dict.npz'), 'id'),
        'relations' : (BASE_ST_MODEL, 'fsn')
    },
    'fine_tuned_st' : {
        'log_name' : 'SBERT',
        'concepts' : (('dictionary', DICTIONARIES_PATH + 'full_train_mini_lm_3_1_sct_dict.npz'), 'id'),
        'relations' : (FINE_TUNED_ST_MODEL, 'fsn')
    },

    # Graph neural networks, trained without the outgoing relationships of the test concepts
    'gnn_mul' : {
        'log_name' : 'GNN_mul',
        'concepts' : (('dictionary', DICTIONARIES_PATH + 'train_gnn_mul_10_ep_no_test_edges_sct_dict.npz'), 'id'),
        'relations' : (FINE_TUNED_ST_MODEL, 'fsn')
    },
    'gnn_sim' : {
        'log_name' : 'GNN_sim',
        'concepts' : (('dictionary', DICTIONARIES_PATH + 'train_gnn_sim_10_ep_no_test_edges_sct_dict.npz'), 'id'),
        'relations' : (FINE_TUNED_ST_MODEL, 'fsn')
    },
    'gnn_walks' : {
        'log_name' : 'GNN_walks',
        'concepts' : (('dictionary', DICTIONARIES_PATH + 'train_gnn_walks_10_ep_no_test_edges_sct_dict.npz'), 'id'),
        'relations' : (WALKS_MODEL, 'id')
    },

    # LLM2Vec
    'mistral' : {
        'log_name' : 'LLM_mistral',
        'concepts' : (('dictionary', DICTIONARIES_PATH + 'llm2vec_mistral7B_sct_dict.npz'), 'id'),
        'relations' : (('dictionary', DICTIONARIES_PATH + 'llm2vec_mistral7B_sct_metadata_dict.npz'), 'id')
    },
    'llama' : {
        'log_name' : 'LLM_llama3',
        'concepts' : (('dictionary', DICTIONARIES_PATH + 'llm2vec_llama3_sct_dict.npz'), 'id'),
        'relations' : (('dictionary', DICTIONARIES_PATH + 'llm2vec_llama3_sct_metadata_dict.npz'), 'id')
    },

    # Knowledge graph embeddings, trained without the triples whose subject is a test concept
    'transe' : {
        'log_name' : 'TransE',
        'concepts' : (('dictionary', DICTIONARIES_PATH + 'kge_transe_200_no_test_edges_sct_dict.npz'), 'id'),
        'relations' : (('dictionary', DICTIONARIES_PATH + 'kge_transe_200_no_test_edges_sct_metadata_dict.npz'), 'id')
    },
    'distmult' : {
        'log_name' : 'DistMult',
        'concepts' : (('dictionary', DICTIONARIES_PATH + 'kge_distmult_200_no_test_edges_sct_dict.npz'), 'id'),
        'relations' : (('dictionary', DICTIONARIES_PATH + 'kge_distmult_200_no_test_edges_sct_metadata_dict.npz'), 'id')
    }
}

# Hyperparameters of the probes
PROBES = {
    # Probe used in the main experiments
    'main' : {
        'batch_size' : 1024,
        'epochs' : 50,
        'dense_layers' : [2000, 2000, 2000, 2000],
        'activation_function' : 'relu',
        'kernel_initializer' : 'glorot_uniform',
        'dropout_layers' : None,
        'batch_normalization' : None,
        'regularization_type' : None,
        'l1_regularization' : 0.01,
        'l2_regularization' : 0.01,
        'optimizer_function' : 'adamax',
        'learning_rate' : 0.001
    },

    # Simpler probe, used to assess the effect of the probe capacity
    'simple' : {
        'batch_size' : 512,
        'epochs' : 50,
        'dense_layers' : [300, 200, 100],
        'activation_function' : 'relu',
        'kernel_initializer' : 'glorot_uniform',
        'dropout_layers' : None,
        'batch_normalization' : None,
        'regularization_type' : None,
        'l1_regularization' : 0.01,
        'l2_regularization' : 0.01,
        'optimizer_function' : 'adam',
        'learning_rate' : 0.001
    }
}

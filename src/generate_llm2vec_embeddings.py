import sys
import time

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from llm2vec import LLM2Vec

from python_libraries.snomed import Snomed
from python_libraries.embedding_models.embedding_model import save_embeddings

# Generation of SNOMED CT embeddings with LLM2Vec, which converts decoder-only LLMs into text encoders.
# The embedding of each concept is obtained from its FSN, together with an instruction prompt. Two
# dictionaries are generated: one for the concepts and one for the metadata concepts, which include
# the relationships of SNOMED CT.
#
# Usage:
#   python src/generate_llm2vec_embeddings.py <model>
# where model is either 'mistral' (Mistral 7B) or 'llama' (Llama 3).

MODELS = {
    'mistral' : ('McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp', 'llm2vec_mistral7B'),
    'llama' : ('McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp', 'llm2vec_llama3')
}

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else 'mistral'
MODEL_NAME, OUTPUT_PREFIX = MODELS[MODEL_KEY]

CONCEPTS_OUTPUT_PATH = OUTPUT_PREFIX + '_sct_dict.npz'
METADATA_OUTPUT_PATH = OUTPUT_PREFIX + '_sct_metadata_dict.npz'

CONCEPTS_PATH = 'snomed_data/conceptInternational_20240101.txt'
RELATIONS_PATH = 'snomed_data/relationshipInternational_20240101.txt'
DESCRIPTIONS_PATH = 'snomed_data/descriptionInternational_20240101.txt'

INSTRUCTION = (
    "Given the full name of a SNOMED CT concept, find relevant features such as semantic type and relationships to other SNOMED CT concepts:"
)

def encode_concepts(l2v : LLM2Vec, snomed : Snomed, concepts : list[int]) -> dict:
    '''Function to obtain the LLM2Vec embedding of each SNOMED CT concept from its FSN.
    
    Parameters:
        l2v (LLM2Vec):
            LLM2Vec model used to encode the FSNs.
        snomed (Snomed):
            SNOMED CT, used to obtain the FSNs.
        concepts (list):
            Identifiers of the SNOMED CT concepts to encode.
    
    Returns:
        A dictionary that links the identifier of each concept, as a string, to its embedding.
    '''
    # Each query is made up of the instruction and the FSN of the concept
    queries = [[INSTRUCTION, snomed.get_fsn(concept_id)] for concept_id in concepts]

    print('Encoding queries')
    t4 = time.time()
    q_reps = l2v.encode(queries)
    print('Encode queries time:', time.time() - t4)

    print('Number of embeddings:', len(q_reps))
    print('Embedding length:', len(q_reps[0]))

    return {str(concept_id): q_reps[index].tolist() for index, concept_id in enumerate(concepts)}

# Load SNOMED CT
print('Loading SNOMED')
snomed = Snomed(CONCEPTS_PATH, RELATIONS_PATH, DESCRIPTIONS_PATH)

# Loading base model, along with custom code that enables bidirectional connections in decoder-only LLMs
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
config = AutoConfig.from_pretrained(
    MODEL_NAME, trust_remote_code=True
)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)

# Loading MNTP (Masked Next Token Prediction) model
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME
)

# Wrapper for encoding and pooling operations
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

# Generate and save the embeddings of the concepts
concept_dict = encode_concepts(l2v, snomed, snomed.get_sct_concepts(metadata=False))
save_embeddings(embedding_dictionary=concept_dict, filename=CONCEPTS_OUTPUT_PATH)

# Generate and save the embeddings of the metadata concepts, which include the relationships
metadata_dict = encode_concepts(l2v, snomed, snomed.get_sct_concepts(metadata=True, concepts=False))
save_embeddings(embedding_dictionary=metadata_dict, filename=METADATA_OUTPUT_PATH)

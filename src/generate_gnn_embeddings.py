import random

import json
import torch
from torch_geometric.nn import SAGEConv, to_hetero
from tqdm import tqdm

from python_libraries.embedding_models.sentencetransformer_EM import SentenceTransformerEM
from python_libraries.snomed import Snomed, IS_A_ID
from python_libraries.snomed_gnn_data import preprocess_fsn, SnomedData

DICTIONARY_PATH = 'train_gnn_sim_10_ep_sct_dict.json'
EMBEDDING_MODEL_PATH = 'models/full_st_minilm_3_neg_1_ep_sct_train.model'
EMBEDDING_SIZE = 384

N_NEGATIVE_SAMPLES = 3
SIM_VALUE_IS_A = 0.75
SIM_VALUE_RELATED = 0.5
SIM_VALUE_NOT_RELATED = 0

EPOCHS = 10

HIDDEN_CHANNELS = 128
EMBEDDING_OUTPUT_SIZE = EMBEDDING_SIZE

class GNNModel(torch.nn.Module):
    """Class that represents a simple Graph Neural Network model."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Parameters:
            in_channels (int):
                Size of input sample.
            hidden_channels (int):
                Size of hidden channels.
            out_channels (int):
                Size of output sample.
        """
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

def decode(z, gnn_train_data : list[str, int, str, int], use_cosine_sim : bool = True):
    """Method that calculates a similarity value between the embeddings produced during the training of the GNN. It is
    used to calculate the loss. If use_cosine_sim is True, it will use cosine similarity. Otherwise, it will use the sum of
    the product of the embeddings.
    """
    embeddings_A = []
    embeddings_B = []

    for top_A, index_A, top_B, index_B in gnn_train_data:
        emb_A = z[top_A][[index_A]]
        embeddings_A.append(emb_A)
        
        emb_B = z[top_B][[index_B]]
        embeddings_B.append(emb_B)

    if use_cosine_sim:
        sim_cos = torch.nn.CosineSimilarity(dim=1)
        return sim_cos(torch.cat(embeddings_A, 0), torch.cat(embeddings_B, 0))
    else:
        return (torch.cat(embeddings_A, 0) * torch.cat(embeddings_B, 0)).sum(dim=-1)

def generate_train_tuples(snomed : Snomed, snomed_concepts : list[int], snomed_data : SnomedData) -> tuple[list[str, int, str, int], list[float]]:
    """Method to generate the tuples of concepts to train the GNN. Returns a list of tuples that can be used to train the GNN in the concept similarity task, and the
    corresponding similarity values for each sample.

    Parameters:
        snomed (Snomed):
            Snomed CT object.
        snomed_concepts (list):
            List of SNOMED CT identifiers that represent the concepts for which to create the training tuples.
        snomed_data (SnomedData):
            A SnomedData object that contains SNOMED CT information as a HeteroData object.
    
    Returns: 
        A list of tuples (top_level_A, index_A, top_level_B, index_B), where top_level_A, and top_level_B are the names of the top level hierarchies of the concepts,
        and index_A, and index_B are the positions in snomed_data of those concepts.
        A list of float values of same length to the other list that contains the similarity values for each training pair.
    """
    # List of tuples (top_level_concept_A, index_A, top_level_concept_B, index_B)
    gnn_train_data = []

    # Label information
    label = []

    # We iterate through each SNOMED CT concept
    for sct_id in tqdm(snomed_concepts):
        # For each one we obtain the top level concept
        top_level_id_A = snomed.get_top_level_concept(sct_id)
        top_level_A = preprocess_fsn(snomed.get_fsn(top_level_id_A))

        # And the index position in the GNN
        index_A = snomed_data.mappings[top_level_A][sct_id]

        related_objs = []
        # Then, for each of that concept's related concepts
        for rel_id, obj_id in snomed.get_related_concepts(sct_id):
            # We obtain the top level concept
            top_level_id_B = snomed.get_top_level_concept(obj_id)
            top_level_B = preprocess_fsn(snomed.get_fsn(top_level_id_B))

            # Obtain the index position in the GNN
            index_B = snomed_data.mappings[top_level_B][obj_id]

            # And we store it, applying a different similarity value
            # depending on the relation
            gnn_train_data.append((top_level_A, index_A, top_level_B, index_B))

            # If they are part of a child-parent relation, we give a 0.75
            if rel_id == IS_A_ID:
                label.append(SIM_VALUE_IS_A)
            # Otherwise we give them a 0.5
            else:
                label.append(SIM_VALUE_RELATED)

            related_objs.append(obj_id)

        # We also create negative instances by searching for non directly related concepts
        for i in range(N_NEGATIVE_SAMPLES):
            random_sct_id = random.choice(snomed_concepts)
            while(random_sct_id in related_objs): random_sct_id = random.choice(snomed_concepts)

            # We obtain the top level concept
            top_level_id_B = snomed.get_top_level_concept(random_sct_id)
            top_level_B = preprocess_fsn(snomed.get_fsn(top_level_id_B))
            
            # Obtain the index position in the GNN
            index_B = snomed_data.mappings[top_level_B][random_sct_id]

            # And we store it, applying a 0 in similarity value
            gnn_train_data.append((top_level_A, index_A, top_level_B, index_B))
            label.append(SIM_VALUE_NOT_RELATED)

    return gnn_train_data, label

# MAIN PROGRAM

# Load SNOMED CT
con_path = "snomed_data/conceptInternational_20240101.txt"
rel_path = "snomed_data/relationshipInternational_20240101.txt"
desc_path = "snomed_data/descriptionInternational_20240101.txt"
snomed = Snomed(con_path, rel_path, desc_path)

# Load the model
embedding_model = SentenceTransformerEM(model_path=EMBEDDING_MODEL_PATH)

# Transform SNOMED CT into GNN data
snomed_data = SnomedData('snomed_gnn/', snomed, embedding_model)

data = snomed_data.data

# Device to train the GNN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

# Create the model
model = GNNModel(in_channels=EMBEDDING_SIZE, hidden_channels=HIDDEN_CHANNELS, out_channels=EMBEDDING_OUTPUT_SIZE).to(device)

# Transform it into an heterogeneous model
model = to_hetero(model, metadata=data.metadata())

# Optimizer and loss
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

# Obtain the concepts in scope
concepts_in_scope = []
print('Reading file...')
with open('datasets/train_concepts.txt', 'r') as file:
    for line in file.readlines():
        concepts_in_scope.append(int(line))

# Model training
print('Training model...')
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    z = model.forward(data.x_dict, data.edge_index_dict)

    # Generate the positive and negative pairs
    # The positive pairs are always the same, but the negative ones change each iteration
    gnn_data, labels = generate_train_tuples(snomed, concepts_in_scope, snomed_data)

    out = decode(z, gnn_data).view(-1)
    loss = criterion(out, torch.tensor(labels))
    loss.backward()
    optimizer.step()

    print('Epoch:', epoch, 'Loss:', loss)

# Obtain all the embeddings
embeddings = model.forward(data.x_dict, data.edge_index_dict)

# Now generate and store the embeddings
print('Storing embeddings')
concept_dict = {}
for concept_id in tqdm(snomed.get_sct_concepts(metadata=False)):
    # For each one we obtain the top level concept
    top_level_id = snomed.get_top_level_concept(concept_id)
    top_level = preprocess_fsn(snomed.get_fsn(top_level_id))

    # And the index position in the GNN
    index = snomed_data.mappings[top_level][concept_id]

    # Link the key to the embedding
    concept_dict[concept_id] = embeddings[top_level][index].tolist()

# Save the dictionary
dictionary_file = open(DICTIONARY_PATH, 'w')
json.dump(concept_dict, dictionary_file, indent=4)
dictionary_file.close()
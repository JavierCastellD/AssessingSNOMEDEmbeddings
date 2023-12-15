from snomed import Snomed
import random
import math

def snomed_walks(snomed : Snomed, depth : int = 1, concepts : list = None):
    """Method that returns each walk up to a given depth from SNOMED CT. If concepts
    is not None, only those concepts will be considered for the walks.
    A walk is considered as a sequence of concepts_id alternated with relationship_ids,
    while depth indicates how many concepts are in the sequence besides the initial
    concept. For example, a walk of depth 1 would be (sct_idA, rel_id, sct_idB).
    
    Parameters:
        snomed (Snomed):
            Snomed object that contains the information about SNOMED CT.
        depth (int):
            Maximum depth of the walks.
        concepts (list):
            List of integers representing the subset of sct_ids to be considered for the walks.
    Returns:
        A list containing walks, where each walk is a tuple with one or more sct_ids.
    """
    all_walks = set()
    if concepts is not None:
        for sct_id in concepts:
            walks = sct_concept_walk(snomed, sct_id, depth, concepts)

            for walk in walks:
                all_walks.add(walk)
    else:
        for sct_id in snomed.get_sct_concepts(metadata=False):
            walks = sct_concept_walk(snomed, sct_id, depth)

            for walk in walks:
                all_walks.add(walk)

    return list(all_walks)

def sct_concept_walk(snomed : Snomed, initial_concept : int, depth : int = 1, concepts : list = None):
    """Method that returns each possible walk up to a given depth starting from the concept
    initial_concept. If concepts is not None, only those concepts will be considered for the walks.
    A walk is considered as a sequence of concepts_id alternated with relationship_ids. 
    Depth indicates how many concepts are in the sequence besides the initial concept. For example,
    a walk of depth 1 would be (sct_idA, rel_id, sct_idB).
    
    Parameters:
        snomed (Snomed):
            Snomed object that contains the information about SNOMED CT.
        initial concept (int):
            Integer that represents the sct_id of the concept from which to perform the walks.
        depth (int):
            Maximum depth of the walks.
        concepts (list):
            List of integers representing the subset of sct_ids to be considered for the walks.

    Returns:
        A list containing walks, where each walk is a tuple with one or more sct_ids.
    """
    walks = {(initial_concept,)}

    for i in range(depth):
        walks_copy = walks.copy()

        for walk in walks_copy:
            node = walk[-1]
            related_concepts = snomed.get_related_concepts(node)

            if concepts is not None:
                related_concepts = [(rel_id, sct_id) for rel_id, sct_id in related_concepts if sct_id in concepts]

            if len(related_concepts) > 0:
                walks.remove(walk)

            for rel_id, sct_id in related_concepts:
                if i == depth - 1:
                    walks.add(walk + (rel_id, sct_id))
                else:
                    walks.add(walk + (rel_id, sct_id,))

    return list(walks)

def create_walk_corpus(snomed : Snomed, corpus_path : str, concepts : list = None, uri_depth : int = 1, 
                       word_depth : int = 1, do_uri_walks : bool = True, do_word_walks : bool = True, 
                       do_mix_walks : bool = True):
    """Method to create a corpus from walks on SNOMED CT. If a list of concepts IDs is given, only those concepts
    will appear on the walks. The created corpus can contain up to three different type of sentences: URI sentences,
    word sentences, and mix sentences. URI sentences contains only IDs/URIs of the concepts, word sentences contains
    each description/name of each concept, and mix sentences are word sentences but one randomly chosen concept in the
    walk uses its ID instead of its name.

    Parameters:
        snomed (Snomed):
            Snomed object that contains the information needed about SNOMED CT.
        corpus_path (str):
            Path to save the created corpus file.
        concepts (list):
            List of integers that represents the IDs of the concepts to be used in the walks.
        uri_depth (int):
            Depth of the walks for the URI sentences.
        word_depth (int):
            Depth of the walks for the word/mix sentences.
        do_uri_walks (bool):
            Whether to include URI sentences in the corpus. 
        do_word_walks (bool):
            Whether to include word sentences in the corpus.
        do_mix_walks (bool):
            Whether to include mix sentences in the corpus.
    """
    # Perform the walks, filtering the concepts to be used
    if do_uri_walks:
        walks_uris = snomed_walks(snomed, uri_depth, concepts)
    if do_word_walks or do_mix_walks:
        walks_words = snomed_walks(snomed, word_depth, concepts)

    # Create the URI sentences, word sentences, and mix sentences (we do so in the same corpus)
    with open(corpus_path, 'w', encoding='utf8') as corpus_file:
        # Write the sentences using only URIs/IDs
        if do_uri_walks:
            for walk_uri in walks_uris:
                if len(walk_uri) > 1:
                    line = ''

                    for uri in walk_uri:
                        line += str(uri) + ' '

                    corpus_file.write(line + '\n')

        # We then obtain the sentences of words, where relationships are represented as their IDs
        # We also obtain the sentences of words mixed with URIs
        if do_word_walks or do_mix_walks:
            for walk_word in walks_words:
                # Number of concepts besides the initial concept
                n_concepts = math.floor(len(walk_word) / 2)
                
                # Each random walk will contain at least one concept
                descriptions = snomed.get_descriptions(walk_word[0])
                sentences = [[description] for description in descriptions]

                # We create each possible combination of descriptions
                # between the concepts in the walk 
                for n in range(n_concepts):
                    concept_index = 2 * n + 2
                    relation_index = 2 * n + 1

                    descriptions = snomed.get_descriptions(walk_word[concept_index])
                    relation = str(walk_word[relation_index])

                    sentences_aux = []

                    for description in descriptions:
                        for sentence in sentences:
                            sentences_aux.append(sentence + [relation, description])
                    
                    sentences = sentences_aux

                # If there are at least 3 elements in the walk, we can randomly choose and transform
                # one of the concepts into its SCT_ID instead of the word. This way we create both
                # word sentences and mix sentences
                if len(walk_word) > 3 and do_mix_walks:
                    for sentence in sentences:
                        line = ''

                        lineMix = ''
                        concept_replace_uri = random.randint(0, n_concepts) * 2

                        for i, word in enumerate(sentence):
                            line = line + ' ' + word

                            if i == concept_replace_uri:
                                lineMix = lineMix + ' ' + str(walk_word[concept_replace_uri])
                            else:
                                lineMix = lineMix + ' ' + word

                        if do_word_walks:
                            corpus_file.write(line + '\n')
                        corpus_file.write(lineMix + '\n')
                elif do_word_walks:
                    for sentence in sentences:
                        line = ''

                        for i, word in enumerate(sentence):
                            line = line + ' ' + word

                        corpus_file.write(line + '\n')
                
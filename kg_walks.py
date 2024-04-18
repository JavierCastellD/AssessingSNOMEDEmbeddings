import math
import random

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS, SKOS

from python_libraries.snomed import Snomed

LABELS_PREDICATES = [RDFS.label, SKOS.altLabel, SKOS.prefLabel, SKOS.hiddenLabel]

def get_value_from_datatype(obj):
    #TODO
    pass

def get_annotations(graph : Graph, subject : URIRef, language : str = "en"):
    annotations = set()
    for label_predicate in LABELS_PREDICATES:
        for obj in graph.objects(subject=subject, predicate=label_predicate):
            if hasattr(obj, "language") and obj.language == language:
                annotations.add(str(obj))
            else:
                print(obj)
    
    return annotations

def neighbouring_nodes(graph : Graph, subject_node : URIRef):
    neighbours = []

    for pred, obj in graph.predicate_objects(subject=subject_node):
        if pred not in LABELS_PREDICATES:
            neighbours.append((pred, obj))
    
    return neighbours

def kg_walks(graph : Graph, depth : int = 1):
    all_walks = set()
    for subj in graph.subjects():
        walks = subject_walk(graph, subj, depth)

        for walk in walks:
            all_walks.add(walk)
    
    return list(all_walks)


def subject_walk(graph : Graph, initial_node : URIRef, depth : int = 1):
    walks = {(initial_node,)}

    for i in range(depth):
        walks_copy = walks.copy()

        for walk in walks_copy:
            node = walk[-1]
            neighbours = neighbouring_nodes(graph, node)

            """
            if concepts is not None:
                neighbours = [(rel_id, sct_id) for rel_id, sct_id in neighbours if sct_id in concepts]
            """

            if len(neighbours) > 0:
                walks.remove(walk)

            for pred, obj in neighbours:
                if i == depth - 1:
                    walks.add(walk + (pred, obj))
                else:
                    walks.add(walk + (pred, obj,))

    return list(walks)

def create_walk_corpus(graph : Graph, corpus_path : str, iri_depth : int = 1, language : str = "en",
                       label_depth : int = 1, do_iri_walks : bool = True, 
                       do_label_walks : bool = True, do_mix_walks : bool = True):
    # Perform the walks, filtering the concepts to be used
    if do_iri_walks:
        walks_iris = kg_walks(graph, iri_depth)
    if do_label_walks or do_mix_walks:
        walks_label = kg_walks(graph, label_depth)

    # Create the IRI sentences, word sentences, and mix sentences (we do so in the same corpus)
    with open(corpus_path, 'w', encoding='utf8') as corpus_file:
        # Write the sentences using only URIs/IDs
        if do_iri_walks:
            for walk_iri in walks_iris:
                if len(walk_iri) > 1:
                    line = ''

                    for iri in walk_iri:
                        line += str(iri) + ' '

                    corpus_file.write(line + '\n')

        # We then obtain the sentences of words, where relationships are represented as their IDs
        # We also obtain the sentences of words mixed with URIs
        if do_label_walks or do_mix_walks:

            for walk_word in walks_label:
                # Number of concepts besides the initial concept
                n_concepts = math.floor(len(walk_word) / 2)
                
                # Each random walk will contain at least one concept
                descriptions = get_annotations(graph, walk_word[0], language) if len(get_annotations(graph, walk_word[0], language)) > 0 else str(walk_word[0]).split('/')[-1]
                sentences = [[description] for description in descriptions]

                # We create each possible combination of descriptions
                # between the concepts in the walk 
                for n in range(n_concepts):
                    concept_index = 2 * n + 2
                    relation_index = 2 * n + 1

                    if walk_word[concept_index].__class_ is URIRef:
                        descriptions = get_annotations(graph, walk_word[concept_index], language) if len(get_annotations(graph, walk_word[concept_index], language)) > 0 else str(walk_word[concept_index]).split('/')[-1]
                    else:
                        descriptions = []

                    relation = get_annotations(graph, walk_word[relation_index], language)[0] if len(get_annotations(graph, walk_word[relation_index], language)) > 0 else str(walk_word[relation_index]).split('/')[-1]

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
                        concept_replace_iri = random.randint(0, n_concepts) * 2

                        for i, word in enumerate(sentence):
                            line = line + ' ' + word

                            if i == concept_replace_iri:
                                lineMix = lineMix + ' ' + str(walk_word[concept_replace_iri])
                            else:
                                lineMix = lineMix + ' ' + word

                        if do_label_walks:
                            corpus_file.write(line + '\n')
                        corpus_file.write(lineMix + '\n')
                elif do_label_walks:
                    for sentence in sentences:
                        line = ''

                        for i, word in enumerate(sentence):
                            line = line + ' ' + word

                        corpus_file.write(line + '\n')

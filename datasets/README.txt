### INFORMATION ABOUT THE DATASETS ###
# SEMANTIC TYPE CLASSIFICATION
- Dataset about predicting which of the semantic types the concept is related to.
- There are 20 different semantic types.
- It is unbalanced towards disorder, procedure, finding, and body structure.

# TOP LEVEL HIERARCHY CLASSIFICATION
- Dataset about predicting which of the top level hierarchies the concept is related to.
- There are 11 different top level hierarchies.
- It is unbalanced towards clinical finding and procedure.

# RELATION PREDICTION
- Dataset about predicting the possible relationship between two concepts.
- There are 50 possible relationships. We only kept those relationships with a frequency above 0.2% in SNOMED CT.
- The dataset is unbalanced.
- The IS_A relation is not part of the dataset.

# IS THERE RELATION CLASSIFICATION
- Whether there is a relation between two concepts of SNOMED CT.
- For each positive instance of a relation, there is also a negative instance for the same subject concept.
- The possible range of negative instances depends on the domain:
    · If the domain is train, the range is train.
    · If the domain is dev, the range is train + dev.
    · If the domain is test, the range is train + dev + test.

# IS THERE RELATION CLASSIFICATION v2
- Similar to IS THERE RELATION CLASSIFICATION, except that:
    · Negative samples concepts are of similar semantic type to the positive instance, so that false triples comply with SNOMED CT logical model for the semantic type.

# IS SON OF CLASSIFICATION
- Whether one concept is the son of the other concept.
- For each positive instance of a relation, there is also a negative instance for the same subject concept.
- The possible range of negative instances depends on the domain:
    · If the domain is train, the range is train.
    · If the domain is dev, the range is train + dev.
    · If the domain is test, the range is train + dev + test.

# IS SON OF CLASSIFICATION v2
- Similar to IS SON OF CLASSIFICATION, except that:
    · Negative samples concepts are of similar semantic type to the positive instance, so that false triples comply with SNOMED CT logical model for the semantic type.

# ANALOGY PPREDICTION
- Dataset about predicting the possible object concept given the subject concept and the relationship.
- The IS_A relation is not part of the dataset.
- The dataset is the same as RELATION PREDICTION, although the columns are in a different order. Nevertheless, is significantly
  harder to perform analogy prediction (1/300.000) than relation prediction (1/50).

# TRIPLE CLASSIFICATION
- Whether one triple is true or not.
- For each positive instance of a triple, there is also a negative instance for the same subject concept.
- The possible range of negative instances depends on the domain:
    · If the domain is train, the range is train.
    · If the domain is dev, the range is train + dev.
    · If the domain is test, the range is train + dev + test.

# TRIPLE CLASSIFICATION v2
- Similar to TRIPLE CLASSIFICATION, except that:
    · Negative samples concepts are of similar semantic type to the positive instance, so that false triples comply with SNOMED CT logical model for the semantic type.
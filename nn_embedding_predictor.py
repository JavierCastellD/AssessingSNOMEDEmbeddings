import tensorflow as tf
import keras
from keras import regularizers
from keras.layers import InputLayer, Dense, Activation, Dropout, BatchNormalization
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_LOSS_FUNCTION = 'mean_absolute_error'
DEFAULT_OPTIMIZER_FUNCTION = 'adam'

SIMILARITY_INDEX = 0
ID_INDEX = 1

class EmbeddingPredictor():
    """Class that represents a neural network used for outputing embeddings that predict either relationships or concepts.
    The prediction is done by performing multi-dimensional regression, and then using cosine similarity to find the closest
    concept in the embedding_space to that value.
    
    Attributes:
        embedding_space (dict)

    """
    def __init__(self, embedding_space : dict, batch_size : int, epochs : int, dense_layers : list, 
                 dropout_layers : list = None, activation_function : str = 'relu', 
                 kernel_initializer : str = 'glorot_uniform', batch_normalization : str = None,
                 regularization_type : str = None, l1_regularization : float = 0.01, l2_regularization : float = 0.01,
                 embedding_size : int = 200, relation_prediction : bool = True, rules : dict = None):
        """
        Parameters:
            embedding_space (dict):
                Dictionary that associates ids (keys) to embeddings (values).
            batch_size (int):
                Number of training examples used in one iteration of the training.
            epochs (int):
                Number of epochs (iterations through the dataset) during training.
            dense_layers (list):
                Structure of the neural network. The length of the list is the number of hidden layers, while
                the value of each index is the amount of units in that layer.
            dropout_layers (list):
                Similar to dense_layers, but defines whether there is a dropout layer after each hidden layer. The value
                in each position defines how much dropout is applied (between 0 and 1). If for a certain index the value is 0,
                no dropout layer is created.
            activation_function (str):
                String that represents the activation function to be applied. It uses the names established by keras.
            kernel_initializer (str):
                String that represents how to intialize the weights of the kernel. It uses the names established by keras.
            batch_normalization (str):
                String that represents whether to apply batch normalization 'After' or 'Before' the activation function. If set to None,
                no batch normalization will be applied.
            regularization_type (str):
                String that represents whether to apply 'l1', 'l2', or 'l1l2' regularization. If set to None, no regularization will be applied.
            l1_regularization (float):
                Float value that represents the l1 regularization to apply if regularization_type is 'l1' or 'l1l2'.
            l2_regularization (float):
                Float value that represents the l2 regularization to apply if regularization_type is 'l2' or 'l1l2'.
            embedding_size (int):
                Number of dimensions of the embeddings.
            relation_prediction (bool):
                Represents if the task of the Neural Network will be relation prediction (True) or analogy prediction (False).
            rules (dict):
                Two level dictionary used for filtering the possible options in the prediction. The two keys are
                the elements used in the Neural Network, while the value is set the of options.
            
        """
        # Set the embedding space, so that after predicting the embedding, the similarity is
        # calculated with embeddings of this space
        self.embedding_space = embedding_space

        # Rules to be used for filtering results in the prediction step. It is a two level dictionary, where the
        # first key is the subject concept, and the second key is either the object concept (relation prediction)
        # or the relationship (analogy prediction)
        self.rules = rules
        
        # Establish the task to perform
        self.relation_prediction = relation_prediction

        # Establish the size of the embeddings
        self.embedding_size = embedding_size

        # Specify training hyperparameters of the NN
        if self.relation_prediction:
            self.name = 'EmbeddingRelationPredictor'
        else:
            self.name = 'EmbeddingAnalogyPredictor'
        self.batch_size = batch_size
        self.epochs = epochs

        # Specify the structure of the NN
        self.dense_layers = dense_layers
        self.dropout_layers = dropout_layers

        # Specify other hyperparameters of the NN
        self.activation_function = activation_function
        self.kernel_initializer = kernel_initializer
        self.batch_normalization = batch_normalization
        if regularization_type == 'l1':
            self.regularization = regularizers.L1(l1=l1_regularization)
        elif regularization_type == 'l2':
            self.regularization = regularizers.L2(l2=l2_regularization)
        elif regularization_type == 'l1l2':
            self.regularization = regularizers.L1L2(l1=l1_regularization, l2=l2_regularization)
        else:
            self.regularization = None


        self.model = self.build_model()

    def build_model(self) -> keras.Sequential:
        """Method that builds, compiles, and returns a keras sequential model using the hyperparameters
        defined in the constructor.
        
        Returns:
            A compiled keras Sequential model.
        """
        # Create the model
        model = keras.Sequential(name=self.name)

        # Define the input according to the task
        if self.relation_prediction:
            model.add(InputLayer(input_shape=(self.embedding_size * 2, )))
        else:
            # TODO: We need to decide the input for the analogy prediction task
            pass

        # Hidden layers of the model
        for i in range(len(self.dense_layers)):
            # Add the hidden layer
            model.add(Dense(self.dense_layers[i], kernel_initializer=self.kernel_initializer,
                            kernel_regularizer=self.regularization))
            
            # Add batch normalization if set to Before
            if self.batch_normalization == 'Before':
                model.add(BatchNormalization())
            
            # Add the activation function
            model.add(Activation(self.activation_function))

            # Add batch normalization if set to After
            if self.batch_normalization == 'After':
                model.add(BatchNormalization())
            
            # Add any possible dropout layer
            if self.dropout_layers is not None and self.dropout_layers[i] != 0:
                model.add(Dropout(self.dropout_layers[i]))


        # Set the output layer
        model.add(Dense(self.embedding_size))

        # Compile the model
        model.compile(loss= DEFAULT_LOSS_FUNCTION,
                      optimizer=DEFAULT_OPTIMIZER_FUNCTION)

        return model
    
    def train_model(self, X_train, y_train, X_dev = None, y_dev = None, verbose : int = 1):
        """Method to train the model. If X_dev and y_dev are not None, then validation data will also be returned. 
        
        Parameters:
            X_train:
                Input data for training, which can be a numpy array, tensorflow tensor, etc.
            y_train:
                Target data for training. Similar to X_train, and should be consistent with it.
            X_dev:
                Input data for validation.
            y_dev:
                Target data for validation.
            verbose (int):
                Verbosity mode, where 0 is silent, 1 is a progress bar, and 2 is a single line.
        
        Returns:
            A History keras object, which records the training and validation loss and metrics values at successive epochs.
        """
        if X_dev is not None and y_dev is not None:
            history = self.model.fit(X_train, y_train, batch_size=self.batch_size,
                                     epochs=self.epochs, verbose=verbose,
                                     validation_data=(X_dev, y_dev))
        else:
            history = self.model.fit(X_train, y_train, batch_size=self.batch_size,
                                     epochs=self.epochs, verbose=verbose)

        return history

    def predict(self, X_test, X_ids = None, return_top10 : bool = True):
        """Method to perform the prediction using the input data X_test. If the tuple X_ids is not None, and rules
        was set during the initialization of the class, those rules are used to filter results.
        
        Parameters:
            X_test:
                Input data for prediction, which can be a numpy array, tensorflow tensor, etc.
            X_ids:
                List of tuples of size consistent with X_test, that contains the values needed for using the rules 
                for filtering.
            return_top10 (bool):
                Whether to return the 10 most similar concepts for each entry in the input data, or just the most
                similar.
        
        Returns:
            If return_top10 is set to True, returns a list of size consistent with X_test, where each entry is a
            list of up to 10 tuples (similarity_value, relation_id). If set to False, returns a list of tuples
            (similarity_value, relation_id).
        """
        # Extract IDs and embeddings from the target embedding space
        space_ids = list(self.embedding_space.keys())
        space_embeddings = list(self.embedding_space.values())

        # List to store the predicted values after searching in the embedding space
        predicted_values = []
        
        # Perform the prediction
        prediction = self.model.predict(X_test)

        # Perform the cosine similarity searching between the prediction and the embeddings from the target space
        similarity_values = cosine_similarity(prediction, space_embeddings)
        
        
        for i in range(len(similarity_values)):
            similarity_value = similarity_values[i]
            
            # We zip the similarity values with the space ids, and sort it
            # to find which is the most similar ID from the target space
            sim_zip = list(zip(similarity_value, space_ids))
            sim_zip.sort(key = lambda x : x[SIMILARITY_INDEX], reverse=True)

            # If there are filtering rules, and IDs to use it, we filter the results
            if self.rules is not None and X_ids is not None:
                rules_keys_ids = X_ids[i]
                possible_targets = []

                # Obtain the possible targets using the rules dictionary
                for first_level_key  in rules_keys_ids[0]:
                    if first_level_key in self.rules:
                        for second_level_key in rules_keys_ids[1]:
                            if second_level_key in self.rules[first_level_key][second_level_key]:
                                possible_targets += self.rules[first_level_key][second_level_key]

                # We only keep the results that are part of the possible targets
                if len(possible_targets) > 0:
                    sim_zip_aux = []

                    for value_zip in sim_zip:
                        target_id = value_zip[ID_INDEX]
                        if target_id in possible_targets:
                            sim_zip_aux.append(value_zip)

                    sim_zip = sim_zip_aux
            
            # We either return the top 10 values for metric purposes (to study top5, top10, etc.)
            # or we return only the most similar
            if return_top10:
                predicted_values.append(sim_zip[:10])
            else:
                predicted_values.append(sim_zip[0]) 
   

        return predicted_values
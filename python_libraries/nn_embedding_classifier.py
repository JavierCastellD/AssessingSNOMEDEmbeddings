import keras
from keras import regularizers, optimizers
from keras.layers import InputLayer, Dense, Activation, Dropout, BatchNormalization

DEFAULT_LOSS_FUNCTION = 'categorical_crossentropy'
DEFAULT_OPTIMIZER_FUNCTION = 'adam'

SIMILARITY_INDEX = 0
ID_INDEX = 1

class EmbeddingClassifier:
    """Class that represents a neural network which receives a number of embeddings to predict a number of classes given in the
    constructor.
    
    Attributes:
        model (keras.Sequential):
            Neural network with the architecture defined in the constructor used for prediction.
    """
    def __init__(self, number_of_classes : int, embedding_input : int, batch_size : int, epochs : int, dense_layers : list, 
                 dropout_layers : list = None, activation_function : str = 'relu', 
                 kernel_initializer : str = 'glorot_uniform', batch_normalization : str = None,
                 regularization_type : str = None, l1_regularization : float = 0.01, l2_regularization : float = 0.01,
                 optimizer_function : str = 'adam', learning_rate : float = 0.001, embedding_size : int = 200):
        """
        Parameters:
            number_of_classes (int):
                Number of classes for the prediction.
            embedding_input (int):
                Number of input embeddings to perform the classification.
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
            optimizer_function (str):
                String that represents which optimizer to use, i.e, the function to adjust the weights of the neural network. The ones available are:
                adam, sgd, rmsprop, adadelta, adagrad, adamax, and nadam.
            learning_rate (float):
                Float value between 0 and 1 that represents the learning rate for the optimizer.
            embedding_size (int):
                Number of dimensions of the embeddings.
        """
        # Number of classes
        self.number_of_classes = number_of_classes

        # Establish the number of embeddings. Used for calculating input size
        self.embedding_input = embedding_input

        # Establish the size of the embeddings
        self.embedding_size = embedding_size

        # Specify training hyperparameters of the NN
        self.name = 'EmbeddingClassifier'
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

        if optimizer_function == 'adam':
          self.optimizer_function = optimizers.Adam(learning_rate)
        elif optimizer_function == 'sgd':
          self.optimizer_function = optimizers.SGD(learning_rate)
        elif optimizer_function == 'rmsprop':
          self.optimizer_function = optimizers.RMSprop(learning_rate)
        elif optimizer_function == 'adadelta':
          self.optimizer_function = optimizers.Adadelta(learning_rate)
        elif optimizer_function == 'adagrad':
          self.optimizer_function = optimizers.Adagrad(learning_rate)
        elif optimizer_function == 'adamax':
          self.optimizer_function = optimizers.Adamax(learning_rate)
        elif optimizer_function == 'nadam':
          self.optimizer_function = optimizers.Nadam(learning_rate)
        else:
          self.optimizer_function = DEFAULT_OPTIMIZER_FUNCTION

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
        model.add(InputLayer(input_shape=(self.embedding_size * self.embedding_input, )))
        
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
        model.add(Dense(self.number_of_classes, activation='softmax'))

        # Compile the model
        model.compile(loss= DEFAULT_LOSS_FUNCTION,
                      optimizer=self.optimizer_function,
                      metrics=["categorical_accuracy"])

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

    def predict(self, X_test, verbose : int = 1):
        """Method to perform the prediction using the input data X_test. 
        
        Parameters:
            X_test:
                Input data for prediction, which can be a numpy array, tensorflow tensor, etc.
            verbose (int):
                Verbosity mode, where 0 is silent, 1 is a progress bar, and 2 is a single line.
        
        Returns:
            Returns a list of size consistent with X_test with the predictions.
        """
        # Perform the prediction
        return self.model.predict(X_test, verbose=verbose)
    
    def evaluate(self, X, y, verbose : int = 1):
        """Method to predict and evaluate using input data X, and target data y.

        Parameters:
            X:
                Input data for evaluation, which can be a numpy array, tensorflow tensor, etc.
            y:
                Target data for evaluation. Similar to X, and should be consistent with it.
            verbose (int):
                Verbosity mode, where 0 is silent, 1 is a progress bar, and 2 is a single line.

        Returns:
            A list of scalars.
        """
        # Perform the evaluation
        return self.model.evaluate(X, y, verbose=verbose)
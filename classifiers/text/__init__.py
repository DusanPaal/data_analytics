# third-party imports
import numpy as np

class Vectorizer:
    '''Vectorizer for text sequences and their 
    labels for a binary classification model.
    '''

    def __init__(self, dtype) -> None:
        '''Initializes the Vectorizer.

        Parameters:
        -----------
        dtype (numpy.dtype):
        The data type to use for the one-hot encoded
        vectors.
        '''

        self._dtype = dtype

    def vectorize_text(self, sequences, dimension) -> np.ndarray:
        '''Converts a list of sequences  
        into a one-hot encoded matrix.

        Parameters:
        -----------
        sequences (list of list of int):
        A list where each element is a sequence of integers.

        dimension (int):
        The size of the one-hot encoding dimension.

        Returns:
        --------
        numpy.ndarray: 
        A 2D numpy array with shape (len(sequences), dimension)
        where each row is a one-hot encoded vector.
        '''

    
        vec_shape = (len(sequences), dimension)
        vec = np.zeros(vec_shape, dtype=self._dtype)
    
        for idx, sequence in enumerate(sequences):
            vec[idx, sequence] = 1
    
        return vec
    
    def vectorize_labels(self, labels) -> np.ndarray:
        '''Converts a list of labels into a
        one-hot encoded matrix.

        Parameters:
        -----------
        labels (list of int):
        A list of integer labels.

        Returns:
        --------
        numpy.ndarray:
        A 2D numpy array with shape (len(labels), 1)
        where each row is a one-hot encoded vector.
        '''
        
        return np.asarray(labels).astype(self._dtype)
    
class Splitter:
    '''Splitter for text sequences and their
    labels into training and validation sets.
    '''

    def __init__(self, split_idx) -> None:
        '''Initializes the Splitter.

        Parameters:
        -----------
        split_idx (int):
        The index at which to split the sequence.
        Elements from this index onwards will be
        in the training set, and elements before
        this index will be in the validation set.
        '''

        self._idx = split_idx

    def split_train_validation(self, sequence) -> tuple:
        '''Splits a sequence into training and 
        validation sets based on the provided index.

        Parameters:
        -----------
        sequence (list or any sequence type):
        The sequence to be split.

        Returns:
        --------
        A tuple containing two elements:
        - training (same type as input sequence):
            The training set, containing elements 
            from the index to the end of the sequence.
        - validation (same type as input sequence): 
            The validation set, containing elements 
            from the start of the sequence up to 
            (but not including) the index.
        '''

        training = sequence[self._idx:]
        validation = sequence[:self._idx]

        if training.shape[0] == 0 or validation.shape[0] == 0:
            raise ValueError(
                'Training set is empty! This is likely '
                'due to an invalid split index. Ensure '
                'the split index is less than the length '
                'of the sequence.'
            )

        return (training, validation)
    

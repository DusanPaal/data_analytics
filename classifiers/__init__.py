# standard library imports
import os

# third-party imports
import numpy as np

# NOTE: Tensorflow log level must be set before the tensorflow is imported. 
KERAS_BACKEND_TF = 'tensorflow'
KERAS_BACKEND_TA = 'theano'
KERAS_BACKEND_PT = 'pytorch'

os.environ['KERAS_BACKEND'] = KERAS_BACKEND_TF


# NOTE: Keras should only be imported after the backend has been configured.
# The backend cannot be changed once the package is imported.
TF_LOG_ALL = '0'        # all messages (default)
TF_LOG_INFO = '1'       # info logs
TF_LOG_WARNING = '2'    # warning messages
TF_LOG_ERROR = '3'      # error messages

tf_log_level = TF_LOG_INFO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_level


# now that keras backend and tensorflow
# log level are set, we can import tensorflow
import tensorflow as tf


class System:
    '''Prints system information.'''

    # NOTE: imlementing GPU and CPU detection 
    # as a separate function in case we decide 
    # # to amke it public at a later stage
    @staticmethod
    def _detect_devices(dev: str) -> list:
        '''Detects the CPU and GPU devices
        available to TensorFlow.

        Parameters:
        -----------
        dev : str
            The device to detect:
            - 'CPU': Central Processing Unit
            - 'GPU: Graphics Processing Unit

        Returns:
        --------
        list
            A list of devices available to TensorFlow.
        '''

        if dev.upper() not in ['CPU', 'GPU']:
            raise ValueError(
                "Invalid device! "
                "Choose from 'CPU' or 'GPU'."
            )
        
        return tf.config.list_physical_devices(dev.upper())

    @staticmethod
    def print_info():
        '''Prints information about the setup of the system.'''

        print("System Information:")
        print('-------------------')

        keras_backend = os.environ.get('KERAS_BACKEND', None)
        print(f"Keras Backend: {keras_backend}")

        tf_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', TF_LOG_ALL)
        print(f"Tensorflow log level: {tf_log_level}")

        # List available physical devices
        print("CPUs Available:", System._detect_devices('CPU'))
        print("GPUs Available:", System._detect_devices('GPU'))

        print('-------------------')
        
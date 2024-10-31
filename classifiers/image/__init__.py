'''Package to manage a binary classifier.'''

import os
from os.path import join
import shutil
import numpy as np
from tensorflow.keras.preprocessing import image

# NOTE: Tensorflow log level must be set before the tensorflow is imported. 
keras_backend = 'tensorflow'
os.environ["KERAS_BACKEND"] = keras_backend
print("Info:", f"Environment variable 'KERAS_BACKEND' set to '{keras_backend}'")

# NOTE: Keras should only be imported after the backend has been configured.
# The backend cannot be changed once the package is imported.
tf_log_level = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = tf_log_level
print("Info:", f"Environment variable 'TF_CPP_MIN_LOG_LEVEL' set to '{tf_log_level}'")

class Sampler:
    '''Class to manage splitting a pool of images 
    into training, testing, and validation sets.
    '''
    
    def __init__(self) -> None:
        '''Instantiates the class.'''

        root = join('data', 'dogs_cats')
        base = 'base'
        
        self._mapping = {
        
            # dogs
            'dogs_train': {
                'images': [f'{i}.jpg' for i in range(1000)],
                'src_dir': join(root, base, 'dogs'),
                'dst_dir': join(root, 'train', 'dogs')
            },
     
            'dogs_test': {
                'images': [f'{i}.jpg' for i in range(1500, 2000)],
                'src_dir': join(root, base, 'dogs'),
                'dst_dir': join(root, 'test', 'dogs')
            },

            'dogs_validation': {
                'images': [f'{i}.jpg' for i in range(1000, 1500)],
                'src_dir': join(root, base, 'dogs'),
                'dst_dir': join(root, 'validation', 'dogs')
            },

            # cats
            'cats_train': {
               'images': [f'{i}.jpg' for i in range(1000)],
                'src_dir': join(root, base, 'cats'),
                'dst_dir': join(root, 'train', 'cats'),
            },
 
            'cats_test': {
                'images': [f'{i}.jpg' for i in range(1500, 2000)],
                'src_dir': join(root, base, 'cats'),
                'dst_dir': join(root, 'test', 'cats'),
            },
            
            'cats_validation': {
                'images': [f'{i}.jpg' for i in range(1000, 1500)],
                'src_dir': join(root, base, 'cats'),
                'dst_dir': join(root, 'validation', 'cats')
            }
        
        }

    def split_images(self) -> None:
        '''Splits a pool of images 
        into training, testing, and
        validation sets.
        '''

        for val in self._mapping.values():
            for img in val['images']:
                src = join(val['src_dir'], img)
                dst = join(val['dst_dir'], img)
                shutil.copyfile(src, dst)

class BinaryClassifier:
    '''Classifies images into two categories.'''

    def __init__(self) -> None:
        '''Instantiates the class.'''
        self._model = None

    @property
    def model(self):
        '''The binary classification model.'''
        return self._model

    def predict(self, img_path):
        '''Predicts the category of an image.

        # Parameters
        ----------
        img_path : str
            The path to the image file.
        '''

        # if self.model is None:
        #     raise ValueError(
        #         'Model not available! Create or load '
        #         'the model before making predictions.'
        #     )
        
        img = image.load_img(img_path, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.0

        return self.model.predict(img_tensor)

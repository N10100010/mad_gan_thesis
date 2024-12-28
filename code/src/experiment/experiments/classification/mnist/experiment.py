import tensorflow as tf

from datasets.mnist import dataset_func
from experiment.base_experiments import BaseExperiment

class CLASS_MNIST_Experiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        
    def _setup(self):
        pass
    
    def _load_data(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = dataset_func(
            self.size_dataset, self.batch_size, self.n_gen
        )
    
    def _initialize_models(self):
        inp = tf.keras.layers.Input(shape=(28, 28, 1))
        
        x = tf.keras.layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]
        )(inp)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Flatten()(x)
        out = tf.keras.layers.Dense(10, activation="softmax")(x)
        
        model = tf.keras.models.Model(inp, out, name="MNIST Classifier")
        self.classifier = model
    
    def _run(self):
        pass

import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
from logging_practice import setup_logger, log_function

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

logger = setup_logger(level=logging.DEBUG)

"""
Croe idea unlike pytorch, labels MUST be standardized.

Data format problem - for classification - we may need different formats for specif cufucntion - check to make sure heuristically that the right format is used, whether integer encoding(sparse representation) or one hot encoding.

output - probabiltiies distribution
data - somehow a one hot encooding is needed. 


differences in sparse forrmat vs one hot format - this is  DF forest probelm but i should be aware of this issue.=

"""

# Issue 1 - nesuring with pytorch, adequate label reuqiremetns - harder to debug with tensorflow due to dataset object being used not the tensors / raw data.


@log_function(logger, logging.INFO)
def create_dummy_tf_datasets(hyperparameters):
    tf.random.set_seed(42)
    np.random.seed(42)

    batch_size, img_height, img_width, n_channels, n_classes, n_total_images = hyperparameters

    logging.debug(f"num classes: {n_classes}")
    
    # Create dummy images
    dummy_images = tf.random.uniform(
        shape=(n_total_images, img_height, img_width, n_channels),
        minval=0,
        maxval=255,
        dtype=tf.float32
    )
    
    # Create one-hot encoded labels
    dummy_labels = tf.one_hot([i % n_classes for i in range(n_total_images)], n_classes) # NOTE here that the albels are of 2 classes -->
    
    train_size = int(0.8 * n_total_images)
    
    train_images = dummy_images[:train_size]
    train_labels = dummy_labels[:train_size]
    val_images = dummy_images[train_size:]
    val_labels = dummy_labels[train_size:]
    
    # Create tf.data.Dataset objects
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    
    # Configure datasets for performance
    train_ds = train_ds.shuffle(buffer_size=1000, seed=42)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    validation_ds = validation_ds.batch(batch_size)
    validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)
    
    class_names = [str(i) for i in range(n_classes)]
    train_ds.class_names = class_names
    validation_ds.class_names = class_names
    
    return train_ds, validation_ds

def create_model(img_height, img_width, n_channels, num_classes):
    logger.info("Defining model object")
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                           input_shape=(img_height, img_width, n_channels)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    # Hyperparameters
    batch_size = 2
    img_height = 30
    img_width = 30
    n_channels = 3
    num_classes = 10
    n_total_images = 10

    hyperparameters = [batch_size, img_height, img_width, n_channels, num_classes, n_total_images]

    # Create datasets
    train_ds, validation_ds = create_dummy_tf_datasets(hyperparameters)
    
    # Print dataset information
    logger.debug(f"Train dataset info: {train_ds}")
    
    # Create and compile model
    model = create_model(img_height, img_width, n_channels, num_classes)
    
    learning_rate = 0.001
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Use categorical crossentropy since we're using one-hot encoded labels
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    
    metrics = [
        'accuracy',
        keras.metrics.Recall()
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    # Print model summary
    model.summary()

    # Train the model
    epochs = 8
    try:
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=epochs,
            verbose=1
        )
        logger.info("Training completed successfully")
        return history
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
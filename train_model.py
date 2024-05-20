from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import wandb
from wandb.integration.keras import WandbMetricsLogger
import os

run = wandb.init(
    project="wandb-trash-classification",
    # hyperparameters are set for the sake of speed and simplicity
    config={
        "learning_rate": 0.1,
        "loss": "categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 5,
        "batch_size": 32
    }
)

config = wandb.config
entity = "adamata-selection"

def generate_data(images_dir):
    # Without data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255, 
        validation_split=0.2
    )
    
    
    train_generator = datagen.flow_from_directory(
        images_dir,
        target_size=(128, 128),
        batch_size=config.batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        images_dir,
        target_size=(128, 128),
        batch_size=config.batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

def build_cnn(train_generator):
    n_class = len(train_generator.class_indices)

    i = Input(shape=(128, 128, 3))

    # Convolutional Layers {Conv --> BatchNorm --> Conv --> BatchNorm --> MaxPooling (3x)}
    x = Conv2D(32, (3,3), padding='same', activation='relu')(i)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    # Neural Networks Layer
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(n_class, activation='softmax')(x)

    model = Model(i, x)

    return model

def compile_train_model(train_generator, validation_generator, model):
    model.compile(optimizer=Adam(learning_rate=config.learning_rate), 
              loss=config.loss, 
              metrics=[config.metric])
    
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=config.epoch,
        callbacks=[
            WandbMetricsLogger()
        ]
    )
    

    # Save model to W&B
    path = "./trash-classification-automated.keras"
    registered_model_name = "trash-classification-dev"

    run.link_model(path=path, registered_model_name=registered_model_name)

    print(f"The model has been trained and saved in W&B '{path}'")

    wandb.finish()

if __name__ == "__main__":

    images_dir = os.path.expanduser('dataset-resized')

    train_generator, validation_generator = generate_data(images_dir)

    model = build_cnn(train_generator)

    compile_train_model(train_generator, validation_generator, model)
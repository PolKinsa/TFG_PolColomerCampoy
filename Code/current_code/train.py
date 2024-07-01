import os
import numpy as np
from utils.data_generator import generate_toy_problems
from utils.preprocessing import prepare_data
from models.unet import unet_model
import tensorflow as tf

def train_model(epochs=30, batch_size=2, validation_split=0.2, save_interval=3):
    simple_elements_dir = 'data/simple_elements'
    if not os.path.exists(simple_elements_dir):
        raise FileNotFoundError(f"Directorio no encontrado: {simple_elements_dir}")

    # Generar 10-15 ToyProblems
    toy_problems = generate_toy_problems(simple_elements_dir, num_problems=15)
    print("Toy problems generated, starting to prepare data . . .")

    inputs, targets = prepare_data(toy_problems)

    print(f"Shape of inputs: {inputs.shape}")
    print(f"Shape of targets: {targets.shape}")

    input_shape = (inputs.shape[1], inputs.shape[2], 1)
    model = unet_model(input_shape)

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_freq=save_interval * len(inputs) // batch_size)

    class ShowPerformance(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(
                f"Loss: {logs['loss']}, Accuracy: {logs['accuracy']}, Validation Loss: {logs.get('val_loss')}, Validation Accuracy: {logs.get('val_accuracy')}")

    show_performance_callback = ShowPerformance()

    model.fit(inputs, targets, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
              callbacks=[cp_callback, show_performance_callback])

    model.save('models/unet_model.h5')

if __name__ == '__main__':
    train_model(epochs=30, batch_size=2, validation_split=0.2, save_interval=3)

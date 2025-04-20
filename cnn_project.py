import psutil
import tensorflow as tf
import tensorflow.keras.preprocessing as tfkp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import time


class MemoryUsageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if tf.config.list_physical_devices('GPU'):
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            print(f"[Epoch {epoch + 1}] GPU Memory - "
                  f"Current: {mem_info['current'] / 1e6:.2f} MB | "
                  f"Peak: {mem_info['peak'] / 1e6:.2f} MB")

def normalize(image, label):
    image = image/255
    return image, label


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    dataset = tfkp.image_dataset_from_directory(
        'dataset_split/train',
        labels='inferred',
        label_mode='int',
        class_names=['Bacterial_spot', 'Early_blight', 'Healthy', 'Late_blight',
                     'Leaf_mold', 'Septoria_leaf_spot', 'Spider_mites', 'Target_spot',
                     'Tomato_mosaic_virus', 'Tomato_yellow_leaf_curl_virus'],
        color_mode='rgb',
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        verbose=True
    )
    dataset = dataset.map(normalize)

    dataset_val = tfkp.image_dataset_from_directory(
        'dataset_split/val',
        labels='inferred',
        label_mode='int',
        class_names=['Bacterial_spot', 'Early_blight', 'Healthy', 'Late_blight',
                     'Leaf_mold', 'Septoria_leaf_spot', 'Spider_mites', 'Target_spot',
                     'Tomato_mosaic_virus', 'Tomato_yellow_leaf_curl_virus'],
        color_mode='rgb',
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        verbose=True
    )
    dataset_val = dataset_val.map(normalize)

    dataset_test = tfkp.image_dataset_from_directory(
        'dataset_split/test',
        labels='inferred',
        label_mode='int',
        class_names=['Bacterial_spot', 'Early_blight', 'Healthy', 'Late_blight',
                     'Leaf_mold', 'Septoria_leaf_spot', 'Spider_mites', 'Target_spot',
                     'Tomato_mosaic_virus', 'Tomato_yellow_leaf_curl_virus'],
        color_mode='rgb',
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        verbose=True
    )
    dataset_test = dataset_test.map(normalize)

    class_num = 10  # number of classes/labels

    # CNN Model
    model = Sequential([
        Input(shape=(256,256,3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(class_num, activation='softmax')
    ])

    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Summary
    model.summary()

    # Train the model
    start_time = time.time()
    model.fit(
        dataset,
        epochs=10,
        validation_data=dataset_val,
        verbose=2
    )
    training_time = time.time() - start_time

    # Save the model
    model.save("leaf_disease_classifier_6.h5")
    print("Total training time: %.2f" % training_time)

    # Test the model
    loss, accuracy = model.evaluate(dataset_test, verbose=2)
    print('Accuracy: %.2f | Loss: %.2f' % (accuracy * 100, loss))

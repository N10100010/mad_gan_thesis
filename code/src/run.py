from pathlib import Path

import tensorflow as tf
from datasets.new_dataset_creator import DatasetCreator
from matplotlib import pyplot as plt

if __name__ == "__main__":
    dsc = DatasetCreator(
        dataset="mnist",
        experiment_folder_path="C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-12_MADGAN_MNIST_5_GEN_DataCreation_SPEC_GEN_0",
        tf_dataset_load_func=tf.keras.datasets.mnist.load_data,
        number_of_generated_images_per_class={i: 2000 for i in range(10)},
        number_of_real_images_per_class={i: 2000 for i in range(10)},
    )

    train_x, train_y, test_x, test_y = dsc.create_dataset()

    # Define a simple CNN model
    # model = keras.Sequential(
    #     [
    #         layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    #         layers.MaxPooling2D((2, 2)),
    #         layers.Conv2D(64, (3, 3), activation="relu"),
    #         layers.MaxPooling2D((2, 2)),
    #         layers.Flatten(),
    #         layers.Dense(128, activation="relu"),
    #         layers.Dense(10, activation="softmax"),  # 10 classes for classification
    #     ]
    # )

    # # Compile the model
    # model.compile(
    #     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    # )

    # # Train the model
    # model.fit(
    #     train_x, train_y, epochs=5, batch_size=32, validation_data=(test_x, test_y)
    # )

    # # Evaluate on the test set
    # test_loss, test_acc = model.evaluate(test_x, test_y)
    # print(f"Test Accuracy: {test_acc:.4f}")

    # from pathlib import Path
    #
    # from matplotlib import pyplot as plt
    #
    Path("./test").mkdir(exist_ok=True)
    #

    def denormalize_image(image):
        """
        Convert image from arbitrary range to [0, 255] range.
        This handles both the [-1, 1] normalization and any outliers.
        """
        # First clip the values to a reasonable range
        image = tf.clip_by_value(image, -1.0, 1.0)
        # Convert from [-1, 1] to [0, 1]
        image = (image + 1.0) / 2.0
        # Convert to [0, 255] range and ensure uint8 type
        # I dont like that this rounds the values, but it's the simplest way
        image = tf.cast(image * 255, tf.uint8)
        return image

    #
    train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train = train.take(10)
    for i, (image, label) in enumerate(train):
        print(f"\nProcessing image {i}:")
        print(f"Image shape: {image.shape}")
        print(
            f"Image value range: [{tf.reduce_min(image):.2f}, {tf.reduce_max(image):.2f}]"
        )
        print(f"Label: {label.numpy()}")
        #
        save_image = denormalize_image(image)
        #
        save_path = str(Path("./test") / f"image_{i}.png")
        fig, ax = plt.subplots()
        ax.imshow(save_image.numpy())
        ax.set_title(str(label.numpy()))
        ax.axis("off")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    #
    # tf.keras.utils.save_img(save_path, save_image.numpy())

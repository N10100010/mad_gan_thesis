# from experiment.experiments.cifar_vanilla_gan.experiment import (
#     CIFAR_VanillaGAN_Experiment,
# )

import tensorflow as tf
from experiment.experiments.cifar_vanilla_gan.experiment import (
    CIFAR_VanillaGAN_Experiment,
)
from experiment.experiments.generative_creation.gan.experiment import (
    GAN_GenerativeCreationExperiment,
)
from experiment.experiment_queue import ExperimentQueue

if __name__ == "__main__":
    experiments = [
        # GenerativeCreationExperiment(
        #     name="Fashion_MNIST_DataCreation",
        #     experiment_class=FASHION_MNIST_MADGAN_Experiment,
        #     experiment_path="experiments\\2025-01-02_FASHION_MNIST_MADGAN_Experiment__6_n_gen_6",
        #     latent_point_generator=generate_latent_points,
        #     n_images=1,
        # ),
        # CIFAR_MADGAN_Experiment(
        #     name="TEST_better_discriminator_CIFAR_MADGAN_Experiment_2",
        #     n_gen=2,
        #     latent_dim=256,
        #     epochs=2,
        #     experiment_suffix="n_gen_2",
        # ),
        # MNIST_VanillaGAN_Experiment(
        #     name="MNIST_VanillaGAN_Experiment__150",
        #     latent_dim=100,
        #     epochs=150,
        #     experiment_suffix="",
        # ),
        # MADGAN_GenerativeCreationExperiment(
        #     name="Fashion_MNIST_DataCreation",
        #     experiment_class=FASHION_MNIST_MADGAN_Experiment,
        #     experiment_path="experiments\\2025-01-02_FASHION_MNIST_MADGAN_Experiment__6_n_gen_6",
        #     latent_point_generator=generate_latent_points,
        #     n_images=1,
        # ),
        # MNIST_VanillaGAN_Experiment(
        #     name="MNIST_VanillaGAN_Experiment__",
        #     latent_dim=100,
        #     epochs=10,
        #     experiment_suffix="epochs_10",
        # ),
        # FASHION_MNIST_VanillaGAN_Experiment(
        #     name="FASHION_MNIST_VanillaGAN_Experiment__",
        #     latent_dim=100,
        #     epochs=10,
        #     experiment_suffix="epochs_10",
        # ),
        # GAN_GenerativeCreationExperiment(
        #     name="generative_creation_test",
        #     experiment_class=MNIST_VanillaGAN_Experiment,
        #     experiment_path="experiments/2025-01-14_MNIST_VanillaGAN_Experiment___epochs_10",
        #     latent_point_generator=tf.random.normal,
        #     n_images=50,
        # ),
        # GAN_GenerativeCreationExperiment(
        #     name="generative_creation_test_fashion",
        #     experiment_class=FASHION_MNIST_VanillaGAN_Experiment,
        #     experiment_path="experiments/2025-01-14_FASHION_MNIST_VanillaGAN_Experiment___epochs_10",
        #     latent_point_generator=tf.random.normal,
        #     n_images=50,
        # ),
        # GAN_GenerativeCreationExperiment(
        #     name="generative_creation_test_cifar",
        #     experiment_class=CIFAR_VanillaGAN_Experiment,
        #     experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_200_epochs_200",
        #     latent_point_generator=tf.random.normal,
        #     n_images=50,
        # ),
        # CIFAR_VanillaGAN_Experiment(
        #     name="CIFAR_VanillaGAN_Experiment__",
        #     latent_dim=128,
        #     epochs=200,
        #     experiment_suffix="latent_128_epochs_200",
        # ),
        # CIFAR_VanillaGAN_Experiment(
        #     name="CIFAR_VanillaGAN_Experiment__",
        #     latent_dim=256,
        #     epochs=200,
        #     experiment_suffix="latent_256_epochs_200",
        # ),
        CIFAR_VanillaGAN_Experiment(
            name="CIFAR_VanillaGAN_Experiment__",
            latent_dim=200,
            epochs=2000,
            experiment_suffix="latent_200_epochs_2000",
        ),
        GAN_GenerativeCreationExperiment(
            name="generative_creation_test_cifar_1",
            experiment_class=CIFAR_VanillaGAN_Experiment,
            experiment_path="experiments/2025-01-31_CIFAR_VanillaGAN_Experiment___latent_200_epochs_200",
            latent_point_generator=tf.random.normal,
            n_images=5,
            save_raw_image=True
        ),
        # GAN_GenerativeCreationExperiment(
        #     name="CIFAR_GENERATIVE_VanillaGAN_Experiment",
        #     experiment_class=CIFAR_VanillaGAN_Experiment,
        #     experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_200_epochs_200",
        #     latent_point_generator=tf.random.normal,
        #     experiment_suffix="__latent_200_epochs_200",
        #     save_raw_image=True,
        #     n_images=20,
        # )
        # GAN_GenerativeCreationExperiment(
        #     name="CIFAR_GENERATIVE_VanillaGAN_Experiment",
        #     experiment_class=CIFAR_VanillaGAN_Experiment,
        #     experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_100_epochs_200",
        #     latent_point_generator=tf.random.normal,
        #     experiment_suffix="__latent_100_epochs_200",
        #     n_images=20,
        # ),
        # GAN_GenerativeCreationExperiment(
        #     name="CIFAR_GENERATIVE_VanillaGAN_Experiment",
        #     experiment_class=CIFAR_VanillaGAN_Experiment,
        #     experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_150_epochs_200",
        #     latent_point_generator=tf.random.normal,
        #     experiment_suffix="__latent_150_epochs_200",
        #     n_images=20,
        # ),
        # GAN_GenerativeCreationExperiment(
        #     name="CIFAR_GENERATIVE_VanillaGAN_Experiment",
        #     experiment_class=CIFAR_VanillaGAN_Experiment,
        #     experiment_path="experiments/2025-01-14_CIFAR_VanillaGAN_Experiment___latent_200_epochs_200",
        #     latent_point_generator=tf.random.normal,
        #     experiment_suffix="__latent_200_epochs_200",
        #     save_raw_image=True,
        #     n_images=20,
        # )
        # CLASS_MNIST_Experiment(name="TEST--CLASS_MNIST_Experiment__", epochs=20),
        # CLASS_FashionMNIST_Experiment(
        #     name="TEST--CLASS_FashionMNIST_Experiment__", epochs=20
        # ),
        # CLASS_CIFAR10_Experiment(name="TEST--CLASS_CIFAR10_Experiment__", epochs=50),
    ]

    queue = ExperimentQueue()
    for exp in experiments:
       queue.add_experiment(exp)
    queue.run_all()

# import numpy as np
# import tensorflow as tf
#
#
# def preprocess_image(img_path, target_size=(32, 32)):
#     """
#     Preprocess an image for inference.
#
#     Args:
#         img_path (str): Path to the image file.
#         target_size (tuple): Target size of the image (height, width).
#
#     Returns:
#         np.array: Preprocessed image with shape (1, height, width, channels).
#     """
#     # Load the image and resize it
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
#
#     # Convert the image to a numpy array
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#
#     # Normalize pixel values to [0, 1]
#     img_array = img_array - 127.5 / 127.5
#
#     # Add a batch dimension
#     img_array = np.expand_dims(img_array, axis=0)
#
#     return img_array
#
#
# from model_definitions.classifiers import CIFAR10Classifier
#
# classifier = CIFAR10Classifier()
# _ = classifier(tf.random.normal((1, 32, 32, 3)))
#
# classifier.load_weights(
#     "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-01-30_TEST--CLASS_CIFAR10_Experiment__\\checkpoints\\best_weights.h5"
# )
#
#
# # Example usage
# img_path = "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-01-30_TEST--CLASS_CIFAR10_Experiment__\\0001.png"
# preprocessed_image = preprocess_image(img_path)
#
# print("shape ", preprocessed_image.shape)
#
# # Make predictions
# predictions = classifier(preprocessed_image)
# print(predictions)
#
# # Step 1: Get the predicted class index
# predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
#
# # Step 2: Map the index to the class name
# class_names = [
#     "airplane",
#     "automobile",
#     "bird",
#     "cat",
#     "deer",
#     "dog",
#     "frog",
#     "horse",
#     "ship",
#     "truck",
# ]
# predicted_class_name = class_names[predicted_class_index]
#
# # Print the result
# print(f"Predicted class index: {predicted_class_index}")
# print(f"Predicted class name: {predicted_class_name}")

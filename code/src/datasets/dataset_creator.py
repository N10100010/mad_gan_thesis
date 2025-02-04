import json
from collections.abc import Callable
from pathlib import Path

import tensorflow as tf


class DatasetCreator:
    """
    SKETCH. Check if the code is correct and complete.
    """

    def __init__(
        self,
        created_images_folder: Path,
        tf_dataset_load_func: Callable,
        number_of_generated_images_per_class: dict,
        number_of_real_images_per_class: dict,
    ):
        self.created_images_folder = created_images_folder
        self.created_images_labels_file = created_images_folder / "labels.json"
        self.tf_dataset_load_func = tf_dataset_load_func
        self.number_of_generated_images_per_class = number_of_generated_images_per_class
        self.number_of_real_images_per_class = number_of_real_images_per_class

    def create_dataset(self) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset object from the generated data (with labels) and real data,
        clamped to the given absolute numbers per class, and merged via concatenation.
        Both generated and real images are scaled to [-1, 1].

        Returns:
            tf.data.Dataset: the merged dataset.
        """

        generated_data = self._load_and_preprocess_generated_data()
        real_data = self.tf_dataset_load_func()

        # If the returned real_data is a tuple (e.g., (images, labels)), convert it into a tf.data.Dataset.
        if isinstance(real_data, tuple):
            # Expecting tuple of (images, labels)
            images, labels = real_data
            real_data = tf.data.Dataset.from_tensor_slices((images, labels))

        # Map real_data to scale its images to [-1, 1].
        real_data = real_data.map(
            self._normalize_real_image, num_parallel_calls=tf.data.AUTOTUNE
        )

        # Clamp both datasets per class using the provided absolute numbers.
        generated_data = self._clamp_dataset_to_percentage(
            generated_data, self.percentage_of_generated_images_per_class
        )
        real_data = self._clamp_dataset_to_percentage(
            real_data, self.percentage_of_real_images_per_class
        )

        # Merge the two datasets by concatenation.
        merged_dataset = self._merge_datasets(generated_data, real_data)
        return merged_dataset

    def _normalize_real_image(self, image, label):
        """
        Convert image to float32 and scale it from [0,255] (or [0,1]) to [-1,1].
        It assumes that if image is of type uint8, then it is in [0, 255].
        """
        image = tf.cast(image, tf.float32)
        # If image max is greater than 1, assume [0,255] and scale accordingly.
        # Otherwise assume it is in [0,1].
        image = tf.cond(
            tf.reduce_max(image) > 1.0,
            lambda: (image / 127.5) - 1.0,
            lambda: (image * 2.0) - 1.0,
        )
        return image, label

    def _clamp_dataset_to_percentage(
        self, dataset: tf.data.Dataset, clamp_dict: dict
    ) -> tf.data.Dataset:
        """
        For each class label in the dataset, limit the number of samples to the absolute number
        specified in clamp_dict. If a class label is not present in clamp_dict, all samples are taken.

        Args:
            dataset (tf.data.Dataset): dataset of (image, label)
            clamp_dict (dict): dictionary mapping class label (int) to desired number of images.

        Returns:
            tf.data.Dataset: a new dataset with at most the specified number of images per class.
        """
        # Collect the images and labels per class.
        data_by_class = {}
        for image, label in dataset.as_numpy_iterator():
            # Ensure the label is an integer.
            class_label = (
                int(label) if not isinstance(label, (list, tuple)) else int(label[0])
            )
            data_by_class.setdefault(class_label, []).append(image)

        clamped_images = []
        clamped_labels = []
        for class_label, images in data_by_class.items():
            target = clamp_dict.get(class_label, None)
            if target is not None:
                selected_images = images[:target]
            else:
                selected_images = images
            clamped_images.extend(selected_images)
            clamped_labels.extend([class_label] * len(selected_images))

        # Convert back to a tf.data.Dataset.
        # If images are tensors, stacking them ensures proper shape.
        images_tensor = tf.stack(clamped_images)
        labels_tensor = tf.convert_to_tensor(clamped_labels)
        return tf.data.Dataset.from_tensor_slices((images_tensor, labels_tensor))

    def _load_and_preprocess_generated_data(self) -> tf.data.Dataset:
        """
        Loads generated images (png files) from the self.created_images_folder, using the labels
        from labels.json, in the same folder. Preprocesses them by decoding and scaling to [-1,1].
        Supports both colored images (3 channels) and grayscale images (1 channel).

        Returns:
            tf.data.Dataset: dataset of (image, label) where image is scaled to [-1,1].
        """
        # Load labels from the JSON file.
        with open(self.created_images_labels_file, "r") as f:
            labels_data = json.load(f)

        images = []
        labels = []
        for filename, class_label in labels_data.items():
            image_path = self.created_images_folder / filename
            if not image_path.exists():
                print(f"Warning: {image_path} does not exist. Skipping.")
                continue
            # Read and decode the image without forcing a channel count.
            img_raw = tf.io.read_file(str(image_path))
            img = tf.image.decode_png(
                img_raw
            )  # Let the image keep its original number of channels.
            # Convert to float32 in [0, 1].
            img = tf.image.convert_image_dtype(img, tf.float32)
            # Scale to [-1, 1]
            img = (img * 2.0) - 1.0
            images.append(img)
            labels.append(class_label)

        if not images:
            raise ValueError(
                "No generated images were loaded. Please check the folder and labels.json."
            )

        # Stack images to ensure proper shape.
        images_tensor = tf.stack(images)
        labels_tensor = tf.convert_to_tensor(labels)
        return tf.data.Dataset.from_tensor_slices((images_tensor, labels_tensor))

    def _merge_datasets(
        self, generated_data: tf.data.Dataset, real_data: tf.data.Dataset
    ) -> tf.data.Dataset:
        """
        Merges the two datasets (generated and real) via concatenation.

        Args:
            generated_data (tf.data.Dataset): dataset of (image, label)
            real_data (tf.data.Dataset): dataset of (image, label)

        Returns:
            tf.data.Dataset: merged dataset.
        """
        return generated_data.concatenate(real_data)

if __name__ == "__main__":
    pass


# dsc = DatasetCreator(
#     created_images_folder=Path(
#         "C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\code\\experiments\\2025-02-03_generative_creation_test_cifar10"
#     ),
#     tf_dataset_load_func=tf.keras.datasets.cifar10.load_data,
#     number_of_generated_images_per_class={
#         0: 1,
#         1: 1,
#         2: 1,
#         3: 1,
#         4: 1,
#         5: 1,
#         6: 1,
#         7: 1,
#         8: 1,
#         9: 1,
#     },
#     number_of_real_images_per_class={
#         0: 1,
#         1: 1,
#         2: 1,
#         3: 1,
#         4: 1,
#         5: 1,
#         6: 1,
#         7: 1,
#         8: 1,
#         9: 1,
#     },
# )
# # Get the dataset without any batching
# ds = dsc.create_dataset()
#
# Path("./test").mkdir(exist_ok=True)
#
# def denormalize_image(image):
#     """
#     Convert image from arbitrary range to [0, 255] range.
#     This handles both the [-1, 1] normalization and any outliers.
#     """
#     # First clip the values to a reasonable range
#     image = tf.clip_by_value(image, -1.0, 1.0)
#     # Convert from [-1, 1] to [0, 1]
#     image = (image + 1.0) / 2.0
#     # Convert to [0, 255] range and ensure uint8 type
#     # I dont like that this rounds the values, but it's the simplest way
#     image = tf.cast(image * 255, tf.uint8)
#     return image
#
# for i, (image, label) in enumerate(ds):
#     print(f"\nProcessing image {i}:")
#     print(f"Image shape: {image.shape}")
#     print(f"Image value range: [{tf.reduce_min(image):.2f}, {tf.reduce_max(image):.2f}]")
#     print(f"Label: {label.numpy()}")
#
#     save_image = denormalize_image(image)
#     save_path = str(Path("./test") / f"image_{i}.png")
#     fig, ax = plt.subplots()
#     ax.imshow(save_image.numpy())
#     ax.set_title(str(label.numpy()))
#     ax.axis('off')
#     fig.savefig(save_path, bbox_inches='tight')
#     plt.close(fig)
#
#     # tf.keras.utils.save_img(save_path, save_image.numpy())
#
#

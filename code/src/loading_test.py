import tensorflow as tf
import numpy as np
from pathlib import Path
import os
import math
import argparse
import time

from model_definitions.cmadgan.cmadgan import CMADGAN # Replace with your actual import path


from model_definitions.discriminators.cmadgan_mnists.disc import define_discriminator 
from model_definitions.generators.cmadgan_mnists.gen import define_generators 



def generate_images(
    model_weights_path: str,
    output_base_dir: str,
    latent_dim: int,
    condition_dim: int,
    num_generators: int,
    data_shape: tuple,
    define_generator_func, # Function to build generator architecture
    define_discriminator_func, # Function to build discriminator architecture
    num_images_per_class: int = 5000,
    batch_size: int = 64,
):
    """
    Loads a CMADGAN model and generates images for each class from each generator.

    Args:
        model_weights_path: Path to the saved .weights.h5 file.
        output_base_dir: Directory to save the generated images.
        latent_dim: Latent dimension of the model.
        condition_dim: Number of classes (conditions).
        num_generators: Number of generators in the model.
        data_shape: Shape of the output images (H, W, C).
        define_generator_func: Function used to define generator architecture during training.
        define_discriminator_func: Function used to define discriminator architecture during training.
        num_images_per_class: Number of images to generate per class per generator.
        batch_size: Batch size for generation.
    """
    print("--- Starting Image Generation ---")
    print(f"Model Weights: {model_weights_path}")
    print(f"Output Dir: {output_base_dir}")
    print(f"Latent Dim: {latent_dim}, Condition Dim: {condition_dim}")
    print(f"Num Generators: {num_generators}, Data Shape: {data_shape}")
    print(f"Images per Class/Gen: {num_images_per_class}, Batch Size: {batch_size}")

    output_base_path = Path(output_base_dir)
    model_weights_path = Path(model_weights_path)

    if not model_weights_path.is_file():
        print(f"ERROR: Model weights file not found at {model_weights_path}")
        return

    # 1. Instantiate the Model Structure
    print("\nInstantiating model structure...")
    try:
        # Instantiate with the *exact same* parameters and architecture functions used during training
        # diversity_weight doesn't matter for generation
        model = CMADGAN(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            data_shape=data_shape,
            num_generators=num_generators,
            diversity_weight=0.0, # Not used in generation
            define_generator=define_generator_func,
            define_discriminator=define_discriminator_func
        )
        # Perform a dummy generation pass to ensure the model is built before loading weights
        # This helps if the model uses subclassing and build isn't called explicitly
        print("Building model with dummy call...")
        dummy_noise = tf.zeros((1, latent_dim))
        dummy_cond = tf.zeros((1, condition_dim))
        _ = model.generate(dummy_noise, dummy_cond)
        print("Model structure instantiated and built.")

    except Exception as e:
        print(f"ERROR: Failed to instantiate model structure: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Load the Weights
    print(f"Loading weights from {model_weights_path}...")
    try:
        model.load_weights(str(model_weights_path)) # Ensure path is string
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load weights. Ensure the model structure matches the saved weights: {e}")
        import traceback
        traceback.print_exc()
        # Often fails if architecture (layers, names) doesn't match *exactly*.
        print("\n--- Generator Summaries (for debugging mismatch) ---")
        for i, gen in enumerate(model.generators):
             print(f"\n--- Generator {i} ---")
             gen.summary()
        return

    # 3. Loop through Generators and Classes to Generate Images
    start_time_total = time.time()
    total_images_generated = 0

    for gen_nr in range(num_generators):
        print(f"\n--- Processing Generator {gen_nr}/{num_generators-1} ---")
        start_time_gen = time.time()

        for class_idx in range(condition_dim):
            print(f"  Generating images for Class {class_idx}/{condition_dim-1}...")
            start_time_class = time.time()

            # Create specific output directory
            class_output_dir = output_base_path / f"generator_{gen_nr}" / f"class_{class_idx}"
            class_output_dir.mkdir(parents=True, exist_ok=True)

            images_saved_count = 0
            num_batches = math.ceil(num_images_per_class / batch_size)

            for batch_nr in range(num_batches):
                current_batch_size = min(batch_size, num_images_per_class - images_saved_count)
                if current_batch_size <= 0:
                    break

                # Generate noise and condition for the current batch
                noise = tf.random.normal(shape=(current_batch_size, latent_dim))
                conditions_int = tf.constant([class_idx] * current_batch_size, dtype=tf.int32)
                conditions = tf.one_hot(conditions_int, depth=condition_dim)

                # Generate image batch
                # model.generate returns a list of tensors (one per generator)
                generated_batches_all_gens = model.generate(noise, conditions)
                generated_batch_tensor = generated_batches_all_gens[gen_nr] # Get output for current generator
                generated_batch_np = generated_batch_tensor.numpy()

                # De-normalize: [-1, 1] -> [0, 1] -> [0, 255] uint8
                generated_batch_np = (np.clip(generated_batch_np, -1.0, 1.0) + 1.0) / 2.0 * 255.0
                generated_batch_uint8 = generated_batch_np.astype(np.uint8)

                # Save individual images
                for i in range(current_batch_size):
                    img_idx_in_class = images_saved_count + i
                    img_to_save = generated_batch_uint8[i] # Shape (H, W, C)

                    # Handle grayscale saving if needed (PIL expects (H,W) or (H,W,C))
                    #if img_to_save.shape[-1] == 1:
                    #    img_to_save = np.squeeze(img_to_save, axis=-1)

                    save_path = class_output_dir / f"image_{img_idx_in_class:05d}.png"
                    tf.keras.utils.save_img(save_path, img_to_save, data_format='channels_last', scale=False) # scale=False as already in [0,255]

                images_saved_count += current_batch_size
                total_images_generated += current_batch_size

                if (batch_nr + 1) % 20 == 0: # Print progress every 20 batches
                    print(f"    Batch {batch_nr+1}/{num_batches} done. Saved {images_saved_count}/{num_images_per_class} images.")

            class_time = time.time() - start_time_class
            print(f"  Finished Class {class_idx}. Saved {images_saved_count} images in {class_time:.2f} seconds.")

        gen_time = time.time() - start_time_gen
        print(f"--- Finished Generator {gen_nr} in {gen_time:.2f} seconds ---")

    total_time = time.time() - start_time_total
    print("\n--- Image Generation Complete ---")
    print(f"Total images generated: {total_images_generated}")
    print(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from a trained CMADGAN model.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the saved model weights (.weights.h5 file).")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save generated images.")
    parser.add_argument("--latent_dim", type=int, required=True, help="Latent dimension used during training.")
    parser.add_argument("--cond_dim", type=int, required=True, help="Number of classes (condition dimension).")
    parser.add_argument("--num_gen", type=int, required=True, help="Number of generators in the model.")
    # Add arguments for data shape (important for different datasets)
    parser.add_argument("--height", type=int, required=True, help="Height of the images.")
    parser.add_argument("--width", type=int, required=True, help="Width of the images.")
    parser.add_argument("--channels", type=int, required=True, help="Number of image channels (1 for grayscale, 3 for color).")
    parser.add_argument("--num_images", type=int, default=5000, help="Number of images to generate per class per generator.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for generation.")
    # Add arguments for potentially different architecture definition functions if needed,
    # otherwise assume they are imported directly as shown above.

    args = parser.parse_args()

    data_shape = (args.height, args.width, args.channels)

    gen_func = define_generators
    disc_func = define_discriminator

    # ---

    generate_images(
        model_weights_path=args.weights,
        output_base_dir=args.output_dir,
        latent_dim=args.latent_dim,
        condition_dim=args.cond_dim,
        num_generators=args.num_gen,
        data_shape=data_shape,
        define_generator_func=gen_func,
        define_discriminator_func=disc_func,
        num_images_per_class=args.num_images,
        batch_size=args.batch_size,
    )
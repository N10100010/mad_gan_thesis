import tensorflow as tf
import numpy as np
from pathlib import Path
import os
import math
import argparse
import time
import json # <--- Import json module

# Assume these imports point to the correct, potentially modified, definitions
# that expect integer labels if you adapted your model previously.
from model_definitions.cmadgan.cmadgan import CMADGAN # Replace with your actual import path
from model_definitions.discriminators.cmadgan_mnists.disc import define_discriminator # Example path
from model_definitions.generators.cmadgan_mnists.gen import define_generator # Example path (renamed for clarity below)


def generate_images(
    model_weights_path: str,
    output_base_dir: str, # Renamed: Directory where 'generated_images' and 'labels.json' will be placed
    latent_dim: int,
    # --- Renamed and Added Parameters ---
    num_classes: int,       # Renamed from condition_dim
    embedding_dim: int,     # Added: Embedding dimension used during training
    # --- End Renamed ---
    num_generators: int,
    data_shape: tuple,
    define_generator_func,    # Function to build generator architecture
    define_discriminator_func,# Function to build discriminator architecture
    num_images_per_class: int = 5000,
    batch_size: int = 64,
):
    """
    Loads a CMADGAN model and generates images for each class from each generator,
    saving them into a flat directory structure and creating a labels.json file.
    Assumes the CMADGAN model expects integer labels for conditions.

    Args:
        model_weights_path: Path to the saved .weights.h5 file.
        output_base_dir: Base directory where 'generated_images' folder and 'labels.json' will be created.
        latent_dim: Latent dimension of the model.
        num_classes: Number of classes (conditions).
        embedding_dim: Embedding dimension used inside the models.
        num_generators: Number of generators in the model.
        data_shape: Shape of the output images (H, W, C).
        define_generator_func: Function used to define generator architecture during training.
        define_discriminator_func: Function used to define discriminator architecture during training.
        num_images_per_class: Number of images to generate per class per generator.
        batch_size: Batch size for generation.
    """
    print("--- Starting Image Generation ---")
    print(f"Model Weights: {model_weights_path}")
    print(f"Output Base Dir: {output_base_dir}")
    print(f"Latent Dim: {latent_dim}, Num Classes: {num_classes}, Embedding Dim: {embedding_dim}")
    print(f"Num Generators: {num_generators}, Data Shape: {data_shape}")
    print(f"Images per Class/Gen: {num_images_per_class}, Batch Size: {batch_size}")

    output_base_path = Path(output_base_dir)
    model_weights_path = Path(model_weights_path)

    # --- MODIFIED: Define the single flat output directory for images ---
    image_output_dir = output_base_path / "generated_images"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to: {image_output_dir}")
    # --- END MODIFICATION ---

    # --- MODIFIED: Initialize dictionary for labels.json ---
    labels_data = {}
    # --- END MODIFICATION ---


    if not model_weights_path.is_file():
        print(f"ERROR: Model weights file not found at {model_weights_path}")
        return

    # 1. Instantiate the Model Structure
    print("\nInstantiating model structure...")
    try:
        # Instantiate with the *exact same* parameters and architecture functions used during training
        model = CMADGAN(
            latent_dim=latent_dim,
            # --- MODIFIED: Pass num_classes and embedding_dim ---
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            # --- END MODIFICATION ---
            data_shape=data_shape,
            num_generators=num_generators,
            diversity_weight=0.0, # Not used in generation
            define_generator=define_generator_func,
            define_discriminator=define_discriminator_func
        )
        # Perform a dummy generation pass to ensure the model is built
        print("Building model with dummy call...")
        dummy_noise = tf.zeros((1, latent_dim))
        # --- MODIFIED: Use integer label for dummy call ---
        dummy_cond_int = tf.zeros((1, 1), dtype=tf.int32) # Shape (batch, 1)
        _ = model.generate(dummy_noise, dummy_cond_int)
         # --- END MODIFICATION ---
        print("Model structure instantiated and built.")

    except Exception as e:
        print(f"ERROR: Failed to instantiate model structure: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Load the Weights
    print(f"Loading weights from {model_weights_path}...")
    try:
        # Ensure path is a string for load_weights
        model.load_weights(str(model_weights_path))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load weights. Ensure the model structure matches the saved weights: {e}")
        import traceback
        traceback.print_exc()
        # Print summaries for debugging mismatches
        print("\n--- Generator Summaries (for debugging mismatch) ---")
        for i, gen in enumerate(model.generators):
             print(f"\n--- Generator {i} ---")
             gen.summary(line_length=100) # Add line_length for potentially wider models
        print("\n--- Discriminator Summary ---")
        model.discriminator.summary(line_length=100)
        return

    # 3. Loop through Generators and Classes to Generate Images
    start_time_total = time.time()
    total_images_generated = 0
    # --- MODIFIED: Keep track of absolute image index for unique filenames if needed ---
    # absolute_image_counter = 0 # Alternative if batch/img in batch is not desired

    for gen_nr in range(num_generators):
        print(f"\n--- Processing Generator {gen_nr}/{num_generators-1} ---")
        start_time_gen = time.time()

        for class_idx in range(num_classes): # Use num_classes here
            print(f"  Generating images for Class {class_idx}/{num_classes-1}...")
            start_time_class = time.time()

            # --- REMOVED: Nested directory creation ---
            # class_output_dir = output_base_path / f"generator_{gen_nr}" / f"class_{class_idx}"
            # class_output_dir.mkdir(parents=True, exist_ok=True)
            # --- END REMOVAL ---

            images_generated_for_class_count = 0 # Track generated images for this class/gen combo
            num_batches = math.ceil(num_images_per_class / batch_size)

            for batch_nr in range(num_batches):
                current_batch_size = min(batch_size, num_images_per_class - images_generated_for_class_count)
                if current_batch_size <= 0:
                    break

                # Generate noise and condition for the current batch
                noise = tf.random.normal(shape=(current_batch_size, latent_dim))
                # --- MODIFIED: Use INTEGER labels (shape: [batch, 1]) ---
                conditions_int = tf.constant([[class_idx]] * current_batch_size, dtype=tf.int32) # Shape (batch_size, 1)
                # Remove one-hot encoding:
                # conditions = tf.one_hot(conditions_int, depth=num_classes) # OLD
                # --- END MODIFICATION ---

                # Generate image batch using integer conditions
                generated_batches_all_gens = model.generate(noise, conditions_int) # Pass integer labels
                generated_batch_tensor = generated_batches_all_gens[gen_nr] # Get output for current generator
                generated_batch_np = generated_batch_tensor.numpy()

                # De-normalize: [-1, 1] -> [0, 1] -> [0, 255] uint8
                generated_batch_np = (np.clip(generated_batch_np, -1.0, 1.0) + 1.0) / 2.0 * 255.0
                generated_batch_uint8 = generated_batch_np.astype(np.uint8)

                # Save individual images
                for i in range(current_batch_size):
                    img_to_save = generated_batch_uint8[i] # Shape (H, W, C)

                    # --- MODIFIED: Filename Format ---
                    # Old format used index within class: img_idx_in_class = images_generated_for_class_count + i
                    # New format uses gen_nr, batch_nr, and index within batch (i)
                    filename = f"gen_{gen_nr}_batch_{batch_nr}_img_{i}.png"
                    # --- END MODIFICATION ---

                    # --- MODIFIED: Save Path (use flat directory) ---
                    save_path = image_output_dir / filename # Save directly in image_output_dir
                    # --- END MODIFICATION ---

                    # Handle grayscale saving if needed (PIL expects (H,W) or (H,W,C))
                    if img_to_save.shape[-1] == 1:
                        # Squeeze if single channel (H,W,1) -> (H,W) for saving
                        img_to_save_squeezed = np.squeeze(img_to_save, axis=-1)
                        tf.keras.utils.save_img(save_path, img_to_save_squeezed, data_format='channels_last', scale=False)
                    else:
                         # Save RGB directly
                        tf.keras.utils.save_img(save_path, img_to_save, data_format='channels_last', scale=False) # scale=False as already in [0,255]

                    # --- MODIFIED: Add entry to labels dictionary ---
                    labels_data[filename] = class_idx
                    # --- END MODIFICATION ---

                    # absolute_image_counter += 1 # Alternative indexing

                images_generated_for_class_count += current_batch_size
                total_images_generated += current_batch_size

                if (batch_nr + 1) % 20 == 0: # Print progress every 20 batches
                    print(f"    Batch {batch_nr+1}/{num_batches} done. Saved {images_generated_for_class_count}/{num_images_per_class} images for this class.")

            class_time = time.time() - start_time_class
            print(f"  Finished Class {class_idx}. Generated {images_generated_for_class_count} images in {class_time:.2f} seconds.")

        gen_time = time.time() - start_time_gen
        print(f"--- Finished Generator {gen_nr} in {gen_time:.2f} seconds ---")

    # --- MODIFIED: Save the labels dictionary as JSON ---
    labels_json_path = output_base_path / "labels.json"
    print(f"\nSaving label dictionary to {labels_json_path}...")
    try:
        with open(labels_json_path, 'w') as f:
            json.dump(labels_data, f, indent=4)
        print("labels.json saved successfully.")
    except Exception as e:
        print(f"ERROR: Failed to save labels.json: {e}")
    # --- END MODIFICATION ---

    total_time = time.time() - start_time_total
    print("\n--- Image Generation Complete ---")
    print(f"Total images generated: {total_images_generated}")
    print(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    # Assuming your project structure allows these relative imports
    # Adjust if your 'model_definitions' is elsewhere or needs to be added to PYTHONPATH
    try:
        # Example: Assuming generators/discriminators are defined for MNIST-like 28x28 data
        from model_definitions.generators.cmadgan_mnists.gen import define_generator as define_generator_mnist_like
        from model_definitions.discriminators.cmadgan_mnists.disc import define_discriminator as define_discriminator_mnist_like
    except ImportError:
        print("ERROR: Could not import default generator/discriminator definitions.")
        print("Please ensure model_definitions paths are correct.")
        # Define dummy functions if needed for testing script structure
        def define_generator_mnist_like(*args, **kwargs): raise NotImplementedError("Generator definition missing")
        def define_discriminator_mnist_like(*args, **kwargs): raise NotImplementedError("Discriminator definition missing")
        # exit(1) # Optional: exit if definitions are missing


    base_output_dir_root = Path("./experiments/CMADGAN_MODELS_DATACREATION")
    experiments_dir = Path("./experiments/CMADGAN_MODELS_PROTOTYPES/MNIST")

    if not experiments_dir.is_dir():
        print(f"ERROR: Experiments directory not found at {experiments_dir}")
        exit(1)

    experiments = [d.name for d in experiments_dir.iterdir() if d.is_dir()] # Get only directories

    print(experiments)
    exit(1)


    # --- Configuration (Should match training config for the loaded weights!) ---
    # These need to be correct for the weights you are loading
    LATENT_DIM = 128
    NUM_CLASSES = 10      # e.g., CIFAR-10 or MNIST
    EMBEDDING_DIM = 64    # IMPORTANT: Set this to the embedding dim used during training!
    DATA_SHAPE = (28, 28, 1) # IMPORTANT: Set for the target output (e.g., 28x28 grayscale)
    # NUM_GENERATORS will be read from folder name below

    # --- Generation Settings ---
    NUM_IMAGES_PER_CLASS = 5000 # How many images to generate
    BATCH_SIZE = 64        # Batch size for generation process

    print(f"Found experiments: {experiments}")

    for exp in experiments:
        print(f"\nProcessing experiment: {exp}")
        try:
            # --- Attempt to parse experiment name ---
            # Example name format: CMADGAN_DATASET_ngen_3_...
            parts = exp.split("_")
            ds_name = parts[1]
            gen_count_index = parts.index("ngen") + 1
            n_gen_str = parts[gen_count_index]
            n_gen = int(n_gen_str) # Convert to int
            # --- End Parsing ---

            print(f"  Dataset: {ds_name}, Num Generators: {n_gen}")

            # --- MODIFIED: Correct weights filename typo ---
            weights_path = experiments_dir / exp / "final_model.weights.h5" # Corrected typo
            # --- END MODIFICATION ---

            # --- MODIFIED: Define output base for this experiment ---
            # The function will create 'generated_images' and 'labels.json' inside this
            output_dir_exp = base_output_dir_root / f"{ds_name}" / f"CMADGAN_{n_gen}_GEN"
             # --- END MODIFICATION ---


            # --- Select correct model definition functions ---
            # You might need logic here if different experiments used different architectures
            define_gen_func_to_use = define_generator_mnist_like
            define_disc_func_to_use = define_discriminator_mnist_like
            print(f"  Using generator function: {define_gen_func_to_use.__name__}")
            print(f"  Using discriminator function: {define_disc_func_to_use.__name__}")


            if not weights_path.is_file():
                print(f"  WARNING: Weights file not found for experiment {exp} at {weights_path}. Skipping.")
                continue

            generate_images(
                model_weights_path=str(weights_path), # Pass as string
                output_base_dir=str(output_dir_exp), # Pass as string
                latent_dim=LATENT_DIM,
                num_classes=NUM_CLASSES,            # Pass num_classes
                embedding_dim=EMBEDDING_DIM,        # Pass embedding_dim
                num_generators=n_gen,               # Parsed from folder name
                data_shape=DATA_SHAPE,              # Use defined data shape
                define_generator_func=define_gen_func_to_use, # Pass correct function
                define_discriminator_func=define_disc_func_to_use,# Pass correct function
                num_images_per_class=NUM_IMAGES_PER_CLASS,
                batch_size=BATCH_SIZE,
            )

        except ValueError as ve:
             print(f"  ERROR: Could not parse experiment name '{exp}' correctly: {ve}. Skipping.")
             continue # Skip to next experiment
        except Exception as e:
             print(f"  ERROR: Failed processing experiment {exp}: {e}")
             import traceback
             traceback.print_exc()
             continue # Skip to next experiment


    print("\nScript finished.")
    # Removed exit(0) to allow script to finish naturally
import tensorflow as tf
# Assuming your model definition is correctly imported
from model_definitions.classifiers.cifar10 import CIFAR10Classifier
from model_definitions.classifiers.mnist import MNISTClassifier
from model_definitions.classifiers.fashion_mnist import FashionMNISTClassifier

# generators: 
from model_definitions.generators.cmadgan_mnists.gen import define_generators as define_cmadgan_mnists_gen
from model_definitions.generators.madgan_mnists.gen import define_generators as define_madgan_mnists_gen
from model_definitions.generators.conditional_mnists.gen import define_generator as define_conditional_mnists_gen
from model_definitions.generators.vanilla_mnist.gen import define_generator as define_vanilla_mnist_gen
from model_definitions.generators.vanilla_fashion_mnist.gen import define_generator as define_vanilla_fashion_mnist_gen

from model_definitions.discriminators.cmadgan_mnists.disc import define_discriminator as define_cmadgan_mnists_disc
from model_definitions.discriminators.madgan_mnists.disc import define_discriminator as define_madgan_mnists_disc
from model_definitions.discriminators.conditional_mnists.disc import define_discriminator as define_conditional_mnists_disc
from model_definitions.discriminators.vanilla_mnist.disc import define_discriminator as define_vanilla_mnist_disc
from model_definitions.discriminators.vanilla_fashion_mnist.disc import define_discriminator as define_vanilla_fashion_mnist_disc



import visualkeras
from PIL import ImageFont # Optional: For font customization
import os # To check font file existence

classifier_instance = FashionMNISTClassifier(num_classes=10)
output_filename = 'C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\latex\\master_thesis\\abb\\netron_network_archs\\classifying_Classifier_FashionMNIST.png'



# --- Keep your text_callable function definition here ---
def text_callable(layer_index, layer):
    # (Your function definition as provided)
    # Every other piece of text is drawn above the layer, the first one below
    above = bool(layer_index%2)

    # Get the output shape of the layer
    output_shape = []
    try:
        if hasattr(layer, 'output_shape'):
             output_shape = [x for x in list(layer.output_shape) if x is not None]
             # If the output shape is a list of tuples, we only take the first one
             if output_shape and isinstance(output_shape[0], tuple):
                 output_shape = list(output_shape[0])
                 output_shape = [x for x in output_shape if x is not None]
    except Exception: # Broad exception just in case shape access fails unexpectedly
        pass # Leave output_shape as empty list

    # Variable to store text which will be drawn
    output_shape_txt = ""

    # Create a string representation of the output shape
    if len(output_shape) == 1:
         output_shape_txt = str(output_shape[0])
    elif len(output_shape) > 1:
        for ii in range(len(output_shape)):
            output_shape_txt += str(output_shape[ii])
            if ii < len(output_shape) - 2: # Add an x between dimensions, e.g. 3x3
                output_shape_txt += "x"
            if ii == len(output_shape) - 2: # Add a newline between the last two dimensions, e.g. 3x3 \n 64
                output_shape_txt += "\n"

    # Add the name of the layer to the text, as a new line
    layer_name = layer.name if hasattr(layer, 'name') and layer.name else layer.__class__.__name__
    if layer_name.startswith(layer.__class__.__name__.lower()): # Use class name if default name
        layer_name = layer.__class__.__name__

    if output_shape_txt:
        output_shape_txt += f"\n{layer_name}"
    else:
        output_shape_txt = layer_name # Fallback to just name if shape is empty/invalid

    # Return the text value and if it should be drawn above the layer
    return output_shape_txt, above

def create_image(name, model): 
    output_filename = f'C:\\Users\\NiXoN\\Desktop\\_thesis\\mad_gan_thesis\\latex\\master_thesis\\abb\\netron_network_archs\\{name}.png'
# --- Instantiate your model ---
    try:
        print(f"Instantiating {name} model...")
        # Ensure CIFAR10Classifier is correctly defined/imported before this line
        model_to_visualize = model

        # --- Optional: Font setup (as before) ---
        font = None
        # ... (font loading code if needed) ...

        # --- Generate visualization ---
        # Use an absolute path or ensure the directory exists
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        print(f"Generating visualization using visualkeras to {output_filename}...")

        visualkeras.layered_view(
            model_to_visualize,
            legend=True,
            to_file=output_filename,
            text_callable=text_callable,        # <<< Use the correct 'draw_text' argument
            font=font,
            spacing=20,
            padding=50,
            scale_xy=10,
            scale_z=1.5,
            min_z=20,
            min_xy=20,
            max_xy=100,
            max_z=100
        )

        print(f"Architecture visualization saved successfully to: {output_filename}")


    except NameError as e:
         print(f"ERROR: Could not instantiate model or find definition. Is CIFAR10Classifier correctly defined/imported? Details: {e}")
    except ImportError as e:
         print(f"ERROR: visualkeras or Pillow might not be installed. Install using 'pip install visualkeras Pillow'. Details: {e}")
    except AttributeError as e:
         print(f"ERROR: Problem accessing model attributes. Is the model definition correct? Details: {e}")
    except FileNotFoundError as e:
         print(f"ERROR: Output directory path seems invalid or cannot be created. Check path: {output_filename}. Details: {e}")
    except OSError as e:
         print(f"ERROR accessing font file (if specified) or writing PNG. Check permissions/path. Details: {e}")
    except Exception as e:
         print(f"ERROR: An unexpected error occurred: {e}")
         # import traceback
         # traceback.print_exc()
         
if __name__ == "__main__":

    models = {
        "define_cmadgan_mnists_gen": define_cmadgan_mnists_gen(100, 10, (28,28,1)),
        "define_madgan_mnists_gen": define_madgan_mnists_gen(1, 256)[0],
        "define_conditional_mnists_gen": define_conditional_mnists_gen(),
        "define_vanilla_mnist_gen": define_vanilla_mnist_gen(),
        "define_vanilla_fashion_mnist_gen": define_vanilla_fashion_mnist_gen(),
        "define_cmadgan_mnists_disc": define_cmadgan_mnists_disc((28,28,1), 10),
        "define_madgan_mnists_disc": define_madgan_mnists_disc(3),
        "define_conditional_mnists_disc": define_conditional_mnists_disc(),
        "define_vanilla_mnist_disc": define_vanilla_mnist_disc(),
        "define_vanilla_fashion_mnist_disc": define_vanilla_fashion_mnist_disc(),
    }
    
    for name, model in models.items():
        create_image(name,model)
        print(f"Model {name} created successfully.")
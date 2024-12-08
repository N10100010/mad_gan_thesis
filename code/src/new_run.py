# Initialize the manager with necessary parameters and functions
from experimenter.manager import ExperimentManager
from datasets.mnist import dataset_func
from latent_points.mnist import generate_latent_points
from model_definitions.discriminators.mnist.disc import define_discriminator
from model_definitions.generators.mnist.gen import define_generators
from model_definitions.mad_gan.mnist import MADGAN
from monitors.generator import GANMonitor
from run import Generators_loss_function


manager = ExperimentManager(
    n_gen=2,
    latent_dim=256,
    batch_size=256,
    size_dataset=60_000,
    epochs=3,
    experiment_type="no-stack",
    dataset_func=dataset_func,
    define_discriminator=define_discriminator,
    define_generators=define_generators,
    MADGAN=MADGAN,
    Generators_loss_function=Generators_loss_function,
    GANMonitor=GANMonitor,
    generate_latent_points=generate_latent_points,
)

# Run the experiment
manager._check_gpu()
manager.load_data()
manager.initialize_models()
manager.run_experiment()

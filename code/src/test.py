import matplotlib.pyplot as plt
import numpy as np
import json 

def plot_scores(data):
    # Convert string keys to integers if necessary
    data = {
        int(epoch): {
            int(gen): values for gen, values in gen_data.items()
        }
        for epoch, gen_data in data.items()
    }
    
    epochs = sorted(data.keys())
    num_generators = len(next(iter(data.values())))  # Assumes all epochs have the same number of generators
    
    # FID Scores Plot (Single Figure)
    plt.figure(figsize=(10, 5))
    for gen in range(num_generators):
        fid_scores = [data[epoch][gen]["FID"] for epoch in epochs]
        plt.plot(epochs, fid_scores, marker="o", label=f"Generator {gen}")
    
    plt.xlabel("Epochs")
    plt.ylabel("FID Score")
    plt.title("FID Scores Over Epochs")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("/home/stud/n/nr063/mounted_home/mad_gan_thesis/code/experiments/CMADGAN_MODELS_PROTOTYPES/MNIST/2025-04-02_TEST__MNIST_CMADGAN_Experiment___latent_100_1_gen_250_epochs/scores/FID.png")

    plt.close()

    # IS Scores Plot (Subplots for each Generator)
    fig, axes = plt.subplots(num_generators, 1, figsize=(10, 5 * num_generators), sharex=True)

    if num_generators == 1:
        axes = [axes]  # Ensure axes is iterable even for one generator

    for gen in range(num_generators):
        is_scores = [data[epoch][gen]["IS"] for epoch in epochs]
        is_stds = [data[epoch][gen]["IS_std"] for epoch in epochs]

        axes[gen].plot(epochs, is_scores, marker="o", label=f"Generator {gen}", color=f"C{gen}")
        axes[gen].fill_between(epochs, np.array(is_scores) - np.array(is_stds),
                               np.array(is_scores) + np.array(is_stds), alpha=0.2, color=f"C{gen}")

        axes[gen].set_ylabel("IS Score")
        axes[gen].set_title(f"IS Score - Generator {gen}")
        axes[gen].legend()
        axes[gen].grid(True)

    axes[-1].set_xlabel("Epochs")
    plt.tight_layout()
    plt.savefig("/home/stud/n/nr063/mounted_home/mad_gan_thesis/code/experiments/CMADGAN_MODELS_PROTOTYPES/MNIST/2025-04-02_TEST__MNIST_CMADGAN_Experiment___latent_100_1_gen_250_epochs/scores/IS.png")



with open("/home/stud/n/nr063/mounted_home/mad_gan_thesis/code/experiments/CMADGAN_MODELS_PROTOTYPES/MNIST/2025-04-02_TEST__MNIST_CMADGAN_Experiment___latent_100_1_gen_250_epochs/scores/metrics.json", "r") as f:
   data = json.load(f)
plot_scores(data)
# plot_model(model, to_file=".\model.png", show_shapes=True, show_layer_names=True)
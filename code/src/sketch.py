from experiment.experiments.mnist_madgan import MNIST_MADGAN_Experiment

if __name__ == "__main__":
    experiment = MNIST_MADGAN_Experiment("MNIST_MADGAN_Experiment")
    experiment.run()
    
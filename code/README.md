# TODO

- answer quest: Is one generator better than the other, given a specific class to be generated?
  - Given the fact that specific generators should converge to specifc modes in the data dist, each generator should be 'better' in its specialized modes, than in those it did not specialize in
    - This hypothesis holds true for the case `n_gen < n_classes`
    - For the case `n_gen == n_classes`, there should presumably be one generator that is 'best' for a specific class
  - how can we identify the modes a generator is 'best' in recreating?
    - Can we utilize the FID-score of generated images, compared to real images? For a 'good' image of a certain class, the FID-score should be bigger, given images of the same class, than the resulting score for images of a different class. Right?

## check

### server

https://deeplearn.mi.hdm-stuttgart.de/user/nr063/lab

### how to

https://deeplearn.pages.mi.hdm-stuttgart.de/docs/quickstart/

# Datasets

All datasets can be loaded

- mnist
  - https://www.tensorflow.org/datasets/catalog/mnist
- fashion mnist
  - https://www.tensorflow.org/datasets/catalog/fashion_mnist
- cifar10
  - https://www.tensorflow.org/datasets/catalog/cifar10

# Notes

## Conda Environment

[local environment](./environment.yml):
`conda env create -f environment.yml`

[server environment](./server_env.yml):
`conda env create -f server_env.yml`

to create a conda environment called `__env`

## python environment

Add src to the python path
**WIN**

`set PYTHONPATH=%PYTHONPATH%;C:\Users\NiXoN\Desktop\_thesis\mad_gan_thesis\code\src`

## Load a past madgan experiment

To load a past experiment, one hast to:

- load the experiment from its saved path
- load the models weights from the afore given path

```
experiment = FASHION_MNIST_MADGAN_Experiment.load_from_path(
    Path(
        "experiments\\2024-12-22_FASHION_MNIST_MADGAN_Experiment__1_n_gen_1"
    )
)

experiment.load_model_weights()
```

After that, the MADGAN is initialized, with its two submodels (discriminator and n-generators). Also, the generators can be used to create images they were trained on, after laoding the weights.

```
from latent_points.utils import generate_latent_points
from matplotlib import pyplot as plt
latent_points = generate_latent_points(
    latent_dim=experiment.latent_dim, batch_size=experiment.batch_size, n_gen=experiment.n_gen
)

generators = experiment.madgan.generators
generated_images = []
for g in range(experiment.n_gen):
    generated_images.append(generators[g](latent_points[g]))
for image in generated_images:
    print(type(image))
    plt.imshow(image[0])
    plt.show()
```

# TODO CODE

- implement the classification with images from the different architecture

  - should this also be an experiment?! No reason to track any metrics...
    - Q: How do we save the classified information, including the images

- inception scores:
  - mnist: https://github.com/ChunyuanLI/MNIST_Inception_Score
  - fashion mnist: https://www.kaggle.com/code/babbler/inception-model-for-fashion-mnist
  - cifar10: https://github.com/Ahmed-Habashy/calculate-Inception-Score-for-CIFAR-10-in-Keras

## windows hacks:

- hold down Windows Key + CTRL + SHIFT + B - this clears your GPU memory

## GAN hacks:

https://github.com/soumith/ganhacks

## check

### server

https://deeplearn.mi.hdm-stuttgart.de/user/nr063/lab

#### How to run the code without having to have it open:

```
python src/run.py 2>&1 | while IFS= read -r line; do
  echo "$line" >> output.log
  tail -n 100 output.log > output.log.tmp
  mv output.log.tmp output.log
done &
```

_As a one liner_
`
python src/run_mad_gan_creation.py 2>&1 | while IFS= read -r line; do   echo "$line" >> output.log;   tail -n 100 output.log > output.log.tmp;   mv output.log.tmp output.log; done &
`

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

## SAVING CHECKPOINTS via callback

when defining an int to decide which modulo should save the model, for MADGAN, you need to combine the steps*per_epch * epochs (e.g. 234 \_ 25 - this will save the madgan model every 25th episode)

## Load a past madgan experiment

**Due to a version mismatch between the local- and the servers version of tensorflow (server: 2.15, local: 2.10), server-models cannot be loaded locally. BUT, loading models on the creating machine works as planned.**

# Literature

- Brief summary, with links, for what is what in the context of training and using a GAN
  - https://github.com/amidstdebug/CIFAR100-GAN/blob/main/CIFAR-100-GAN/research.md
- GAN hacks; a summary of tips for training a GNA
  - https://github.com/soumith/ganhacks

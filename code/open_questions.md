# Open Questions

## Questions to Prof. Maucher

### IS- and FID-Score

_Context_: The model for IS- and FID-score is a classifier (inception model V3), trained on ImageNet dataset.

- For CIFAR10, this is fine
- For MNIST, FashionMNIST however, the data distribution is to different
  _Question_: Use the inception model?
  _Solution_:
- Build own classifer (IS) / feature extractor (FID) for MNIST and FashionMNIST
  - Recommendations?

### Comparison to other GANs?

_Question_: Need to compare to other GANs?
_Options_:

- Wassserstein
- Progressive Growing
- Self-Attention
  _Problem_: MNIST, FashionMNIST are too simple

### Memory Issues with MAD GAN

_Problem_: More complex datasets require big amounts of memory
_Observation_: Paper used the architecture for style transfer. In CIFAR

## General Questions

- answer quest: Is one generator better than the other, given a specific class to be generated?
  - Given the fact that specific generators should converge to specifc modes in the data dist, each generator should be 'better' in its specialized modes, than in those it did not specialize in
    - This hypothesis holds true for the case `n_gen < n_classes`
    - For the case `n_gen == n_classes`, there should presumably be one generator that is 'best' for a specific class
  - how can we identify the modes a generator is 'best' in recreating?
    - Can we utilize the FID-score of generated images, compared to real images? For a 'good' image of a certain class, the FID-score should be bigger, given images of the same class, than the resulting score for images of a different class. Right?

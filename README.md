# Master Thesis - Outline

<p align="center" style="text-align: center;">
Nicolas Reinhart
Under the supervision of
Prof. Dr.-Ing. Johannes Maucher,
Prof. Dr.-Ing. Oliver Kretzschmar
14. November 2024
</p>

During my thesis with the title
<span style="font-size: 20px">_Generative Data Augmentation using Multi-Agent Diverse GAN’s_</span>
, I will explore the utilization of Multi-Agent Diverse
Generative Adversarial Networks (MAD GANs) for the purpose of extending a
training data set for an subsequent classifier on image data.

**Points of Research:**
The classifier will be a simple CNN. The primary aim is not to improve
performance on the datasets mentioned in Challenges, Risks & Solutions
compared to existing best classifiers. Instead, this work serves as a proof of
concept for the potential success of using the MAD GAN architecture for
Generative Domain Adaptation (GDA). Consequently, the datasets are
deliberately chosen to be less intricate, as the complexity, number of
generators, and discriminator sophistication can be scaled as needed.

- Influence of MAD GAN GDA on classifier performanc
- Impact of different loss functions on the generated image
- Effect of the number of generators in the MAD-GAN on created image
- Ratio between real and fake samples for subsequent classifier
- Impact of MAD GAN for GDA on imbalanced datasets
- Comparison between MAD GAN GDA, Vanilla GAN GDA and classical data
  augmentation strategies (cropping, flipping, saturation changes, …)

**Challenges, Risks & Solutions**

1.  Combining the MAD GANs with conditionality constraint of CGANs

    Solutions: Use auxiliary classifier to label images and check manuall

2.  High computational costs with high-definition samples and high number
    of generators

    Solutions: Use low-resolution datasets (e.g. MNIST, Fashion-MNIST, CIFAR-10, …)

3.  Sensitivity to hyperparameters

    Solutions: Automated hyperparameter optimization, curriculum learning, …

4.  Difficulty proving statistical significance of improvemen

    Solutions: Perform statistical significance tests (paired T-test, Wilcoxon signed-rank test - depending the on form of the data)

from typing import Tuple

import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm


def calculate_fid_score(
    real_images: np.ndarray,
    generated_images: np.ndarray,
    classifier: tf.keras.Model,
    n_splits: int = 10,
) -> Tuple[float, float]:
    """
    ORIGINAL PAPER FOR REFERENCE
    Title: "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
    Authors: Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter
    Link: https://arxiv.org/abs/1706.08500

    Args:
        real_images (np.ndarray): Array of real images to be used for FID score calculation.
        generated_images (np.ndarray): Array of generated images to compare against the real images
        classifier (tf.keras.Model): A pre-trained classifier model used to extract features from the images.
        n_splits (int, optional): Number of splits to divide the images into for FID calculation. Defaults to 10.

    Returns:
        Tuple[float, float]: Mean and standard deviation of the FID scores.
    """

    fid_scores = []
    n_samples = real_images.shape[0]
    n_part = n_samples // n_splits

    for i in range(n_splits):
        start_idx = i * n_part
        end_idx = (i + 1) * n_part

        real_subset = real_images[start_idx:end_idx]
        gen_subset = generated_images[start_idx:end_idx]

        feat_real = classifier.predict(real_subset)
        feat_gen = classifier.predict(gen_subset)

        mu_real, sigma_real = (
            np.mean(feat_real, axis=0),
            np.cov(feat_real, rowvar=False),
        )
        mu_gen, sigma_rgen = np.mean(feat_gen, axis=0), np.cov(feat_gen, rowvar=False)

        # sum squared difference = ssd
        ssd = np.sum(np.square(mu_real - mu_gen) ** 2)
        covmean, _ = sqrtm(sigma_real.dot(sigma_rgen), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid_score = ssd + np.trace(sigma_real + sigma_rgen - 2 * covmean)
        fid_scores.append(fid_score)

    mean_fid = np.mean(fid_scores)
    std_fid = np.std(fid_scores)

    return mean_fid, std_fid

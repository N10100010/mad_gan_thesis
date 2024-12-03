import unittest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import MagicMock
from src.utils.plotting import plot_generated_images 

class TestPlotGeneratedImages(unittest.TestCase):
    def test_valid_inputs(self):
        """
        Tests that the function executes without errors when given valid inputs.
        """
        n_rows = 2
        n_cols = 3
        random_latent_vectors = [np.random.rand(10) for _ in range(n_cols - 1)]
        data = np.random.rand(10, 28, 28, 1)
        generators = [MagicMock(return_value=np.random.rand(10, 28, 28, 1)) for _ in range(n_cols - 1)]
        dir_name = 'test_dir'
        epoch = 1
        save = False
        show = False

        plot_generated_images(n_rows, n_cols, random_latent_vectors, data, generators, dir_name, epoch, save, show)

    def test_invalid_inputs(self):
        """
        Tests that the function raises a TypeError when given invalid inputs.
        """
        n_rows = 'a'
        n_cols = 3
        random_latent_vectors = [np.random.rand(10) for _ in range(n_cols - 1)]
        data = np.random.rand(10, 28, 28, 1)
        generators = [MagicMock(return_value=np.random.rand(10, 28, 28, 1)) for _ in range(n_cols - 1)]
        dir_name = 'test_dir'
        epoch = 1
        save = False
        show = False

        with self.assertRaises(TypeError):
            plot_generated_images(n_rows, n_cols, random_latent_vectors, data, generators, dir_name, epoch, save, show)

    def test_save_true(self):
        """
        Tests that the function saves a plot of the generated images to the given directory when 'save' is True.
        """
        n_rows = 2
        n_cols = 3
        random_latent_vectors = [np.random.rand(10) for _ in range(n_cols - 1)]
        data = np.random.rand(10, 28, 28, 1)
        generators = [MagicMock(return_value=np.random.rand(10, 28, 28, 1)) for _ in range(n_cols - 1)]
        dir_name = 'test_dir'
        epoch = 1
        save = True
        show = False

        plot_generated_images(n_rows, n_cols, random_latent_vectors, data, generators, dir_name, epoch, save, show)
        self.assertTrue((dir_name + '/image_at_epoch_0002.png').exists())

    def test_save_false(self):
        """
        Tests that the function does not save a plot of the generated images
        to the given directory when 'save' is False.
        """
        n_rows = 2
        n_cols = 3
        random_latent_vectors = [np.random.rand(10) for _ in range(n_cols - 1)]
        data = np.random.rand(10, 28, 28, 1)
        generators = [MagicMock(return_value=np.random.rand(10, 28, 28, 1)) for _ in range(n_cols - 1)]
        dir_name = 'test_dir'
        epoch = 1
        save = False
        show = False

        plot_generated_images(n_rows, n_cols, random_latent_vectors, data, generators, dir_name, epoch, save, show)
        self.assertFalse((dir_name + '/image_at_epoch_0002.png').exists())

    def test_show_true(self):
        """
        Tests that the function shows the plot of the generated images when 'show' is True.
        """
        
        n_rows = 2
        n_cols = 3
        random_latent_vectors = [np.random.rand(10) for _ in range(n_cols - 1)]
        data = np.random.rand(10, 28, 28, 1)
        generators = [MagicMock(return_value=np.random.rand(10, 28, 28, 1)) for _ in range(n_cols - 1)]
        dir_name = 'test_dir'
        epoch = 1
        save = False
        show = True

        plot_generated_images(n_rows, n_cols, random_latent_vectors, data, generators, dir_name, epoch, save, show)
        self.assertEqual(plt.gcf().number, 1)

    def test_show_false(self):
        """
        Tests that the function does not show the plot of the generated images when 'show' is False.
        """
        n_rows = 2
        n_cols = 3
        random_latent_vectors = [np.random.rand(10) for _ in range(n_cols - 1)]
        data = np.random.rand(10, 28, 28, 1)
        generators = [MagicMock(return_value=np.random.rand(10, 28, 28, 1)) for _ in range(n_cols - 1)]
        dir_name = 'test_dir'
        epoch = 1
       
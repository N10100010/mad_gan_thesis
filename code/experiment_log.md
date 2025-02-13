# CLASSIFIER

## TRAINING

| type    | folder name                                                 | status |
| :------ | :---------------------------------------------------------- | :----- |
| MNSIT   | SERVER:`code/experiments/2025-02-12_CLASSFIER_MNIST`        | done   |
| FASHION | SERVER:`code/experiments/2025-02-12_CLASSFIER_FashionMNIST` | done   |
| CIFAR10 | SERVER:`code/experiments/2025-02-12_CLASSFIER_CIFAR10`      | done   |
| MNSIT   | LOCAL:`code/experiments/2025-02-13_CLASSFIER_MNIST`         | done   |
| FASHION | LOCAL:`code/experiments/2025-02-13_CLASSFIER_FashionMNIST`  | done   |
| CIFAR10 | LOCAL:`code/experiments/2025-02-13_CLASSFIER_CIFAR10`       | done   |

# MADGAN

## TRAINING

### MNIST

| n-gen | folder name                                                               | status |
| :---- | :------------------------------------------------------------------------ | :----- |
| 1     | SERVER:`code/experiments/2024-12-30_MNIST_MADGAN_Experiment__1_n_gen_1`   | done   |
| 2     | SERVER:`code/experiments/2024-12-30_MNIST_MADGAN_Experiment__2_n_gen_2`   | done   |
| 3     | SERVER:`code/experiments/2024-12-30_MNIST_MADGAN_Experiment__3_n_gen_3`   | done   |
| 4     | SERVER:`code/experiments/2024-12-30_MNIST_MADGAN_Experiment__4_n_gen_4`   | done   |
| 5     | SERVER:`code/experiments/2024-12-30_MNIST_MADGAN_Experiment__5_n_gen_5`   | done   |
| 6     | SERVER:`code/experiments/2024-12-31_MNIST_MADGAN_Experiment__6_n_gen_6`   | done   |
| 7     | SERVER:`code/experiments/2024-12-31_MNIST_MADGAN_Experiment__7_n_gen_7`   | done   |
| 8     | SERVER:`code/experiments/2024-12-31_MNIST_MADGAN_Experiment__8_n_gen_8`   | done   |
| 9     | SERVER:`code/experiments/2024-12-31_MNIST_MADGAN_Experiment__9_n_gen_9`   | done   |
| 10    | SERVER:`code/experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10` | done   |

### FASHION MNIST

| n-gen | folder name                                                                       | status |
| :---- | :-------------------------------------------------------------------------------- | :----- |
| 1     | SERVER:`code/experiments/2025-01-01_FASHION_MNIST_MADGAN_Experiment__1_n_gen_1`   | done   |
| 2     | SERVER:`code/experiments/2025-01-01_FASHION_MNIST_MADGAN_Experiment__2_n_gen_2`   | done   |
| 3     | SERVER:`code/experiments/2025-01-01_FASHION_MNIST_MADGAN_Experiment__3_n_gen_3`   | done   |
| 4     | SERVER:`code/experiments/2025-01-01_FASHION_MNIST_MADGAN_Experiment__4_n_gen_4`   | done   |
| 5     | SERVER:`code/experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__5_n_gen_5`   | done   |
| 6     | SERVER:`code/experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__6_n_gen_6`   | done   |
| 7     | SERVER:`code/experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__7_n_gen_7`   | done   |
| 8     | SERVER:`code/experiments/2025-01-03_FASHION_MNIST_MADGAN_Experiment__8_n_gen_8`   | done   |
| 9     | SERVER:`code/experiments/2025-01-03_FASHION_MNIST_MADGAN_Experiment__9_n_gen_9`   | done   |
| 10    | SERVER:`code/experiments/2025-01-03_FASHION_MNIST_MADGAN_Experiment__10_n_gen_10` | done   |

### CIFAR10

| n-gen | folder name                                                             | status |
| :---- | :---------------------------------------------------------------------- | :----- |
| 1     | SERVER:`code/experiments/2025-01-06_CIFAR_MADGAN_Experiment_1_n_gen_1`  | done   |
| 2     | SERVER:`code/experiments/2025-01-06_CIFAR_MADGAN_Experiment__2_n_gen_2` | done   |
| 3     | SERVER:`code/experiments/2025-01-06_CIFAR_MADGAN_Experiment__3_n_gen_3` | done   |
| 4     | SERVER:`TBD`                                                            | TBD    |
| 5     | SERVER:`TBD`                                                            | TBD    |
| 6     | SERVER:`TBD`                                                            | TBD    |
| 7     | SERVER:`TBD`                                                            | TBD    |
| 8     | SERVER:`TBD`                                                            | TBD    |
| 9     | SERVER:`TBD`                                                            | TBD    |
| 10    | SERVER:`TBD`                                                            | TBD    |

## DATA GENERATION

### MNIST

| n-gen | used generator | folder name                                                                      | status | classified | IS score | FID score | note | n-images per class |
| :---- | :------------- | :------------------------------------------------------------------------------- | :----- | :--------- | :------- | :-------- | :--- | :----------------- |
| 1     | 0              | SERVER:`TBD`                                                                     | TBD    | TBD        | TBD      | TBD       | -    | TBD                |
| 3     | 0              | SERVER:`code/experiments/2025-02-08_MADGAN_3_GEN_MNIST_DataCreation_SPEC_GEN_0`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 3     | 1              | SERVER:`code/experiments/2025-02-08_MADGAN_3_GEN_MNIST_DataCreation_SPEC_GEN_1`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 3     | 2              | SERVER:`code/experiments/2025-02-08_MADGAN_3_GEN_MNIST_DataCreation_SPEC_GEN_2`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 0              | SERVER:`code/experiments/2025-02-11_MADGAN_5_GEN_MNIST_DataCreation_SPEC_GEN_0`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 1              | SERVER:`code/experiments/2025-02-11_MADGAN_5_GEN_MNIST_DataCreation_SPEC_GEN_1`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 2              | SERVER:`code/experiments/2025-02-11_MADGAN_5_GEN_MNIST_DataCreation_SPEC_GEN_2`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 3              | SERVER:`code/experiments/2025-02-11_MADGAN_5_GEN_MNIST_DataCreation_SPEC_GEN_3`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 4              | SERVER:`code/experiments/2025-02-11_MADGAN_5_GEN_MNIST_DataCreation_SPEC_GEN_4`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 0              | SERVER:`code/experiments/2025-02-13_MADGAN_7_GEN_MNIST_DataCreation_SPEC_GEN_0`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 1              | SERVER:`code/experiments/2025-02-13_MADGAN_7_GEN_MNIST_DataCreation_SPEC_GEN_1`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 2              | SERVER:`code/experiments/2025-02-13_MADGAN_7_GEN_MNIST_DataCreation_SPEC_GEN_2`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 3              | SERVER:`code/experiments/2025-02-13_MADGAN_7_GEN_MNIST_DataCreation_SPEC_GEN_3`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 4              | SERVER:`code/experiments/2025-02-13_MADGAN_7_GEN_MNIST_DataCreation_SPEC_GEN_4`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 5              | SERVER:`code/experiments/2025-02-13_MADGAN_7_GEN_MNIST_DataCreation_SPEC_GEN_5`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 6              | SERVER:`code/experiments/2025-02-13_MADGAN_7_GEN_MNIST_DataCreation_SPEC_GEN_6`  | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 0              | SERVER:`code/experiments/2025-02-13_MADGAN_10_GEN_MNIST_DataCreation_SPEC_GEN_0` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 1              | SERVER:`code/experiments/2025-02-13_MADGAN_10_GEN_MNIST_DataCreation_SPEC_GEN_1` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 2              | SERVER:`code/experiments/2025-02-13_MADGAN_10_GEN_MNIST_DataCreation_SPEC_GEN_2` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 3              | SERVER:`code/experiments/2025-02-13_MADGAN_10_GEN_MNIST_DataCreation_SPEC_GEN_3` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 4              | SERVER:`code/experiments/2025-02-13_MADGAN_10_GEN_MNIST_DataCreation_SPEC_GEN_4` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 5              | SERVER:`code/experiments/2025-02-13_MADGAN_10_GEN_MNIST_DataCreation_SPEC_GEN_5` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 6              | SERVER:`code/experiments/2025-02-13_MADGAN_10_GEN_MNIST_DataCreation_SPEC_GEN_6` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 7              | SERVER:`code/experiments/2025-02-13_MADGAN_10_GEN_MNIST_DataCreation_SPEC_GEN_7` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 8              | SERVER:`code/experiments/2025-02-13_MADGAN_10_GEN_MNIST_DataCreation_SPEC_GEN_8` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 9              | SERVER:`code/experiments/2025-02-13_MADGAN_10_GEN_MNIST_DataCreation_SPEC_GEN_9` | done   | TBD        | TBD      | TBD       | -    | TBD                |

### FASHION MNIST

| n-gen | used generator | folder name  | status | classified | IS score | FID score | note | n-images per class |
| :---- | :------------- | :----------- | :----- | :--------- | :------- | :-------- | :--- | :----------------- |
| 1     | 0              | SERVER:`TBD` | TBD    | TBD        | TBD      | TBD       | -    | TBD                |
| 3     | 0              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 3     | 1              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 3     | 2              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 0              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 1              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 2              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 3              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 4              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 0              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 1              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 2              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 3              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 4              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 5              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 6              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 0              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 1              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 2              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 3              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 4              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 5              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 6              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 7              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 8              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 9              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |

### CIFAR10

| n-gen | used generator | folder name  | status | classified | IS score | FID score | note | n-images per class |
| :---- | :------------- | :----------- | :----- | :--------- | :------- | :-------- | :--- | :----------------- |
| 1     | 0              | SERVER:`TBD` | TBD    | TBD        | TBD      | TBD       | -    | TBD                |
| 3     | 0              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 3     | 1              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 3     | 2              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 0              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 1              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 2              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 3              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 5     | 4              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 0              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 1              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 2              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 3              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 4              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 5              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 7     | 6              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 0              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 1              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 2              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 3              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 4              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 5              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 6              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 7              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 8              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |
| 10    | 9              | SERVER:`TBD` | done   | TBD        | TBD      | TBD       | -    | TBD                |

# Vanilla GAN

## TRAINING

### MNIST

| folder name | status | classified | IS score | FID score | note | n-images per class |
| :---------- | :----- | :--------- | :------- | :-------- | :--- | :----------------- |

### FASHION MNIST

| folder name | status | classified | IS score | FID score | note | n-images per class |
| :---------- | :----- | :--------- | :------- | :-------- | :--- | :----------------- |

### CIFAR10

| folder name | status | classified | IS score | FID score | note | n-images per class |
| :---------- | :----- | :--------- | :------- | :-------- | :--- | :----------------- |

## DATA GENERATION

### MNIST

| folder name | status | generated data % | note |
| :---------- | :----- | :--------------- | :--- |

### FASHION MNIST

| folder name | status | classified | IS score | FID score | note | n-images per class |
| :---------- | :----- | :--------- | :------- | :-------- | :--- | :----------------- |

### CIFAR10

| folder name | status | classified | IS score | FID score | note | n-images per class |
| :---------- | :----- | :--------- | :------- | :-------- | :--- | :----------------- |

# Generative Data Augmentation

## Classifier

### MNIST

| folder name   | status | generated data % | note |
| :------------ | :----- | :--------------- | :--- |
| SERVER: `TBD` | TBD    | 0                |      |
| SERVER: `TBD` | TBD    | 10               |      |
| SERVER: `TBD` | TBD    | 20               |      |
| SERVER: `TBD` | TBD    | 30               |      |
| SERVER: `TBD` | TBD    | 40               |      |
| SERVER: `TBD` | TBD    | 50               |      |
| SERVER: `TBD` | TBD    | 60               |      |
| SERVER: `TBD` | TBD    | 70               |      |
| SERVER: `TBD` | TBD    | 80               |      |
| SERVER: `TBD` | TBD    | 90               |      |
| SERVER: `TBD` | TBD    | 100              |      |

### FASHION MNIST

| folder name   | status | generated data % | note |
| :------------ | :----- | :--------------- | :--- |
| SERVER: `TBD` | TBD    | 0                |      |
| SERVER: `TBD` | TBD    | 10               |      |
| SERVER: `TBD` | TBD    | 20               |      |
| SERVER: `TBD` | TBD    | 30               |      |
| SERVER: `TBD` | TBD    | 40               |      |
| SERVER: `TBD` | TBD    | 50               |      |
| SERVER: `TBD` | TBD    | 60               |      |
| SERVER: `TBD` | TBD    | 70               |      |
| SERVER: `TBD` | TBD    | 80               |      |
| SERVER: `TBD` | TBD    | 90               |      |
| SERVER: `TBD` | TBD    | 100              |      |

### CIFAR10

| folder name   | status | generated data % | note |
| :------------ | :----- | :--------------- | :--- |
| SERVER: `TBD` | TBD    | 0                |      |
| SERVER: `TBD` | TBD    | 10               |      |
| SERVER: `TBD` | TBD    | 20               |      |
| SERVER: `TBD` | TBD    | 30               |      |
| SERVER: `TBD` | TBD    | 40               |      |
| SERVER: `TBD` | TBD    | 50               |      |
| SERVER: `TBD` | TBD    | 60               |      |
| SERVER: `TBD` | TBD    | 70               |      |
| SERVER: `TBD` | TBD    | 80               |      |
| SERVER: `TBD` | TBD    | 90               |      |
| SERVER: `TBD` | TBD    | 100              |      |

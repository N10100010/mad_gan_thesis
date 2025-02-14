from experiment.experiment_queue import ExperimentQueue
from experiment.experiments.fashion_mnist_madgan.experiment import (
    FASHION_MNIST_MADGAN_Experiment,
)
from experiment.experiments.mnist_madgan.experiment import (
    MNIST_MADGAN_Experiment,
)
from experiment.experiments.generative_creation.madgan.experiment import (
    MADGAN_GenerativeCreationExperiment,
)
from latent_points.utils import generate_latent_points

experiments = [
    ## GENERATE IMAGES FOR THE MNIST DATASET USING N-GENERATORS [1...10] (pretrained)
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_DataCreation_N_GEN_1",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__1_n_gen_1",
    #     latent_point_generator=generate_latent_points,
    #     n_images=1,
    #     save_raw_image=True
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_DataCreation_N_GEN_2",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__2_n_gen_2",
    #     latent_point_generator=generate_latent_points,
    #     n_images=1,
    #     save_raw_image=True
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_DataCreation_N_GEN_3",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__3_n_gen_3",
    #     latent_point_generator=generate_latent_points,
    #     n_images=1,
    #     save_raw_image=True
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_DataCreation_N_GEN_4",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__4_n_gen_4",
    #     latent_point_generator=generate_latent_points,
    #     n_images=1,
    #     save_raw_image=True
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_DataCreation_N_GEN_5",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__5_n_gen_5",
    #     latent_point_generator=generate_latent_points,
    #     n_images=1,
    #     save_raw_image=True
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_DataCreation_N_GEN_6",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__6_n_gen_6",
    #     latent_point_generator=generate_latent_points,
    #     n_images=1,
    #     save_raw_image=True
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_DataCreation_N_GEN_7",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__7_n_gen_7",
    #     latent_point_generator=generate_latent_points,
    #     n_images=1,
    #     save_raw_image=True
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_DataCreation_N_GEN_8",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__8_n_gen_8",
    #     latent_point_generator=generate_latent_points,
    #     n_images=1,
    #     save_raw_image=True
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_DataCreation_N_GEN_9",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__9_n_gen_9",
    #     latent_point_generator=generate_latent_points,
    #     n_images=1,
    #     save_raw_image=True
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_DataCreation_N_GEN_10",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=1,
    #     save_raw_image=True
    # ),

    ## GENERATE IMAGES FOR THE MNIST DATASET USING THE MADGAN WITH 3 GENERATORS, USING A SPECIFIC ONE FOR EACH CREATION
    
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_DataCreation_SPEC_GEN_0",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__3_n_gen_3",
    #     latent_point_generator=generate_latent_points,
    #     n_images=10_000,
    #     save_raw_image=True,
    #     use_generator=0
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_DataCreation_SPEC_GEN_1",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__3_n_gen_3",
    #     latent_point_generator=generate_latent_points,
    #     n_images=10_000,
    #     save_raw_image=True,
    #     use_generator=1
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_DataCreation_SPEC_GEN_2",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__3_n_gen_3",
    #     latent_point_generator=generate_latent_points,
    #     n_images=10_000,
    #     save_raw_image=True,
    #     use_generator=2
    # ),

    
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_5_GEN_DataCreation_SPEC_GEN_0",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__5_n_gen_5",
    #     latent_point_generator=generate_latent_points,
    #     n_images=10_000,
    #     save_raw_image=True,
    #     use_generator=0
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_5_GEN_DataCreation_SPEC_GEN_1",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__5_n_gen_5",
    #     latent_point_generator=generate_latent_points,
    #     n_images=10_000,
    #     save_raw_image=True,
    #     use_generator=1
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_5_GEN_DataCreation_SPEC_GEN_2",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__5_n_gen_5",
    #     latent_point_generator=generate_latent_points,
    #     n_images=10_000,
    #     save_raw_image=True,
    #     use_generator=2
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_5_GEN_DataCreation_SPEC_GEN_3",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__5_n_gen_5",
    #     latent_point_generator=generate_latent_points,
    #     n_images=10_000,
    #     save_raw_image=True,
    #     use_generator=3
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_5_GEN_DataCreation_SPEC_GEN_4",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-30_MNIST_MADGAN_Experiment__5_n_gen_5",
    #     latent_point_generator=generate_latent_points,
    #     n_images=10_0,
    #     save_raw_image=True,
    #     use_generator=4
    # ),

    # MNIST GENERATOR 7
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_7_GEN_DataCreation_SPEC_GEN_0",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__7_n_gen_7",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=0
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_7_GEN_DataCreation_SPEC_GEN_1",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__7_n_gen_7",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=1
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_7_GEN_DataCreation_SPEC_GEN_2",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__7_n_gen_7",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=2
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_7_GEN_DataCreation_SPEC_GEN_3",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__7_n_gen_7",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=3
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_7_GEN_DataCreation_SPEC_GEN_4",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__7_n_gen_7",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=4
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_7_GEN_DataCreation_SPEC_GEN_5",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__7_n_gen_7",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=5
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_7_GEN_DataCreation_SPEC_GEN_6",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__7_n_gen_7",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=6
    # ),

    
    # MNIST GENERATOR 10
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_0",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=0
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_1",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=1
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_2",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=2
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_3",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=3
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_4",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=4
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_5",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=5
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_6",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=6
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_7",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=7
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_8",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=8
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_9",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=9
    # ),

    #
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_FASHION_MNIST_5_GEN_DataCreation_SPEC_GEN_4",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2025-01-04_FASHION_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=1_0,
    #     save_raw_image=True,
    #     use_generator=4
    # ),


    ######################################################################################################################
    ################################################ FASHOIN MNIST #######################################################
    ######################################################################################################################
    ## GENERATE IMAGES FOR THE MNIST DATASET USING THE MADGAN WITH 3 GENERATORS, USING A SPECIFIC ONE FOR EACH CREATION
    
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_FASHIONMNIST_DataCreation_SPEC_GEN_0",
    #     experiment_class=FASHION_MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2025-01-01_FASHION_MNIST_MADGAN_Experiment__3_n_gen_3",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=0
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_FASHIONMNIST_DataCreation_SPEC_GEN_1",
    #     experiment_class=FASHION_MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2025-01-01_FASHION_MNIST_MADGAN_Experiment__3_n_gen_3",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=1
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_FASHIONMNIST_DataCreation_SPEC_GEN_2",
    #     experiment_class=FASHION_MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2025-01-01_FASHION_MNIST_MADGAN_Experiment__3_n_gen_3",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=2
    # ),

    ## GENERATE IMAGES FOR THE MNIST DATASET USING THE MADGAN WITH 5 GENERATORS, USING A SPECIFIC ONE FOR EACH CREATION
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_FASHIONMNIST_5_GEN_DataCreation_SPEC_GEN_0",
    #     experiment_class=FASHION_MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__5_n_gen_5",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=0
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_FASHIONMNIST_5_GEN_DataCreation_SPEC_GEN_1",
    #     experiment_class=FASHION_MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__5_n_gen_5",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=1
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_FASHIONMNIST_5_GEN_DataCreation_SPEC_GEN_2",
    #     experiment_class=FASHION_MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__5_n_gen_5",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=2
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_FASHIONMNIST_5_GEN_DataCreation_SPEC_GEN_3",
    #     experiment_class=FASHION_MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__5_n_gen_5",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=3
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_FASHIONMNIST_5_GEN_DataCreation_SPEC_GEN_4",
    #     experiment_class=FASHION_MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__5_n_gen_5",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=4
    # ),

    # FASHION MNIST GENERATOR 7
    MADGAN_GenerativeCreationExperiment(
        name="MADGAN_FASHIONMNIST_7_GEN_DataCreation_SPEC_GEN_0",
        experiment_class=FASHION_MNIST_MADGAN_Experiment,
        experiment_path="experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__7_n_gen_7",
        latent_point_generator=generate_latent_points,
        n_images=90_000,
        save_raw_image=True,
        use_generator=0
    ),
    MADGAN_GenerativeCreationExperiment(
        name="MADGAN_FASHIONMNIST_7_GEN_DataCreation_SPEC_GEN_1",
        experiment_class=FASHION_MNIST_MADGAN_Experiment,
        experiment_path="experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__7_n_gen_7",
        latent_point_generator=generate_latent_points,
        n_images=90_000,
        save_raw_image=True,
        use_generator=1
    ),
    MADGAN_GenerativeCreationExperiment(
        name="MADGAN_FASHIONMNIST_7_GEN_DataCreation_SPEC_GEN_2",
        experiment_class=FASHION_MNIST_MADGAN_Experiment,
        experiment_path="experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__7_n_gen_7",
        latent_point_generator=generate_latent_points,
        n_images=90_000,
        save_raw_image=True,
        use_generator=2
    ),
    MADGAN_GenerativeCreationExperiment(
        name="MADGAN_FASHIONMNIST_7_GEN_DataCreation_SPEC_GEN_3",
        experiment_class=FASHION_MNIST_MADGAN_Experiment,
        experiment_path="experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__7_n_gen_7",
        latent_point_generator=generate_latent_points,
        n_images=90_000,
        save_raw_image=True,
        use_generator=3
    ),
    MADGAN_GenerativeCreationExperiment(
        name="MADGAN_FASHIONMNIST_7_GEN_DataCreation_SPEC_GEN_4",
        experiment_class=FASHION_MNIST_MADGAN_Experiment,
        experiment_path="experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__7_n_gen_7",
        latent_point_generator=generate_latent_points,
        n_images=90_000,
        save_raw_image=True,
        use_generator=4
    ),
    MADGAN_GenerativeCreationExperiment(
        name="MADGAN_FASHIONMNIST_7_GEN_DataCreation_SPEC_GEN_5",
        experiment_class=FASHION_MNIST_MADGAN_Experiment,
        experiment_path="experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__7_n_gen_7",
        latent_point_generator=generate_latent_points,
        n_images=90_000,
        save_raw_image=True,
        use_generator=5
    ),
    MADGAN_GenerativeCreationExperiment(
        name="MADGAN_FASHIONMNIST_7_GEN_DataCreation_SPEC_GEN_6",
        experiment_class=FASHION_MNIST_MADGAN_Experiment,
        experiment_path="experiments/2025-01-02_FASHION_MNIST_MADGAN_Experiment__7_n_gen_7",
        latent_point_generator=generate_latent_points,
        n_images=90_000,
        save_raw_image=True,
        use_generator=6
    ),

    
    # MNIST GENERATOR 10
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_0",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=0
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_1",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=1
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_2",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=2
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_3",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=3
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_4",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=4
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_5",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=5
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_6",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=6
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_7",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=7
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_8",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=8
    # ),
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_MNIST_10_GEN_DataCreation_SPEC_GEN_9",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2024-12-31_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=90_000,
    #     save_raw_image=True,
    #     use_generator=9
    # ),

    #
    # MADGAN_GenerativeCreationExperiment(
    #     name="MADGAN_FASHION_MNIST_5_GEN_DataCreation_SPEC_GEN_4",
    #     experiment_class=MNIST_MADGAN_Experiment,
    #     experiment_path="experiments/2025-01-04_FASHION_MNIST_MADGAN_Experiment__10_n_gen_10",
    #     latent_point_generator=generate_latent_points,
    #     n_images=1_0,
    #     save_raw_image=True,
    #     use_generator=4
    # ),

    
    
    
    
    
    
    
    
]

queue = ExperimentQueue()
for exp in experiments:
    queue.add_experiment(exp)
queue.run_all()

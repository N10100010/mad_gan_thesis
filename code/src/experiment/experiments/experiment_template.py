from experiment import BaseExperiment
from src.experiment.base_experiments.base_experiment import call_super


class TemplateExperiment(BaseExperiment):
    """Test implementation of the BaseExperiments class

    Args:
        BaseExperiment (_type_): _description_
    """

    @call_super
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @call_super
    def _run(self):
        pass

    @call_super
    def _setup(self):
        pass

    @call_super
    def _load_data(self):
        pass

    @call_super
    def _initialize_models(self):
        pass

    @call_super
    def _save_results(self):
        pass

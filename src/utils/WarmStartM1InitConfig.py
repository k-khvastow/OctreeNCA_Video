from src.utils.ExperimentWrapper import ExperimentWrapper
from src.losses.WeightedLosses import WeightedLosses
from src.models.Model_OctreeNCA_WarmStart_M1Init import OctreeNCA2DWarmStartM1Init
from src.agents.Agent_OctreeNCA_WarmStart_M1Init import OctreeNCAWarmStartM1InitAgent


class EXP_OctreeNCA_WarmStart_M1Init(ExperimentWrapper):
    def createExperiment(self, study_config: dict, detail_config: dict = {}, dataset_class=None, dataset_args=None):
        if dataset_args is None:
            dataset_args = {}
        if dataset_class is None:
            assert False, "Dataset is None"

        model = OctreeNCA2DWarmStartM1Init(study_config)
        agent = OctreeNCAWarmStartM1InitAgent(model)
        loss_function = WeightedLosses(study_config)

        return super().createExperiment(study_config, model, agent, dataset_class, dataset_args, loss_function)

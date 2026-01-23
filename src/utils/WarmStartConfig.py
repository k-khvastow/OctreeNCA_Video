from src.utils.ExperimentWrapper import ExperimentWrapper
from src.losses.WeightedLosses import WeightedLosses
from src.models.Model_OctreeNCA_WarmStart import OctreeNCA2DWarmStart
from src.agents.Agent_OctreeNCA_WarmStart import OctreeNCAWarmStartAgent

class EXP_OctreeNCA_WarmStart(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = study_config
        if dataset_class is None:
            assert False, "Dataset is None"
            
        # Initialize Custom Warm Start Model
        model = OctreeNCA2DWarmStart(config)
        
        # Initialize Custom Sequential Agent
        agent = OctreeNCAWarmStartAgent(model)
        
        loss_function = WeightedLosses(config) 

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)
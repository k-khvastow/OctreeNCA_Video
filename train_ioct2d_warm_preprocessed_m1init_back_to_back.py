from src.utils.WarmStartM1InitConfig import EXP_OctreeNCA_WarmStart_M1Init
from src.utils.Study import Study
import wonderwords

from train_ioct2d_warm_preprocessed_m1init import (
    iOCTSequentialDatasetForExperiment,
    get_dataset_args as get_base_dataset_args,
    get_study_config as get_base_study_config,
)


r = wonderwords.RandomWord()
random_word = r.word(include_parts_of_speech=["nouns"])

# When training from scratch, this gives direct supervision starting at t=0.
USE_T0_FOR_LOSS = True
M1_CHECKPOINT_PATH = "/vol/data/OctreeNCA_Video/<path>/<path>/octree_study_new/Experiments/iOCT2D_hospital_24_Training OctreeNCA on iOCT 2D frames./models/epoch_99/model.pth"


def get_study_config():
    study_config = get_base_study_config()

    # Back-to-back training: do not load pretrained M1, and train M1 + M2 jointly.
    study_config["model.m1.pretrained_path"] = M1_CHECKPOINT_PATH
    study_config["model.m1.freeze"] = False
    study_config["model.m1.eval_mode"] = False
    study_config["model.m1.load_strict"] = False
    study_config["model.m1.use_probs"] = True
    study_config["model.m1.use_t0_for_loss"] = USE_T0_FOR_LOSS

    study_config["experiment.description"] = (
        "Back-to-back training of M1 and M2 on sequential iOCT frames (no pretrained M1)."
    )
    study_config["experiment.name"] = (
        f"WarmStart_M1Init_B2B_iOCT2D_{random_word}_{study_config['model.channel_n']}"
    )
    return study_config


def get_dataset_args(study_config):
    return get_base_dataset_args(study_config)


if __name__ == "__main__":
    study_config = get_study_config()
    dataset_args = get_dataset_args(study_config)

    study = Study(study_config)
    exp = EXP_OctreeNCA_WarmStart_M1Init().createExperiment(
        study_config,
        detail_config={},
        dataset_class=iOCTSequentialDatasetForExperiment,
        dataset_args=dataset_args,
    )
    study.add_experiment(exp)

    print(f"Starting experiment: {study_config['experiment.name']}")
    study.run_experiments()

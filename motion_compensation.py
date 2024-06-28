import sys
import json
import torch
import hydra
import torch
import random
import numpy as np
from pathlib import Path
from optimizer import Optimizer
from reconstruction_settings import *
from target_function import TargetFunction
from omegaconf import DictConfig, OmegaConf
from motion_compensation_data_loader import Data
sys.path.append('motion_models')
from motion_models.motion_models_2d_torch import MotionModel2DTorch


device = torch.device('cuda')


@hydra.main(version_base=None, config_path='hydra_configs', config_name='config_ncsnpp')
def main(config: DictConfig):
    # print config
    print(OmegaConf.to_yaml(config))

    # set random seed for comparable results across multiple runs
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    motion_model_perturb = MotionModel2DTorch(config.motion_model_perturb.choice, num_projections=num_projections,
                                              num_nodes=config.motion_model_perturb.num_nodes)
    motion_model_recover = MotionModel2DTorch(config.motion_model_recover.choice, num_projections=num_projections,
                                              num_nodes=config.motion_model_recover.num_nodes)

    test_ids = ('368', '369', '370', '372', '373', '378', '380', '383', '384', '386', '388', '389', '390', '392',
                '393', '394', '395', '396', '397', '401', '402', '403', '404', '406', '407', '410', '411', '412',
                '414', '416', '417', '418', '420', '421', '422', '423', '425', '428', '429', '430', '434', '435',
                '436', '439', '440', '441', '442', '443', '444', '446', '449', '450', '451', '452', '454', '456',
                '458', '459', '460', '461', '462', '463', '465', '466', '467', '469', '470', '471', '472', '475',
                '477', '478', '479', '480', '482', '483', '485', '486', '488', '489')

    sample_names = config.data.samples

    image_lookup = json.load(open('image_lookup.json'))

    for sample_name in sample_names:
        assert str(sample_name) in test_ids, 'This is not a sample from the test set.'
        sample_name = 'CQ500CT' + str(sample_name)
        sample_path = Path(config.data_dir) / sample_name
        # select n random slices from that scan
        for image_name in image_lookup[sample_name]:
            image_path = sample_path / image_name
            print(f'{sample_name}: {image_name}')
            data = Data(image_path, motion_model_perturb,
                        amplitude_rotation=config.data.amplitude_rotation,
                        amplitude_translation=config.data.amplitude_translation)
            sample = data.get_sample()
            target_function = TargetFunction(choice=config.target_function.choice, config=config,
                                             projections=sample[0], motion_model=motion_model_recover,
                                             projection_matrices_circular=sample[2],
                                             ground_truth_reconstruction=sample[3])
            optimizer = Optimizer(sample, config.optimizer.choice, target_function, config.optimizer.options,
                                  experiment_name=config.experiment_name, sample_name=sample_name + '_' + image_name)

            # run the optimization
            result = optimizer.optimize()

            # evaluate and store the results (hydra config will be stored automatically in the same folder)
            optimizer.evaluate()


if __name__ == '__main__':
    main()

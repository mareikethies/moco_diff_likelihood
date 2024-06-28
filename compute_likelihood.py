import numpy as np
import pandas as pd
from tqdm import tqdm
from absl import flags, app
import matplotlib.pyplot as plt
import differentiable_likelihood_function as l_diff
from ml_collections.config_flags import config_flags
import sys
sys.path.append('score_sde_pytorch')
import losses
import sde_lib
import likelihood as l
from pathlib import Path
import models.utils as mutils
from utils import restore_checkpoint
from models.ema import ExponentialMovingAverage
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
from custom_datasets import get_paired_motion_datasets


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("datadir", None, "Data directory.")


def compute_likelihood_full_dataset(config, dataset, bpd_num_repeats=1, implementation='differentiable'):
    score_model = initialize_score_model(config)

    sde = initialize_sde(config)

    likelihood_fn = None
    if implementation == 'differentiable':
        likelihood_fn = l_diff.get_likelihood_fn(sde, method='rk4', stepsize=0.1)
    elif implementation == 'original':
        likelihood_fn = l.get_likelihood_fn(sde, method='RK45', atol=1e-2)

    bpds = []
    images = []
    image_gradients = []
    for repeat in range(bpd_num_repeats):
        bpd_iter = iter(dataset)  # pytype: disable=wrong-arg-types
        for batch_id in tqdm(range(len(dataset))):
            batch = next(bpd_iter)['image'].to(config.device).float()
            # batch.requires_grad_(True)
            bpd = likelihood_fn(score_model, batch)[0]
            # bpd.backward()
            bpd = bpd.detach().cpu().numpy().reshape(-1)
            bpds.extend(bpd)
            # images.extend(batch.detach().cpu().numpy())
            # image_gradients.extend(batch.grad.cpu().numpy())

    return bpds, images, image_gradients


def initialize_sde(config):
    # initialize sde
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=config.model.num_scales)
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    return sde


def initialize_score_model(config, checkpoint='checkpoint_ncsnpp_small.pth'):
    # initialize model
    score_model = mutils.create_model(config)
    # num_params = sum(p.numel() for p in score_model.parameters() if p.requires_grad)
    # print(f'Number of parameters of the score model: {num_params}')
    # fill in the trained weights from a suitable checkpoint
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    checkpoint_dir = Path('./checkpoints')
    ckpt_path = checkpoint_dir / checkpoint
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema = state['ema']
    ema.copy_to(score_model.parameters())
    return score_model


def main(argv):
    df = pd.DataFrame(columns=['motion_amplitude', 'bpd', 'implementation'])

    motion_amplitudes = [0, 1, 2, 4, 8]
    implementations = ['differentiable', 'original']
    for implementation in implementations:
        for motion_amplitude in motion_amplitudes:
            _, motion_affected_dataset = get_paired_motion_datasets(FLAGS.datadir, motion_amplitude, motion_amplitude)

            print(f'Computing bpd for motion-affected data, amplitude {motion_amplitude}, {implementation} implementation.')
            motion_affected_likelihoods, _, _ = compute_likelihood_full_dataset(FLAGS.config, motion_affected_dataset,
                                                                                implementation=implementation)

            data_to_append = {'motion_amplitude': [motion_amplitude for i in range(len(motion_affected_likelihoods))],
                              'bpd': motion_affected_likelihoods,
                              'implementation': [implementation for i in range(len(motion_affected_likelihoods))]}
            new_rows = pd.DataFrame(data_to_append)
            df = pd.concat([df, new_rows], ignore_index=True)

        df.to_csv(f'out/bpd_distributions_{implementation}.csv', index=False)


if __name__ == '__main__':
    app.run(main)
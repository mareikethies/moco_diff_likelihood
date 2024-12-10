.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0
.. image:: https://img.shields.io/badge/arXiv-2212.02177-b31b1b.svg
    :target: https://arxiv.org/abs/2404.14747

Differentiable Score-Based Likelihoods: Learning CT Motion Compensation From Clean Images
=========================================================================================

This repository contains the source code for our paper **Differentiable Score-Based Likelihoods: Learning CT Motion
Compensation From Clean Images** which has been presented at `MICCAI 2024 
<https://link.springer.com/chapter/10.1007/978-3-031-72104-5_25>`_. Trained weights and example data are provided
to run the proposed CT motion algorithm using differentiable likelihood computation of diffusion models, as well as code
and weights for the reference methods. This repository also contains the code to create the full data set based on the
publicly available CQ500 data set which can be used to train the models.

Usage with pretrained weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This section describes how to run the motion compensation on an example head CT slice with pretrained weights.

#.  Clone this repository including submodules

    .. code-block::

        git clone https://github.com/mareikethies/moco_diff_likelihood.git --recurse_submodules

#.  Download network weights and example data from figshare using the following links:
    `Example data <http://doi.org/10.6084/m9.figshare.26117416.v1>`_,
    `diffusion model weights <http://doi.org/10.6084/m9.figshare.26117347.v1>`_,
    `autofocus model weights <http://doi.org/10.6084/m9.figshare.26117284.v1>`_. Place both model weights unchanged in the
    ``checkpoints`` subfolder. Extract the example data and place it in the ``data`` subfolder. The folder structure should
    look like

    .. code-block::

        ./data/CQ500CT268/image_60/

#.  To apply a random motion pattern to the example data and compensate it...

    ...using the proposed diffusion model likelihood target function (requires ~ 10 GB GPU memory), run

    .. code-block::

        python motion_compensation.py

    ...using the autofocus target function, run

    .. code-block::

        python motion_compensation.py target_function.choice='autofocus' optimizer.options.stepsize_translation=50 optimizer.options.stepsize_rotation=50

    ...using the supervised MSE optimization with ground truth, run

    .. code-block::

        python motion_compensation.py target_function.choice='mse_supervised' optimizer.options.stepsize_translation=5000 optimizer.options.stepsize_rotation=5000

    Results will be stored in the ``out`` subfolder.

Recreate full data set and train models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This section describes how to process the full CQ500 data set to reproduce the results from the paper and how to use it
for training the networks.

#.  Check the license and download the CQ500 head CT data set from `here <http://headctstudy.qure.ai/dataset>`_.

#.  Filter the data set for reconstructions with thin slice thickness using ``clean_cq500.py``.

#.  Create a fan-beam data set using ``generate_data.py``.

#.  To train the score-based diffusion model, run

    .. code-block::

        python score_sde_pytorch/main.py --config motion_ncsnp_continuous_small.py --mode train --workdir your_output_path --datadir your_path/HeadSimulatedProjectionDataCQ500FanBeam/motion_free/

#.  To train the autofocus model, run

    .. code-block::

        python autofocus_train.py --data_dir your_path/HeadSimulatedProjectionDataCQ500FanBeam/motion_free --run_name your_name --accelerator cuda --devices 1 --max_epochs 2000 --batch_size 32 --learning_rate 1e-3 --num_workers 8

The full data set created in steps 1 - 3 instead of the example data and the ``image_lookup_paper_results.json``
instead of ``image_lookup.json``, will recreate the results in the published paper.

Dependencies and licenses
~~~~~~~~~~~~~~~~~~~~~~~~~
This code depends on (1) our previously published work on gradient propagation to CT geometry parameters
(`see paper <http://doi.org/10.1088/1361-6560/acf90e>`_,
`see github <https://github.com/mareikethies/geometry_gradients_CT>`_) and (2) the original work on score-based
diffusion models by Song et al. (`see paper <https://openreview.net/forum?id=PxTIG12RRHS>`_ in particular section D.2
of the appendix, `see github <https://github.com/yang-song/score_sde_pytorch>`_). The source code for both works is
published under Apache 2.0 license.

We further provide sources for the PyroNN package as used for development of this project. The original, but outdated
repository is `here <https://github.com/csyben/PYRO-NN>`_. For this project, please install pyronn from the provided
sources in the ``pyronn`` folder by running (from within that folder)

    .. code-block::

        python -m pip install -e pyronn-layers-torch
        python -m pip install -e pyronn

This requires a version of pytorch < 2.0.

Citation
~~~~~~~~
If you use this code for your research, please cite our paper:

.. code-block::

    @InProceedings{10.1007/978-3-031-72104-5_25,
        author="Thies, Mareike and Maul, Noah and Mei, Siyuan and Pfaff, Laura and Vysotskaya, Nastassia and Gu, Mingxuan and Utz, Jonas and Possart, Dennis and Folle, Lukas and Wagner, Fabian and Maier, Andreas",
        editor="Linguraru, Marius George and Dou, Qi and Feragen, Aasa and Giannarou, Stamatia and Glocker, Ben and Lekadir, Karim and Schnabel, Julia A.",
        title="Differentiable Score-Based Likelihoods: Learning CT Motion Compensation from Clean Images",
        booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024",
        year="2024",
        publisher="Springer Nature Switzerland",
        address="Cham",
        pages="253--263",
    }

If you have any questions about this repository or the paper, feel free to reach out
(`mareike.thies@fau.de <mareike.thies@fau.de>`_).

import json
import math
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.patches as mpatches
from skimage.metrics import structural_similarity as ssim


# set default seaborn theme
sns.set_theme()
sns.set(font_scale=1.2)
sns.set_style('whitegrid')
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def plot_bpd_distribution():
    df = pd.read_csv('out/bpd_distributions.csv')

    df_original = df[df['implementation'] == 'original']
    df_differentiable = df[df['implementation'] == 'differentiable']

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    sns.kdeplot(df_original, x='bpd', hue='motion_amplitude')
    plt.title('Original implementation')
    plt.ylim([0, 0.27])
    plt.tight_layout()

    plt.subplot(122)
    sns.kdeplot(df_differentiable, x='bpd', hue='motion_amplitude')
    plt.title('Differentiable implementation')
    plt.ylim([0, 0.27])
    plt.tight_layout()

    plt.show()


def compute_rpe(ground_truth_matrices, corrupted_matrices, points, detector_spacing):
    # project all points into all views for both sets of matrices
    ground_truth_points = np.einsum('nij,mj->nmi', ground_truth_matrices, points)
    corrupted_points = np.einsum('nij,mj->nmi', corrupted_matrices, points)

    # dehomogenize projected points
    ground_truth_points = ground_truth_points[:, :, 0] / ground_truth_points[:, :, 1]
    corrupted_points = corrupted_points[:, :, 0] / corrupted_points[:, :, 1]

    # subtract one from the other
    diff = ground_truth_points - corrupted_points

    # compute mean over all views and over all points
    diff = np.mean(np.abs(diff))

    return diff * detector_spacing


def compute_image_based_metrics(df):
    ssim_corrupted = []
    ssim_recovered = []
    rmse_corrupted = []
    rmse_recovered = []
    vif_corrupted = []
    vif_recovered = []

    for i, item in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            folder, run_name = get_subfolder(item)
            ground_truth = imread(str(folder / f'{run_name}_reco_target.tif'))
            corrupted = imread(str(folder / f'{run_name}_reco_init.tif'))
            recovered = imread(str(folder / f'{run_name}_reco_out.tif'))

            ssim_corrupted.append(np.mean(ssim(corrupted, ground_truth, win_size=5, data_range=1.)))
            ssim_recovered.append(np.mean(ssim(recovered, ground_truth, win_size=5, data_range=1.)))
            rmse_corrupted.append(np.sqrt(np.mean(np.square(ground_truth - corrupted))))
            rmse_recovered.append(np.sqrt(np.mean(np.square(ground_truth - recovered))))
            vif_recovered.append(0)
            vif_corrupted.append(0)
        except:
            ssim_corrupted.append(np.nan)
            ssim_recovered.append(np.nan)
            rmse_corrupted.append(np.nan)
            rmse_recovered.append(np.nan)
            vif_corrupted.append(np.nan)
            vif_recovered.append(np.nan)
    df['rmse corrupted'] = rmse_corrupted
    df['rmse recovered'] = rmse_recovered
    df['ssim corrupted'] = ssim_corrupted
    df['ssim recovered'] = ssim_recovered
    df['vif corrupted'] = vif_corrupted
    df['vif recovered'] = vif_recovered

    return df


def get_subfolder(item):
    base_folder = Path('out/gradient_descent')
    experiment_name = 'moco_results_test_set'
    folder = base_folder / item['target function'] / experiment_name / item['sample']
    return folder, experiment_name


def compute_motion_param_metrics(df):
    error_r_corrupted = []
    error_r_recovered = []
    error_tx_corrupted = []
    error_tx_recovered = []
    error_ty_corrupted = []
    error_ty_recovered = []
    for i, item in df.iterrows():
        try:
            folder, run_name = get_subfolder(item)

            with open(folder / f'{run_name}_result.json', 'r') as f:
                result = json.load(f)

            r_init = np.array(result['motion_r_init'])
            tx_init = np.array(result['motion_tx_init'])
            ty_init = np.array(result['motion_ty_init'])

            r_recovered = np.array(result['motion_r_recovered'])
            tx_recovered = np.array(result['motion_tx_recovered'])
            ty_recovered = np.array(result['motion_ty_recovered'])

            error_r_corrupted.append(np.mean(np.abs(r_init)))
            error_r_recovered.append(np.mean(np.abs(r_init + r_recovered)))
            error_tx_corrupted.append(np.mean(np.abs(tx_init)))
            error_tx_recovered.append(np.mean(np.abs(tx_init + tx_recovered)))
            error_ty_corrupted.append(np.mean(np.abs(ty_init)))
            error_ty_recovered.append(np.mean(np.abs(ty_init + ty_recovered)))

        except:
            error_r_corrupted.append(np.nan)
            error_r_recovered.append(np.nan)
            error_tx_corrupted.append(np.nan)
            error_tx_recovered.append(np.nan)
            error_ty_corrupted.append(np.nan)
            error_ty_recovered.append(np.nan)

    df['error r corrupted'] = error_r_corrupted
    df['error r recovered'] = error_r_recovered
    df['error tx corrupted'] = error_tx_corrupted
    df['error tx recovered'] = error_tx_recovered
    df['error ty corrupted'] = error_ty_corrupted
    df['error ty recovered'] = error_ty_recovered

    return df


def compute_proj_matrix_metrics(df):
    detector_spacing = 0.64
    points1 = get_points(100, 50)
    points2 = get_points(100, 100)
    points3 = get_points(100, 25)
    points = np.concatenate((points1, points2, points3), axis=0)

    rpe_corrupted = []
    rpe_recovered = []
    for i, item in df.iterrows():
        try:
            folder, run_name = get_subfolder(item)
            projection_matrices_target = np.load(str(folder / f'{run_name}_projection_matrices_target.npy'))
            projection_matrices_corrupted = np.load(str(folder / f'{run_name}_projection_matrices_init.npy'))
            projection_matrices_recovered = np.load(str(folder / f'{run_name}_projection_matrices_out.npy'))

            rpe_corrupted.append(compute_rpe(projection_matrices_target, projection_matrices_corrupted, points, detector_spacing))
            rpe_recovered.append(compute_rpe(projection_matrices_target, projection_matrices_recovered, points, detector_spacing))
        except:
            rpe_corrupted.append(np.nan)
            rpe_recovered.append(np.nan)

    df['rpe corrupted'] = rpe_corrupted
    df['rpe recovered'] = rpe_recovered

    return df


def get_points(N, radius):
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.ones((N, 3))
    points[:, 0] = np.sin(angles) * radius
    points[:, 1] = np.cos(angles) * radius

    return points


def construct_initial_df():
    test_ids = ('368', '369', '370', '372', '373', '378', '380', '383', '384', '386', '388', '389', '390', '392',
                '393', '394', '395', '396', '397', '401', '402', '403', '404', '406', '407', '410', '411', '412',
                '414', '416', '417', '418', '420', '421', '422', '423', '425', '428', '429', '430')

    df = pd.DataFrame(columns=['sample', 'target function'])

    sample_names = [f'CQ500CT{id}' for id in test_ids]

    image_lookup = json.load(open('image_lookup.json'))

    target_functions = ['diffusion_likelihood', 'mse_supervised', 'autofocus']
    for target_function in target_functions:
        for sample_name in sample_names:
            for image_name in image_lookup[sample_name]:
                data_to_append = {'sample': [sample_name + '_' + image_name], 'target function': [target_function]}
                new_rows = pd.DataFrame(data_to_append)
                df = pd.concat([df, new_rows], ignore_index=True)

    return df


def plot_metrics(df):
    ### image based metrics ###
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplot(131)
    plot = sns.boxplot(df, x='target function', y='rmse recovered', hue='target function', ax=axes[0],
                       flierprops={"marker": "x"}, order=['mse_supervised', 'autofocus', 'diffusion_likelihood'],
                       hue_order=['mse_supervised', 'autofocus', 'diffusion_likelihood'], legend='full',
                       palette=['#5975a4', '#58a7a4', '#cc8963'])
    add_init(df, 'rmse corrupted', legend=True)
    sns.move_legend(plot, loc='upper left', bbox_to_anchor=(-0.3, 1.2), ncols=4, fancybox=True)
    plot.get_legend().set_title(None)
    plot.get_legend().texts[0].set_text('\\textbf{Ideal (MSE)}\nOptimized with ground truth')
    plot.get_legend().texts[1].set_text('\\textbf{Autofocus}\nTrained on motion-affected images')
    plot.get_legend().texts[2].set_text('\\textbf{Ours}\nTrained on motion-free images')
    plot.get_legend().texts[3].set_text('\\textbf{Init}\nInitial motion-affected state')
    plt.ylim([0, 0.073])
    plt.xlabel('')
    plt.ylabel('RMSE')

    plt.subplot(132)
    sns.boxplot(df, x='target function', y='ssim recovered', hue='target function', ax=axes[1], flierprops={"marker": "x"},
                order=['mse_supervised', 'autofocus', 'diffusion_likelihood'],
                hue_order=['mse_supervised', 'autofocus', 'diffusion_likelihood'],
                palette=['#5975a4', '#58a7a4', '#cc8963'])
    add_init(df, 'ssim corrupted')
    plt.xlabel('')
    plt.ylabel('SSIM')

    plt.subplot(133)
    plot = sns.boxplot(df, x='target function', y='rpe recovered', hue='target function', flierprops={"marker": "x"},
                       order=['mse_supervised', 'autofocus', 'diffusion_likelihood'],
                       hue_order=['mse_supervised', 'autofocus', 'diffusion_likelihood'],
                       palette=['#5975a4', '#58a7a4', '#cc8963'])
    add_init(df, 'rpe corrupted')
    plt.xlabel('')
    plt.ylabel('RPE [mm]')
    plt.ylim([0, 3.1])

    fig.subplots_adjust(top=0.8)
    plt.savefig(f'figures/eval_miccai.pdf')

    #### motion based metrics ###
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplot(131)
    plot = sns.boxplot(df, x='target function', y='error tx recovered', hue='target function', ax=axes[0], flierprops={"marker": "x"},
                       order=['mse_supervised', 'autofocus', 'diffusion_likelihood'],
                       hue_order=['mse_supervised', 'autofocus', 'diffusion_likelihood'], legend='full',
                       palette=['#5975a4', '#58a7a4', '#cc8963'])
    add_init(df, 'error tx corrupted', legend=True)
    plot.get_legend().set_title(None)
    plot.get_legend().texts[0].set_text('\\textbf{Ideal target function (MSE)}\nOptimized with ground truth')
    plot.get_legend().texts[1].set_text('\\textbf{Autofocus}\nTrained on motion-affected images')
    plot.get_legend().texts[2].set_text('\\textbf{Ours}\nTrained on motion-free images')
    plot.get_legend().texts[3].set_text('\\textbf{Init}\nInitial motion-affected state')
    plt.xlabel('')
    plt.ylabel(r'Error $t_x$ [mm]')
    plt.ylim(0, 2.)
    plt.subplot(132)
    sns.boxplot(df, x='target function', y='error ty recovered', hue='target function', ax=axes[1], flierprops={"marker": "x"},
                order=['mse_supervised', 'autofocus', 'diffusion_likelihood'],
                hue_order=['mse_supervised', 'autofocus', 'diffusion_likelihood'],
                palette=['#5975a4', '#58a7a4', '#cc8963'])
    add_init(df, 'error ty corrupted')
    plt.xlabel('')
    plt.ylabel(r'Error $t_y$ [mm]')
    plt.ylim(0, 2.)
    plt.legend([], [], frameon=False)
    plt.subplot(133)
    sns.boxplot(df, x='target function', y='error r recovered', hue='target function', ax=axes[2], flierprops={"marker": "x"},
                order=['mse_supervised', 'autofocus', 'diffusion_likelihood'],
                hue_order=['mse_supervised', 'autofocus', 'diffusion_likelihood'],
                palette=['#5975a4', '#58a7a4', '#cc8963'])
    add_init(df, 'error r corrupted')
    plt.xlabel('')
    plt.ylabel(r'Error $r$ [deg]')
    plt.ylim(0, 2.)
    plt.legend([], [], frameon=False)

    # plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.tight_layout()
    plt.savefig(f'figures/eval_motion_based.pdf')

    plt.show()


def add_init(df, label, legend=False):
    df_corrupted = df[df['target function'] == 'mse_supervised']
    values = df_corrupted[label]
    ax = plt.gca()
    extra_box = ax.boxplot([values], positions=[3], widths=(0.8,), patch_artist=True,
                           flierprops={'marker': 'x', 'markersize': 6, 'markeredgecolor': '#4c4c4c'},
                           boxprops={'facecolor': (.5, .5, .5, 1.), 'color': '#4c4c4c', 'linewidth': 1.},
                           medianprops={'color': '#4c4c4c', 'linewidth': 1.},
                           whiskerprops={'color': '#4c4c4c', 'linewidth': 1.},
                           capprops={'color': '#4c4c4c', 'linewidth': 1.})
    ax.set(xticks=[0, 1, 2, 3], xticklabels=['', '', '', ''])
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        patch = mpatches.Patch(facecolor=(.5, .5, .5, 1.), label='\\textbf{Init}', edgecolor='#4c4c4c')
        handles.append(patch)
        plt.legend(handles=handles)


def compute_means(df):
    print_metric(df, 'rmse')
    print_metric(df, 'ssim')
    print_metric(df, 'rpe')
    print_metric(df, 'error tx')
    print_metric(df, 'error ty')
    print_metric(df, 'error r')


def print_metric(df, metric):
    print(f'### {metric} ###')
    ours_df = df[df['target function'] == 'diffusion_likelihood']
    init_mean = ours_df[f'{metric} corrupted'].mean()
    init_std = ours_df[f'{metric} corrupted'].std()
    ours_mean = ours_df[f'{metric} recovered'].mean()
    ours_std = ours_df[f'{metric} recovered'].std()
    mse_df = df[df['target function'] == 'mse_supervised']
    mse_mean = mse_df[f'{metric} recovered'].mean()
    mse_std = mse_df[f'{metric} recovered'].std()
    autofocus_df = df[df['target function'] == 'autofocus']
    autofocus_mean = autofocus_df[f'{metric} recovered'].mean()
    autofocus_std = autofocus_df[f'{metric} recovered'].std()
    print(f'Init: {init_mean:.3f}  +- {init_std:.3f}')
    print(f'MSE: {mse_mean:.3f}  +- {mse_std:.3f}')
    print(f'Autofocus: {autofocus_mean:.3f}  +- {autofocus_std:.3f}')
    print(f'Ours: {ours_mean:.3f}  +- {ours_std:.3f}')
    print('')


def main():
    # construct an empty data frame that needs to filled
    df = construct_initial_df()
    # add rmse, ssim, and vif to df
    df = compute_image_based_metrics(df)
    # add mae for motion parameters to df
    df = compute_motion_param_metrics(df)
    # add rpe to df
    df = compute_proj_matrix_metrics(df)
    # save data frame
    df.to_csv(f'out/evaluation.csv', index=False)

    # read df from file for plotting
    # df = pd.read_csv('out/evaluation.csv', index_col=False)

    # plot some results
    plot_metrics(df)
    compute_means(df)


if __name__ == '__main__':
    plot_bpd_distribution()
    main()
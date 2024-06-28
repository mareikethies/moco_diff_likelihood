import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from reconstruction_settings import *
from skimage.io import imread, imsave
from helper_diffusion_likelihood import setup_geometry
from pyronn.ct_reconstruction.helpers.filters.filters import ram_lak_2D
from pyronn.ct_reconstruction.layers.projection_2d import FanProjection2D
sys.path.append('geometry_gradients_CT')
from geometry_gradients_CT.backprojector_fan import DifferentiableFanBeamBackprojector

device = torch.device('cuda')


def create_dataset(root_dir, save_dir):
    # read data and select relevant slices that contain anatomy
    scan_folders = [f for f in root_dir.iterdir() if f.is_dir()]
    scan_folders = sorted(scan_folders)

    # setup some reco related stuff
    backward_projector = DifferentiableFanBeamBackprojector.apply
    forward_projector = FanProjection2D()
    _, geometry_differentiable, projection_matrices = setup_geometry('full')
    ramp_filter = torch.from_numpy(ram_lak_2D(detector_shape=(1, detector_shape[0]),
                                              detector_spacing=(1, detector_spacing[0]),
                                              number_of_projections=num_projections)).to(device)

    for folder in tqdm(scan_folders):
        scan_name = folder.name
        volume = imread(folder / f'{scan_name}_ct_thin.tif')
        with open(folder / f'{scan_name}_info.json', 'r') as f:
            info = json.load(f)

        cq500_volume_shape = np.array(volume.shape[1:])
        cq500_volume_spacing = np.array([info['pixel_spacing'][0], info['pixel_spacing'][1]])
        cq500_volume_origin = -0.5 * (cq500_volume_shape - 1) * cq500_volume_spacing
        geometry_pyronn, _, _ = setup_geometry('cq500', cq500_volume_shape, cq500_volume_spacing, cq500_volume_origin)

        for slice in range(volume.shape[0]):
            image = volume[slice, ...]
            avg = np.mean(image)
            if (avg <= 300) and (slice > 100):
                print(f'{scan_name}: Stopping at slice {slice}')
                break
            # use only slices with an average gray value of > 300
            else:
                # forward project into fan-beam geometry and ramp filter
                with torch.no_grad():
                    image = torch.FloatTensor(image).to(device)
                    image = torch.unsqueeze(image, 0)
                    sinogram = forward_projector.forward(image, **geometry_pyronn)
                    filtered_sinogram = torch.fft.fft(sinogram, dim=-1)
                    filtered_sinogram = torch.multiply(filtered_sinogram, ramp_filter)
                    filtered_sinogram = torch.fft.ifft(filtered_sinogram, dim=-1).real

                # backproject
                with torch.no_grad():
                    reconstruction = backward_projector(torch.squeeze(filtered_sinogram), projection_matrices,
                                                        geometry_differentiable)
                    reconstruction = reconstruction / 175000.

                scan_id = scan_name[7:].zfill(3)
                save_to = save_dir / (scan_name[:7] + scan_id) / f'image_{slice}'
                save_to.mkdir(parents=True, exist_ok=True)
                imsave(save_to / 'original_reco.tif', image.cpu().numpy())
                imsave(save_to / 'reco.tif', reconstruction.cpu().numpy())
                imsave(save_to / 'filtered_projections.tif', filtered_sinogram.cpu().numpy())


def check_number_of_samples(root_dir):
    train_ids = ('000', '002', '003', '004', '009', '010', '011', '012', '013', '015', '017', '018', '019',
                 '020', '021', '022', '023', '025', '026', '027', '028', '029', '030', '031', '032', '034',
                 '035', '036', '037', '039', '040', '042', '045', '047', '048', '049', '050', '052', '053',
                 '054', '055', '057', '058', '060', '062', '063', '066', '067', '068', '069', '071', '072',
                 '073', '074', '076', '077', '078', '080', '081', '084', '085', '086', '088', '089', '090',
                 '092', '093', '095', '097', '098', '099', '101', '102', '103', '104', '105', '107', '108',
                 '109', '110', '111', '113', '117', '119', '121', '122', '124', '126', '128', '129', '130',
                 '132', '135', '137', '138', '139', '140', '141', '142', '144', '146', '149', '150', '152',
                 '154', '155', '159', '164', '165', '166', '167', '174', '175', '177', '178', '179', '180',
                 '181', '182', '184', '185', '186', '187', '188', '190', '191', '192', '193', '194', '195',
                 '196', '198', '200', '202', '204', '205', '207', '212', '213', '214', '215', '216', '217',
                 '219', '220', '221', '222', '223', '225', '226', '227', '229', '231', '232', '233', '234',
                 '237', '238', '239', '241', '242', '243', '246', '248', '249', '250', '251', '252', '253',
                 '255', '256', '257', '260', '261', '262', '263', '265', '267', '268', '269', '270', '271',
                 '274', '275', '276', '278', '281', '283', '284', '285', '286', '287', '289', '290', '291',
                 '292', '293', '294', '296', '299')

    subject_ids = ['CQ500CT' + subject_id for subject_id in train_ids]

    total_count = 0
    for subject in subject_ids:
        subfolders_count = sum(1 for item in (root_dir / subject).iterdir() if item.is_dir())
        total_count += subfolders_count

    print(total_count)


if __name__ == '__main__':
    root_dir = Path('your_path/CQ500_head_CT_cleaned_thin')
    out_dir = Path('your_path/HeadSimulatedProjectionDataCQ500FanBeam/motion_free')
    create_dataset(root_dir, out_dir)

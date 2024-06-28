import sys
import torch
import numpy as np
from pathlib import Path
from skimage.io import imread
from torchvision import transforms
from helper_diffusion_likelihood import add_motion
from helper_diffusion_likelihood import setup_geometry
from torch.utils.data import Dataset, DataLoader, ConcatDataset
sys.path.append('geometry_gradients_CT')
from geometry_gradients_CT.backprojector_fan import DifferentiableFanBeamBackprojector


device = torch.device('cuda')

def get_dataset(config, root_dir, evaluation=False):
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
                 '292', '293', '294', '296', '299')  # 200 subjects
    eval_ids = ('300', '301', '302', '303', '308', '309', '310', '311', '312', '313', '314', '316', '317',
                '319', '320', '323', '324', '325', '328', '329', '330', '333', '340', '341', '342', '343',
                '344', '346', '347', '348', '353', '356', '357', '359', '360', '361', '362', '363', '365',
                '367')  # 40 subjects

    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size

    # train data loader
    train_dataset = MotionFreeFanBeamData(root_dir, train_ids)
    num_epochs = int(np.ceil(config.training.n_iters * config.training.batch_size / len(train_dataset)))
    train_datasets = [train_dataset] * num_epochs
    concatenated_train_dataset = ConcatDataset(train_datasets)
    train_dataloader = DataLoader(concatenated_train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    # eval dataloader
    eval_dataset = MotionFreeFanBeamData(root_dir, eval_ids)
    eval_datasets = [eval_dataset] * num_epochs
    concatenated_eval_dataset = ConcatDataset(eval_datasets)
    eval_dataloader = DataLoader(concatenated_eval_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    return train_dataloader, eval_dataloader, None


def get_paired_motion_datasets(root_dir, amplitude_rotation, amplitude_translation):
    eval_ids = ('300', '301', '302', '303', '308', '309', '310', '311', '312', '313')  #, '314', '316', '317',
                # '319', '320', '323', '324', '325', '328', '329', '330', '333', '340', '341', '342', '343',
                # '344', '346', '347', '348', '353', '356', '357', '359', '360', '361', '362', '363', '365',
                # '367')  # 40 subjects

    # motion-free data loader
    motion_free_dataset = MotionFreeFanBeamData(root_dir, eval_ids)
    motion_free_dataloader = DataLoader(motion_free_dataset, batch_size=2, shuffle=False, num_workers=0)

    # motion-affected data loader
    motion_affected_dataset = MotionAffectedFanBeamData(root_dir, eval_ids, amplitude_rotation, amplitude_translation)
    motion_affected_dataloader = DataLoader(motion_affected_dataset, batch_size=2, shuffle=False, num_workers=0)

    return motion_free_dataloader, motion_affected_dataloader


def get_data_scaler(config):
    return lambda x: x


def get_data_inverse_scaler(config):
    return lambda x: x


class MotionFreeFanBeamData(Dataset):
    def __init__(self, root_dir, subject_ids):
        self.root_dir = Path(root_dir)
        self.subject_ids = ['CQ500CT' + subject_id for subject_id in subject_ids]

        self.images = []
        for subject in self.subject_ids:
            # load all images belonging to that subject
            for image in (self.root_dir / subject).iterdir():
                self.images.append(imread(image / 'reco.tif')[np.newaxis, :, :])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # transform = transforms.RandomHorizontalFlip(p=0.5)
        return {'image': self.images[item]}


class MotionAffectedFanBeamData(Dataset):
    def __init__(self, root_dir, subject_ids, amplitude_rotation, amplitude_translation):
        self.root_dir = Path(root_dir)
        self.subject_ids = ['CQ500CT' + subject_id for subject_id in subject_ids]

        backward_projector = DifferentiableFanBeamBackprojector.apply
        _, geometry_differentiable, proj_mat_ground_truth = setup_geometry('full')

        self.images = []
        for subject in self.subject_ids:
            # for all slices belonging to that subject, load the sinogram and add motion according to motion_strength
            for slice in (self.root_dir / subject).iterdir():
                filtered_sinogram = torch.from_numpy(imread(slice / 'filtered_projections.tif')).to(device)
                # backproject with perturbed matrices
                with torch.no_grad():
                    proj_mat_perturbed = add_motion(proj_mat_ground_truth, amplitude_rotation, amplitude_translation)
                    perturbed_reco = backward_projector(torch.squeeze(filtered_sinogram), proj_mat_perturbed,
                                                        geometry_differentiable)
                    perturbed_reco = perturbed_reco / 175000
                self.images.append(perturbed_reco.cpu().numpy()[np.newaxis, :, :])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # transform = transforms.RandomHorizontalFlip(p=0.5)
        return {'image': self.images[item]}



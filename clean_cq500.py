import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pydicom import dcmread
from skimage.io import imsave, imread


def read_head_scan(folder):
    image_list = []
    instance_list = []
    file_list = folder.glob('**/*.dcm')
    file_list = sorted(file_list)
    info = {}
    for i, item in enumerate(file_list):
        dicom_object = dcmread(item)
        if i == 0:
            info['pixel_spacing'] = [float(dicom_object.PixelSpacing[0]), float(dicom_object.PixelSpacing[1])]
            info['slice_thickness'] = float(dicom_object.SliceThickness)
            info['manufacturer'] = dicom_object.Manufacturer
            info['rescale_slope'] = dicom_object.RescaleSlope
            info['rescale_intercept'] = dicom_object.RescaleIntercept
        img = dicom_object.pixel_array
        # do not yet convert pixel values to hounsfield units, but store slope and intercept to do it later
        # img_hu = apply_modality_lut(img, dicom_object)
        image_list.append(img)
        # make sure to load slices in the correct order (sometimes file name order is wrong)
        instance_number = int(dicom_object.get(0x00200013).value)
        instance_list.append(instance_number)

    # convert to volume ensuring correct slice order
    num_slices = len(image_list)
    volume = np.zeros((num_slices, *image_list[0].shape))
    for img, ind in zip(image_list, instance_list):
        volume[ind - 1, :, :] = img

    assert not np.all(volume == 0), f'The scan from folder {folder} seems to be all black.'

    return volume, info


def main():
    root_folder = Path('your_path/CQ500_head_CT')
    out_folder = Path('your_path/CQ500_head_CT_cleaned_thin')

    possible_folder_names = ['CT PLAIN THIN',
                             'CT 0.625mm',
                             'CT Thin Plain',
                             'CT PRE CONTRAST THIN',
                             'CT Plain THIN',
                             'CT Thin Stnd']

    for sub_dir in root_folder.iterdir():
        if sub_dir.is_dir() and (sub_dir / 'Unknown Study').is_dir():
            sample_name = sub_dir.name.split(' ')[0]
            subsub_dirs = list((sub_dir / 'Unknown Study').iterdir())

            match = list(set([i.name for i in subsub_dirs]) & set(possible_folder_names))
            if not len(match) == 1:
                print(f'{sample_name}: X')
            else:
                subsub_dir = sub_dir / 'Unknown Study' / match[0]
                try:
                    volume, info = read_head_scan(subsub_dir)

                    # remove circular mask with value -2000 which will have effects during reconstruction
                    volume[volume == -2000] = 0

                    # sort out volumes with too few or too many slices (these are often blurred)
                    if 200 <= volume.shape[0] <= 300:
                        save_path = out_folder / sample_name
                        save_path.mkdir(parents=True, exist_ok=True)

                        imsave(save_path / f'{sample_name}_ct_thin.tif', volume)
                        with open(save_path / f'{sample_name}_info.json', 'w') as f:
                            json.dump(info, f)
                        print(f'{sample_name}')
                    else:
                        print(f'{sample_name}: X')
                except:
                    print(f'{sample_name}: X')


if __name__ == '__main__':
    main()
from torch.utils.data import DataLoader
from autofocus_data_set import MotionCTDataset
from pytorch_lightning import LightningDataModule


class MotionCTDataLoader(LightningDataModule):

    def __init__(self, data_dir, batch_size, num_workers, **kwargs):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_ids = ('000', '002', '003', '004', '009', '010', '011', '012', '013', '015', '017', '018', '019',
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
        self.val_ids = ('300', '301', '302', '303', '308', '309', '310', '311', '312', '313', '314', '316', '317',
                        '319', '320', '323', '324', '325', '328', '329', '330', '333', '340', '341', '342', '343',
                        '344', '346', '347', '348', '353', '356', '357', '359', '360', '361', '362', '363', '365',
                        '367')
        self.test_ids = ('368', '369', '370', '372', '373', '378', '380', '383', '384', '386', '388', '389', '390',
                         '392', '393', '394', '395', '396', '397', '401', '402', '403', '404', '406', '407', '410',
                         '411', '412', '414', '416', '417', '418', '420', '421', '422', '423', '425', '428', '429',
                         '430', '434', '435', '436', '439', '440', '441', '442', '443', '444', '446', '449', '450',
                         '451', '452', '454', '456', '458', '459', '460', '461', '462', '463', '465', '466', '467',
                         '469', '470', '471', '472', '475', '477', '478', '479', '480', '482', '483', '485', '486',
                         '488', '489')

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Data')
        parser.add_argument('--data_dir', type=str)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=0)
        return parent_parser

    def train_dataloader(self):
        print('\nCreating training data set.')
        train_dataset = MotionCTDataset(self.data_dir, self.train_ids)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        print('\nCreating validation data set.')
        val_dataset = MotionCTDataset(self.data_dir, self.val_ids)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        print('\nCreating test data set.')
        test_dataset = MotionCTDataset(self.data_dir, self.test_ids)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

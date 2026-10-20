from typing import Callable, Dict, Optional, Tuple
import numpy as np
import spikingjelly.datasets as sjds
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import shutil
import spikingjelly.configure as configure
from spikingjelly.datasets import np_savez

class HARDVS(sjds.NeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """
        """
        assert train is not None
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)

    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        url = 'https://github.com/Event-AHU/HARDVS'
        return [
            ('MINI_HARDVS_files.zip', url, '9c4cc0d9ba043faa17f6f1a9e9aff982'),
            ('test_label.txt', url, '5b664af5843f9b476a9c22626f7f5a59'),
            ('train_label.txt', url, '0d642b6e6871034f151b2649a89d8d3c'),
            ('val_label.txt', url, 'cd2cebcba80e4552102bbacf2b5df812'),

        ]

    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return False


    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None

        This function defines how to extract download files.
        '''
        temp_ext_dir = os.path.join(download_root, 'temp_ext')
        os.mkdir(temp_ext_dir)
        print(f'Mkdir [{temp_ext_dir}].')
        extract_archive(os.path.join(download_root, 'MINI_HARDVS_files.zip'), temp_ext_dir)
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 2)) as tpe:
            sub_threads = []
            for i in range(1, 301):
                zip_file = os.path.join(temp_ext_dir, 'MINI_HARDVS_files',  'action_' + str(i).zfill(3) + '.zip')
                target_dir = os.path.join(extract_root, 'action_' + str(i).zfill(3))
                print(f'Extract [{zip_file}] to [{target_dir}].')
                sub_threads.append(tpe.submit(extract_archive, zip_file, target_dir))

            for sub_thread in sub_threads:
                if sub_thread.exception():
                    print(sub_thread.exception())
                    exit(-1)

        shutil.rmtree(temp_ext_dir)
        print(f'Rmtree [{temp_ext_dir}].')

        shutil.copy(os.path.join(download_root, 'test_label.txt'), os.path.join(extract_root, 'test_label.txt'))
        shutil.copy(os.path.join(download_root, 'train_label.txt'), os.path.join(extract_root, 'train_label.txt'))
        shutil.copy(os.path.join(download_root, 'val_label.txt'), os.path.join(extract_root, 'val_label.txt'))

    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        extract_root = '/home/chenxiangma/projects/datasets/hardvs/extract'
        events_np_root = '/home/chenxiangma/projects/datasets/hardvs/events_np'

        for prefix in ('train', 'val', 'test'):
            target_dir = os.path.join(events_np_root, prefix)
            os.mkdir(target_dir)
            print(f'Mkdir {target_dir}.')
            for label in range(300):
                class_dir = os.path.join(target_dir, str(label))
                os.mkdir(class_dir)
                print(f'Mkdir {class_dir}.')

            with open(os.path.join(extract_root, f'{prefix}_label.txt')) as f:
                for line in f.readlines():
                    # e.g., "action_001/dvSave-2021_10_15_19_18_02"
                    class_name, sample_name = line.split(' ')[0].split('/')
                    source_file = os.path.join(extract_root, class_name, sample_name, f'{sample_name}.npz')
                    class_name = int(class_name.split('_')[-1]) - 1
                    if os.path.exists(source_file):
                        target_file = os.path.join(target_dir, str(class_name), f'{sample_name}.npz')
                        print(f"target: {target_file}")
                        shutil.copyfile(source_file, target_file)





    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        return np.load(file_name)

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 260, 346


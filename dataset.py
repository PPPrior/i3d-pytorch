import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class I3DDataSet(data.Dataset):
    def __init__(self, root_path, list_file, clip_length=64, frame_size=(320, 240),
                 modality='RGB', image_tmpl='img_{:05d}.jpg',
                 transform=None, random_shift=True, test_mode=False):
        self.root_path = root_path
        self.list_file = list_file
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self._parse_list()

    def _load_image(self, directory, idx):
        root_path = os.path.join(self.root_path, 'rawframes/')  # ../data/ucf101/rawframes/
        directory = os.path.join(root_path, directory)

        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in
                           open(os.path.join(self.root_path, self.list_file))]

    def _sample_index(self, record):
        tick = record.num_frames - self.clip_length
        if tick < 0:
            index = 0
        else:
            if not self.test_mode:
                index = randint(tick + 1) if self.random_shift else int(tick / 2.0)
            else:
                index = int(tick / 2.0)
        return index + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        idx = self._sample_index(record)
        return self.get(record, idx)

    def get(self, record, index):
        images = list()
        p, seg_imgs = int(index), None
        for i in range(self.clip_length):
            if p < record.num_frames:
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                p += 1
            else:
                images.extend(seg_imgs)

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

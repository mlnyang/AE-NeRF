import os
import logging
from torch.utils import data
import numpy as np
import glob
from PIL import Image
from torchvision import transforms
import lmdb
import pickle
import string
import io
import torch 
import random
# fix for broken images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from random import randrange
import imageio

logger = logging.getLogger(__name__)


class LSUNClass(data.Dataset):
    ''' LSUN Class Dataset Class.

    Args:
        dataset_folder (str): path to LSUN dataset
        classes (str): class name
        size (int): image output size
        random_crop (bool): whether to perform random cropping
        use_tanh_range (bool): whether to rescale images to [-1, 1]
    '''

    def __init__(self, dataset_folder,
                 classes='scene_categories/church_outdoor_train_lmdb',
                 size=64, random_crop=False, use_tanh_range=False):
        root = os.path.join(dataset_folder, classes)

        # Define transforms
        if random_crop:
            self.transform = [
                transforms.Resize(size),
                transforms.RandomCrop(size),
            ]
        else:
            self.transform = [
                transforms.Resize((size, size)),
            ]
        self.transform += [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        if use_tanh_range:
            self.transform += [transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(self.transform)

        import time
        t0 = time.time()
        print('Start loading lmdb file ...')
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = '_cache_' + ''.join(
            c for c in root if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(
                    keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))
        print('done!')
        t = time.time() - t0
        print('time', t)
        print("Found %d files." % self.length)

    def __getitem__(self, idx):
        try:
            img = None
            env = self.env
            with env.begin(write=False) as txn:
                imgbuf = txn.get(self.keys[idx])

            buf = io.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            data = {
                'image': img
            }
            return data

        except Exception as e:
            print(e)
            idx = np.random.randint(self.length)
            return self.__getitem__(idx)

    def __len__(self):
        return self.length


class ImagesDataset(data.Dataset):
    ''' Default Image Dataset Class.

    Args:
        dataset_folder (str): path to LSUN dataset
        size (int): image output size
        celebA_center_crop (bool): whether to apply the center
            cropping for the celebA and celebA-HQ datasets.
        random_crop (bool): whether to perform random cropping
        use_tanh_range (bool): whether to rescale images to [-1, 1]
    '''

    def __init__(self, dataset_folder,  size=64, celebA_center_crop=False,
                 random_crop=False, use_tanh_range=False):

        self.size = size
        assert(not(celebA_center_crop and random_crop))
        self.transform = [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
        if use_tanh_range:
            self.transform += [
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(self.transform)

        self.data_type = os.path.basename(dataset_folder).split(".")[-1]
        assert(self.data_type in ["jpg", "png", "npy"])

        import time
        t0 = time.time()
        print('Start loading file addresses ...')
        images = glob.glob(dataset_folder)
        random.shuffle(images)
        t = time.time() - t0
        print('done! time:', t)
        print("Number of images found: %d" % len(images))

        self.images = images
        self.length = len(images)
        self.pose_path = '/workspace/media/dataset2/carla/carla_poses'
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )


    def __getitem__(self, idx):
        try:
            buf = self.images[idx]
            if self.data_type == 'npy':
                img = np.load(buf)[0].transpose(1, 2, 0)
                img = Image.fromarray(img).convert("RGB")
            else:
                img = Image.open(buf).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            idx_sample = buf.split('/')[-1].split('.')[0]
            pose_path = f'{self.pose_path}/{idx_sample}_extrinsics.npy'
            pose = torch.from_numpy(np.load(pose_path)).float()
            pose = torch.cat((pose, torch.tensor([-0.0, 0.0, -0.0, 1.0]).unsqueeze(0)), dim=0)
            # pose = pose @ self._coord_trans       # ?????? ?????? ?????????????????? ??? ?????? -> ????????? 0???

            # normalize 
            pose[:3, 3] = pose[:3, 3] / torch.norm(pose[:3, 3], dim=-1)# * 2.732     # ???????????????..      <- ?????? ????????? ??? ??????????????????..? 
        
            data = {
                'image': img, 
                'pose': pose
            }
            
            return data

        except Exception as e:
            print(e)
            print("Warning: Error occurred when loading file %s" % buf)
            return self.__getitem__(np.random.randint(self.length))

    def __len__(self):
        return self.length


# import os
# import logging
# from torch.utils import data
# import numpy as np
# import glob
# from PIL import Image
# from torchvision import transforms
# import lmdb
# import pickle
# import string
# import io
# import torch 
# import random
# # fix for broken images
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# from random import randrange
# import imageio

# logger = logging.getLogger(__name__)


# ### for original CARLA dataset 
# class ImagesDataset(data.Dataset):
#     ''' Default Image Dataset Class.

#     Args:
#         dataset_folder (str): path to LSUN dataset
#         size (int): image output size
#         celebA_center_crop (bool): whether to apply the center
#             cropping for the celebA and celebA-HQ datasets.
#         random_crop (bool): whether to perform random cropping
#         use_tanh_range (bool): whether to rescale images to [-1, 1]
#     '''

#     def __init__(self, dataset_folder,  size=64, celebA_center_crop=False,
#                  random_crop=False, use_tanh_range=False):

#         self.size = size
#         assert(not(celebA_center_crop and random_crop))     # random crop??? False??? ???????????? ?????? 

#         self.transform = [
#             transforms.Resize((size, size)),
#             transforms.ToTensor(),
#         ]
#         if use_tanh_range:
#             self.transform += [
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#         self.transform = transforms.Compose(self.transform)

#         self.img_folder = os.path.join(dataset_folder, 'carla_images')
#         self.pose_folder = os.path.join(dataset_folder, 'carla_poses')

#         self.data_type = os.path.basename(f'{self.img_folder}/*.png').split(".")[-1]
#         assert(self.data_type in ["png", "png", "npy"])

#         import time
#         t0 = time.time()
#         print('Start loading file addresses ...')
#         images = glob.glob(f'{self.img_folder}/*.png')
#         # random.shuffle(images)        # shuffle??? ??????..? 
#         t = time.time() - t0
#         print('done! time:', t)
#         print("Number of images found: %d" % len(images))

#         self.images = images
#         self.length = len(self.images)

#     def __getitem__(self, idx):
#         try:
#             buf = self.images[idx]
#             if self.data_type == 'npy':
#                 img = np.load(buf)[0].transpose(1, 2, 0)
#                 img = Image.fromarray(img).convert("RGB")
#             else:
#                 img = Image.open(buf).convert('RGB')

#             if self.transform is not None:
#                 img = self.transform(img)
            
#             idx_sample = buf.split('/')[-1].split('.')[0]
#             pose_path = f'{self.pose_folder}/{idx_sample}_extrinsics.npy'
#             pose = np.load(self.pose_path)
#             print(f'pose shape {pose.shape}')
#             data = {
#                 'image': img,
#                 'pose': pose
#             }
#             return data
#         except Exception as e:
#             print(e)
#             print("Warning: Error occurred when loading file %s" % buf)
#             return self.__getitem__(np.random.randint(self.length))

#     def __len__(self):
#         return self.length


# class LSUNClass(data.Dataset):
#     ''' LSUN Class Dataset Class.

#     Args:
#         dataset_folder (str): path to LSUN dataset
#         classes (str): class name
#         size (int): image output size
#         random_crop (bool): whether to perform random cropping
#         use_tanh_range (bool): whether to rescale images to [-1, 1]
#     '''

#     def __init__(self, dataset_folder,
#                  classes='scene_categories/church_outdoor_train_lmdb',
#                  size=64, random_crop=False, use_tanh_range=False):
#         root = os.path.join(dataset_folder, classes)

#         # Define transforms
#         if random_crop:
#             self.transform = [
#                 transforms.Resize(size),
#                 transforms.RandomCrop(size),
#             ]
#         else:
#             self.transform = [
#                 transforms.Resize((size, size)),
#             ]
#         self.transform += [
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ]
#         if use_tanh_range:
#             self.transform += [transforms.Normalize(
#                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#         self.transform = transforms.Compose(self.transform)

#         import time
#         t0 = time.time()
#         print('Start loading lmdb file ...')
#         self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
#                              readahead=False, meminit=False)
#         with self.env.begin(write=False) as txn:
#             self.length = txn.stat()['entries']
#         cache_file = '_cache_' + ''.join(
#             c for c in root if c in string.ascii_letters)
#         if os.path.isfile(cache_file):
#             self.keys = pickle.load(open(cache_file, "rb"))
#         else:
#             with self.env.begin(write=False) as txn:
#                 self.keys = [key for key in txn.cursor().iternext(
#                     keys=True, values=False)]
#             pickle.dump(self.keys, open(cache_file, "wb"))
#         print('done!')
#         t = time.time() - t0
#         print('time', t)
#         print("Found %d files." % self.length)

#     def __getitem__(self, idx):
#         try:
#             img = None
#             env = self.env
#             with env.begin(write=False) as txn:
#                 imgbuf = txn.get(self.keys[idx])

#             buf = io.BytesIO()
#             buf.write(imgbuf)
#             buf.seek(0)
#             img = Image.open(buf).convert('RGB')

#             if self.transform is not None:
#                 img = self.transform(img)

#             data = {
#                 'image': img
#             }
#             return data

#         except Exception as e:
#             print(e)
#             idx = np.random.randint(self.length)
#             return self.__getitem__(idx)

#     def __len__(self):
#         return self.length










# # ### for original CARLA dataset 
# # class ImagesDataset(data.Dataset):
# #     ''' Default Image Dataset Class.

# #     Args:
# #         dataset_folder (str): path to LSUN dataset
# #         size (int): image output size
# #         celebA_center_crop (bool): whether to apply the center
# #             cropping for the celebA and celebA-HQ datasets.
# #         random_crop (bool): whether to perform random cropping
# #         use_tanh_range (bool): whether to rescale images to [-1, 1]
# #     '''

# #     def __init__(self, dataset_folder,  size=64, celebA_center_crop=False,
# #                  random_crop=False, use_tanh_range=False):

# #         self.size = size
# #         assert(not(celebA_center_crop and random_crop))
# #         if random_crop:
# #             self.transform = [
# #                 transforms.Resize(size),
# #                 transforms.RandomCrop(size),
# #                 transforms.RandomHorizontalFlip(),
# #                 transforms.ToTensor(),
# #             ]
# #         elif celebA_center_crop:
# #             if size <= 128:  # celebA
# #                 crop_size = 108
# #             else:  # celebAHQ
# #                 crop_size = 650
# #             self.transform = [
# #                 transforms.CenterCrop(crop_size),
# #                 transforms.Resize((size, size)),
# #                 transforms.RandomHorizontalFlip(),
# #                 transforms.ToTensor()
# #             ]
# #         else:
# #             self.transform = [
# #                 transforms.Resize((size, size)),
# #                 transforms.RandomHorizontalFlip(),
# #                 transforms.ToTensor(),
# #             ]
# #         if use_tanh_range:
# #             self.transform += [
# #                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# #         self.transform = transforms.Compose(self.transform)

# #         self.data_type = os.path.basename(dataset_folder).split(".")[-1]
# #         assert(self.data_type in ["jpg", "png", "npy"])

# #         import time
# #         t0 = time.time()
# #         print('Start loading file addresses ...')
# #         images = glob.glob(dataset_folder)
# #         random.shuffle(images)
# #         t = time.time() - t0
# #         print('done! time:', t)
# #         print("Number of images found: %d" % len(images))

# #         self.images = images
# #         self.length = len(images)

# #     def __getitem__(self, idx):
# #         try:
# #             buf = self.images[idx]
# #             if self.data_type == 'npy':
# #                 img = np.load(buf)[0].transpose(1, 2, 0)
# #                 img = Image.fromarray(img).convert("RGB")
# #             else:
# #                 img = Image.open(buf).convert('RGB')

# #             if self.transform is not None:
# #                 img = self.transform(img)
# #             data = {
# #                 'image': img
# #             }
# #             return data
# #         except Exception as e:
# #             print(e)
# #             print("Warning: Error occurred when loading file %s" % buf)
# #             return self.__getitem__(np.random.randint(self.length))

# #     def __len__(self):
# #         return self.length


# # class LSUNClass(data.Dataset):
# #     ''' LSUN Class Dataset Class.

# #     Args:
# #         dataset_folder (str): path to LSUN dataset
# #         classes (str): class name
# #         size (int): image output size
# #         random_crop (bool): whether to perform random cropping
# #         use_tanh_range (bool): whether to rescale images to [-1, 1]
# #     '''

# #     def __init__(self, dataset_folder,
# #                  classes='scene_categories/church_outdoor_train_lmdb',
# #                  size=64, random_crop=False, use_tanh_range=False):
# #         root = os.path.join(dataset_folder, classes)

# #         # Define transforms
# #         if random_crop:
# #             self.transform = [
# #                 transforms.Resize(size),
# #                 transforms.RandomCrop(size),
# #             ]
# #         else:
# #             self.transform = [
# #                 transforms.Resize((size, size)),
# #             ]
# #         self.transform += [
# #             transforms.RandomHorizontalFlip(),
# #             transforms.ToTensor(),
# #         ]
# #         if use_tanh_range:
# #             self.transform += [transforms.Normalize(
# #                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# #         self.transform = transforms.Compose(self.transform)

# #         import time
# #         t0 = time.time()
# #         print('Start loading lmdb file ...')
# #         self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
# #                              readahead=False, meminit=False)
# #         with self.env.begin(write=False) as txn:
# #             self.length = txn.stat()['entries']
# #         cache_file = '_cache_' + ''.join(
# #             c for c in root if c in string.ascii_letters)
# #         if os.path.isfile(cache_file):
# #             self.keys = pickle.load(open(cache_file, "rb"))
# #         else:
# #             with self.env.begin(write=False) as txn:
# #                 self.keys = [key for key in txn.cursor().iternext(
# #                     keys=True, values=False)]
# #             pickle.dump(self.keys, open(cache_file, "wb"))
# #         print('done!')
# #         t = time.time() - t0
# #         print('time', t)
# #         print("Found %d files." % self.length)

# #     def __getitem__(self, idx):
# #         try:
# #             img = None
# #             env = self.env
# #             with env.begin(write=False) as txn:
# #                 imgbuf = txn.get(self.keys[idx])

# #             buf = io.BytesIO()
# #             buf.write(imgbuf)
# #             buf.seek(0)
# #             img = Image.open(buf).convert('RGB')

# #             if self.transform is not None:
# #                 img = self.transform(img)

# #             data = {
# #                 'image': img
# #             }
# #             return data

# #         except Exception as e:
# #             print(e)
# #             idx = np.random.randint(self.length)
# #             return self.__getitem__(idx)

# #     def __len__(self):
# #         return self.length


# # class ImagesDataset(data.Dataset):
# #     ''' Default Image Dataset Class.

# #     Args:
# #         dataset_folder (str): path to LSUN dataset
# #         size (int): image output size
# #         celebA_center_crop (bool): whether to apply the center
# #             cropping for the celebA and celebA-HQ datasets.
# #         random_crop (bool): whether to perform random cropping
# #         use_tanh_range (bool): whether to rescale images to [-1, 1]
# #     '''

# #     def __init__(self, dataset_folder,  size=64, celebA_center_crop=False,
# #                  random_crop=False, use_tanh_range=False):

# #         self.size = size
# #         assert(not(celebA_center_crop and random_crop)) # ?????? False?????? ??? 
# #         self.transform = [
# #             transforms.Resize((size, size)),
# #             transforms.ToTensor(),
# #         ]

# #         if use_tanh_range:
# #             self.transform += [
# #                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# #         self.transform = transforms.Compose(self.transform)

# #         self.image_to_tensor = self.get_image_to_tensor_balanced()
# #         import time
# #         t0 = time.time()
# #         print('Start loading file addresses ...')   # ??? 12?????? 
# #         base_path = dataset_folder        # <- ????????? 
        
# #         t = time.time() - t0    
# #         print('done! time:', t)

# #         self._coord_trans = torch.diag(
# #             torch.tensor([1, -1, -1, 1], dtype=torch.float32)
# #         )
# #         self.intrins = sorted(
# #             glob.glob(os.path.join(base_path, "*", "intrinsics.txt"))
# #         )

# #     def get_image_to_tensor_balanced(self, image_size=0):
# #         ops = []
# #         if image_size > 0:
# #             ops.append(transforms.Resize(image_size))
# #         ops.extend(
# #             [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
# #         )
# #         return transforms.Compose(ops)

# #     def __getitem__(self, index):     # ?????? pixelnerf??? ???????????? ????????? ????????? ??? 
# #         intrin_path = self.intrins[index]
# #         dir_path = os.path.dirname(intrin_path)
# #         rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
# #         pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))

# #         total_len = len(rgb_paths)
# #         img_idx = randrange(total_len)

# #         img = Image.open(rgb_paths[img_idx]).convert('RGB')
# #         img_tensor = self.transform(img)

# #         pose = torch.from_numpy(
# #             np.loadtxt(pose_paths[img_idx], dtype=np.float32).reshape(4, 4)
# #         )
# #         pose = pose @ self._coord_trans
        
# #         data = {
# #             'image': img_tensor,
# #             'pose': pose
# #         }
# #         return data

# #     def __len__(self):
# #         return len(self.intrins)

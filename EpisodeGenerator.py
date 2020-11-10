import typing as _typing
import os
import random
import numpy as np
from PIL import Image

from _utils import list_dir, list_files

def _load_image(img_url: str, expand_dim: bool = False) -> np.ndarray:
    """Load an image
    """
    img = Image.open(fp=img_url, mode='r')
    img_np = np.asarray(a=img, dtype=np.uint8)

    if len(img_np.shape) < 3: # gray scale image
        # insert channel into the array
        img_np = np.expand_dims(img_np, axis=0)

        if expand_dim:
            img_np = np.repeat(a=img_np, repeats=3, axis=0)
    else:
        img_np = np.transpose(a=img_np, axes=(2, 0, 1)) # convert to (c, h, w)
        img_np = img_np / 255 # normalize to [0, 1]

    return img_np

def _sample_uniform(num_samples: int, low: float, high: float) -> np.ndarray:
    return (low - high) * np.random.random_sample(size=num_samples) + high

def get_cls_img(root: str, suffix: str) -> dict:
    """Get folders from root, and images in each folder

    Args:
        root (str): the desired directory
        suffix (str): the suffix of file or image within each folder

    Returns: dictionary with keys are the folder names,
        and values are the lists of files within the corresponding folder
    """
    cls_img = dict.fromkeys(list_dir(root=root))
    for dir_ in cls_img:
        cls_img[dir_] = list_files(root=os.path.join(root, dir_), suffix=suffix)

    return cls_img

class OmniglotLoader(object):
    folder = '../datasets/omniglot-py/'

    def __init__(
        self,
        root: str = folder,
        train_subset: bool = True,
        suffix: str = '.png',
        min_num_cls: int = 5,
        max_num_cls: int = 20,
        k_shot: int = 20,
        expand_dim: bool = False,
        load_images: bool = True,
        xml_url: _typing.Optional[str] = None
    ) -> None:
        """Initialize a data loader for Omniglot data set or a two-level dataset
            with structure similar to Omniglot: alphabet -> character -> image

        Args:
            root (str): path to the folder of Omniglot data set
            train_subset (bool): if True, this will load data from
                the ``images_background`` folder (or, training set). If False,
                it loads data from ``images_evaluation``` (or, validation set)
            suffix (str): the suffix of images
            min_num_cls (int): minimum number of classes within a generated episode
            max_num_cls (int): maximum number of classes within a generated episode
            expand_dim (bool): if True, repeat the channel dimension from 1 to 3
            load_images (bool): if True, this will place all image data (PIL) on RAM.
                This option is optimal for small data set since it would speed up
                the data loading process. If False, it will load images whenever called.
                This option is suitable for large data set.
            xml_url: dummy keyword to be consistent to other classes

        Returns: an OmniglotLoader instance
        """
        self.root = os.path.join(root, 'images_background' if train_subset else 'images_evaluation')
        self.suffix = suffix
        self.min_num_cls = min_num_cls
        self.max_num_cls = max_num_cls
        self.k_shot = k_shot
        self.expand_dim = expand_dim
        self.load_images = load_images

        # create a nested dictionary to store data
        self.data = dict.fromkeys(list_dir(root=self.root))
        for alphabet in self.data:
            self.data[alphabet] = dict.fromkeys(list_dir(root=os.path.join(self.root, alphabet)))

            # loop through each alphabet
            for character in self.data[alphabet]:
                self.data[alphabet][character] = []

                # loop through all images in an alphabet character
                for img_name in list_files(root=os.path.join(self.root, alphabet, character), suffix=suffix):
                    if self.load_images:
                        # load images
                        img = _load_image(img_url=os.path.join(self.root, alphabet, character, img_name), expand_dim=self.expand_dim)
                    else:
                        img = img_name

                    self.data[alphabet][character].append(img)

    def generate_episode(self, episode_name: _typing.Optional[_typing.List[str]]) -> _typing.List[np.ndarray]:
        """Generate an episode of data and labels

        Args:
            episode_name (List(str)): list of string where the first one is
                the name of the alphabet, and the rest are character names.
                If None, it will randomly pick characters from the given alphabet
            num_imgs (int): number of images per character

        Returns:
            x (List(numpy array)): list of numpy array representing data
        """
        x = []

        if episode_name is None:

            alphabet = random.sample(population=self.data.keys(), k=1)[0]

            max_n_way = min(len(self.data[alphabet]), self.max_num_cls)

            assert self.min_num_cls <= max_n_way

            n_way = random.randint(a=self.min_num_cls, b=max_n_way)
            n_way = min(n_way, self.max_num_cls)

            characters = random.sample(population=self.data[alphabet].keys(), k=n_way)
        else:
            alphabet = episode_name[0]
            characters = episode_name[1:]

        for character in characters:
            x_temp = random.sample(population=self.data[alphabet][character], k=self.k_shot)
            if self.load_images:
                x.append(x_temp)
            else:
                x_ = [_load_image(
                    img_url=os.path.join(self.root, alphabet, character, img_name),
                    expand_dim=self.expand_dim
                ) for img_name in x_temp]
                x.append(x_)

        return x

class ImageFolderGenerator(object):
    def __init__(
        self,
        root: str,
        train_subset: bool = True,
        suffix: str = '.jpg',
        min_num_cls: int = 5,
        max_num_cls: int = 20,
        k_shot: int = 16,
        expand_dim: bool = False,
        load_images: bool = False,
        xml_url: _typing.Optional[str] = None
    ) -> None:
        """Initialize a dataloader instance for image folder structure

        Args:
            root (str): location containing ``train`` and ``test`` folders,
                where each folder contains a number of image folders
            train_subset (bool): If True, take data from ``train`` folder,
                else ``test`` folder
            suffix (str): the suffix of all images
            min_num_cls (int): minimum number of classes to pick to form an episode
            max_num_cls (int): maximum number of classes to pick to form an episode
            expand_dim (bool): useful for black and white images only
                (convert from 1 channel to 3 channels)
            load_images (bool): load images to RAM. Set True if dataset is small
            xml_url (str): location of the XML structure

        """
        self.root = os.path.join(root, 'train' if train_subset else 'test')
        self.suffix = suffix
        self.k_shot = k_shot
        self.expand_dim = expand_dim
        self.load_images = load_images
        self.xml_url = xml_url

        self.data = self.get_data()

        assert min_num_cls <= len(self.data)
        self.min_num_cls = min_num_cls
        self.max_num_cls = min(max_num_cls, len(self.data))

    def get_data(self):
        """Get class-image data stored in a dictionary
        """
        data_str = get_cls_img(root=self.root, suffix=self.suffix)

        if not self.load_images:
            return data_str

        cls_img_data = dict.fromkeys(data_str.keys())
        for cls_ in data_str:
            temp = [0] * len(data_str[cls_])
            for i, img_name in enumerate(data_str[cls_]):
                img = _load_image(
                    img_url=os.path.join(self.root, cls_, img_name),
                    expand_dim=self.expand_dim
                )
                temp[i] = img
            cls_img_data[cls_] = list(temp)

        return cls_img_data

    def generate_episode(self, episode_name: _typing.Optional[_typing.List[str]]) -> _typing.List[np.ndarray]:
        """Generate an episode
        Args:
            episode_name (str): a list of classes to form the episode.
                If None, sample a random list of classes.

        Returns:
            x (list(numpy array)): list of images loaded in numpy array form
        """
        x = []

        if episode_name is not None:
            cls_list = episode_name
        elif self.xml_url is None:
            cls_list = random.sample(
                population=self.data.keys(),
                k=random.randint(a=self.min_num_cls, b=self.max_num_cls)
            )
        else:
            raise ValueError('Not implemeted yet')

        for cls_ in cls_list:
            x_temp = random.sample(population=self.data[cls_], k=self.k_shot)
            if self.load_images:
                x.append(x_temp)
            else:
                x_ = [_load_image(
                    img_url=os.path.join(self.root, cls_, img_name),
                    expand_dim=self.expand_dim
                ) for img_name in x_temp]
                x.append(x_)

        return x

class SineLineGenerator():
    """Create an instance of generator generating
        sinusoidal data, or linear data
    """
    def __init__(self, num_samples: int = 5, config: _typing.Optional[dict] = None) -> None:
        """Initialize

        Args:
            num_samples: number of samples to generate
            config: dictionary to configure the data ranges and others
        """
        self.num_samples = num_samples

        if config is None:
            config = {}

        self.input_range = config.get('input_range', [-5., 5.])
        self.noise_std = config.get('noise_std', 0.3)

        self.amp_range = config.get('amp_range', [.5, 5.0])
        self.phase_range = config.get('phase_range', [0., np.pi])

        self.slope_range = config.get('slope_range', [-3., 3.])
        self.intercept_range = config.get('intercept_range', [-3., 3.])

        self.x0 = np.linspace(
            start=self.input_range[0],
            stop=self.input_range[1],
            num=100
        )

    def generate_sinusoidal_data(self) -> _typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate sinusoidal data

        Args:

        Returns:
            x:
            y:
            y_noisy:
        """
        amp = _sample_uniform(
            num_samples=1,
            low=self.amp_range[0],
            high=self.amp_range[1]
        )
        phase = _sample_uniform(
            num_samples=1,
            low=self.phase_range[0],
            high=self.phase_range[1]
        )

        x = _sample_uniform(
            num_samples=self.num_samples,
            low=self.input_range[0],
            high=self.input_range[1]
        )
        y = amp * np.sin(x + phase)
        y0 = amp * np.sin(self.x0 + phase)

        noise = np.random.randn(self.num_samples)

        y_noisy = y + self.noise_std * noise

        return x, y_noisy, y0

    def generate_linear_data(self) -> _typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate sinusoidal data

        Args:

        Returns:
            x:
            y:
            y_noisy:
        """
        slope = _sample_uniform(
            num_samples=1,
            low=self.slope_range[0],
            high=self.slope_range[1]
        )
        intercept = _sample_uniform(
            num_samples=1,
            low=self.intercept_range[0],
            high=self.intercept_range[1]
        )

        x = _sample_uniform(
            num_samples=self.num_samples,
            low=self.input_range[0],
            high=self.input_range[1]
        )
        y = slope * x + intercept
        y0 = slope * self.x0 + intercept

        noise = np.random.randn(self.num_samples)

        y_noisy = y + self.noise_std * noise

        return x, y_noisy, y0

    def generate_episode(self, episode_name: _typing.Optional[_typing.List[str]] = None) -> _typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate episode

        Args:
            episode_name: either 'line' or 'sine'. If None, randomly pick

        Returns:
            x:
            y:
            y_noisy:
        """
        if episode_name == 'sine':
            p = 0
        elif episode_name == 'line':
            p = 0.5
        else:
            p = random.random()

        if p < 0.5:
            # sine
            x, y_noisy, y0 = self.generate_sinusoidal_data()
        else:
            # line
            x, y_noisy, y0 = self.generate_linear_data()

        return x[:, np.newaxis], y_noisy[:, np.newaxis], y0

import torch

import random
import typing

from scipy.special import comb

class EpisodeSampler(torch.utils.data.BatchSampler):
    """Sample data to form a classification task

    Args:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, sampler: torch.utils.data.Sampler[typing.List[int]], num_ways: int, num_samples_per_class: int, drop_last: bool = True) -> None:
        super().__init__(sampler=sampler, batch_size=num_ways, drop_last=drop_last)
        
        self.num_ways = num_ways
        self.num_samples_per_class = num_samples_per_class
        
        # create a list of dictionary, each has:
        # - key = label
        # - value = img_idx
        self.class_img_idx = [None] * len(sampler.data_source.datasets)
        j = 0 # track the length of each dataset
        for dataset_id in range(len(sampler.data_source.datasets)):
            self.class_img_idx[dataset_id] = {}
            for i in range(len(self.sampler.data_source.datasets[dataset_id])):
                label_idx = self.sampler.data_source.datasets[dataset_id].targets[i]
                if label_idx not in self.class_img_idx[dataset_id]:
                    self.class_img_idx[dataset_id][label_idx] = []
                self.class_img_idx[dataset_id][label_idx].append(i + j)
            j = len(sampler.data_source.datasets[dataset_id])
        

    def __iter__(self) -> typing.Iterator[typing.List[int]]:
        while(True):
            # randomly sample a dataset
            dataset_id = random.randint(a=0, b=len(self.sampler.data_source.datasets) - 1)

            # n-way
            labels = random.sample(population=self.class_img_idx[dataset_id].keys(), k=self.num_ways)

            # variable to store img idx
            batch = []

            for label in labels:
                batch.extend(random.sample(population=self.class_img_idx[dataset_id][label], k=self.num_samples_per_class))

            yield batch
            batch = []
            labels = []
    
    def __len__(self) -> int:
        return comb(N=len(self.label_list), k=self.batch)
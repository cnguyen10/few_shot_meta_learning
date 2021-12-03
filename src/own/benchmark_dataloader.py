import torch
import typing
from src.own.create_benchmarks import create_benchmarks

class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, benchmark) -> None:
        super().__init__()
        self.benchmark = benchmark

    def __len__(self) -> int:
        return self.benchmark.n_task

    def __getitem__(self, index) -> typing.List[torch.Tensor]:
        task = self.benchmark.get_task_by_index(index)
        x = torch.tensor(task.x, dtype=torch.float32).squeeze()
        y = torch.tensor(task.y, dtype=torch.float32).squeeze()
        return [x, y]

def create_benchmark_dataloaders(config):
    bm_meta, bm_test = create_benchmarks(config)
    train_data_loader = torch.utils.data.DataLoader(BenchmarkDataset(bm_meta))
    test_data_loader = torch.utils.data.DataLoader(BenchmarkDataset(bm_test))
    return train_data_loader, test_data_loader
    


import torch
import typing

class SineDataset(torch.utils.data.Dataset):
    """generate data of a random sinusoidal task
    y = A sin(x + phi) + epsilon,
    where: epsilon is sampled from N(0, s^2)
    """
    def __init__(
        self,
        amplitude_range: typing.Tuple[float],
        phase_range: typing.Tuple[float],
        noise_std: float,
        x_range: typing.Tuple[float],
        num_samples: int
    ) -> None:
        """
        Args:
            amplitudes: a tuple consisting the range of A
            phase: a tuple consisting the range of phase
            noise_variance: the variance of the noise
        """
        super().__init__()
        
        self.amplitude_range = [a for a in amplitude_range]
        self.phase_range = [phi for phi in phase_range]
        self.noise_std = noise_std

        self.x = torch.linspace(start=x_range[0], end=x_range[1], steps=num_samples)

    def __len__(self) -> int:
        return 100000

    def __getitem__(self, index) -> typing.List[torch.Tensor]:
        """generate data of a task
        """
        y = self.generate_label()
        y = y + torch.randn_like(input=y) * self.noise_std

        return [self.x, y]
    
    def generate_label(self) -> torch.Tensor:
        """
        """
        # sample parameters randomly
        amplitude = torch.rand(size=(1,)) * (self.amplitude_range[1] - self.amplitude_range[0]) + self.amplitude_range[0]
        phase = torch.rand(size=(1,)) * (self.phase_range[1] - self.phase_range[0]) + self.phase_range[0]

        y = amplitude * torch.sin(input=self.x + phase)

        return y

class LineDataset(torch.utils.data.Dataset):
    """generate a data for a task following the formula:
    y = ax + b
    """
    def __init__(self, slope_range: typing.Tuple[float], intercept_range: typing.Tuple[float], x_range: typing.Tuple[float], num_samples: int, noise_std: float) -> None:
        super().__init__()
        
        self.slope_range = [a for a in slope_range]
        self.intercept_range = [b for b in intercept_range]
        self.noise_std = noise_std

        self.x = torch.linspace(start=x_range[0], end=x_range[1], steps=num_samples)

    def __getitem__(self, index) -> typing.List[torch.Tensor]:
        y = self.generate_label()

        y = y + torch.randn_like(input=y) * self.noise_std

        return [self.x, y]

    def __len__(self) -> int:
        return 100000

    def generate_label(self) -> torch.Tensor:
        slope = torch.rand(size=(1,)) * (self.slope_range[1] - self.slope_range[0]) + self.slope_range[0]
        intercept = torch.rand(size=(1,)) * (self.intercept_range[1] - self.intercept_range[0]) + self.intercept_range[0]

        y = slope * self.x + intercept

        return y
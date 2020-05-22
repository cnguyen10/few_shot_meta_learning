import numpy as np
import os
import random

def SampleUniform(r_size, low, high):
    return (low - high) * np.random.random_sample(size=r_size) + high

class DataGenerator(object):
    def __init__(self, num_samples, config={}):
        self.num_samples = num_samples

        self.input_range = config.get('input_range', [-5., 5.])

        self.amp_range = config.get('amp_range', [.5, 5.0])
        self.phase_range = config.get('phase_range', [0., np.pi])

        self.slope_range = config.get('slope_range', [-3., 3.])
        self.intercept_range = config.get('intercept_range', [-3., 3.])

        self.noise_std = 0.3
    
    def generate_sinusoidal_data(self, noise_flag=False):
        amp = SampleUniform(
            r_size=(1, 1),
            low=self.amp_range[0],
            high=self.amp_range[1]
        )
        phase = SampleUniform(
            r_size=(1, 1),
            low=self.phase_range[0],
            high=self.phase_range[1]
        )

        x_inputs = SampleUniform(
            r_size=(self.num_samples, 1),
            low=self.input_range[0],
            high=self.input_range[1]
        )
        outputs = amp * np.sin(x_inputs + phase)

        if noise_flag:
            output_noises = np.random.randn(self.num_samples, 1)
            outputs = outputs + self.noise_std * output_noises

        return x_inputs, outputs, amp, phase
    
    def generate_line_data(self, noise_flag=False):
        slope = SampleUniform(
            r_size=(1, 1),
            low=self.slope_range[0],
            high=self.slope_range[1]
        )
        intercept = SampleUniform(
            r_size=(1, 1),
            low=self.intercept_range[0],
            high=self.intercept_range[1]
        )

        x_inputs = SampleUniform(
            r_size=(self.num_samples, 1),
            low=self.input_range[0],
            high=self.input_range[1]
        )
        outputs = slope * x_inputs + intercept

        if noise_flag:
            output_noises = np.random.randn(self.num_samples, 1)
            outputs = outputs + self.noise_std * output_noises

        return x_inputs, outputs, slope, intercept

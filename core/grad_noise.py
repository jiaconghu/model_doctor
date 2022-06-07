import torch


class GradNoise:
    def __init__(self, module):
        self.module = module
        self.hook = None

    def add_noise(self):
        self.hook = self.module.register_forward_hook(_modify_feature_map)

    def remove_noise(self):
        self.hook.remove()


# keep forward after modify
def _modify_feature_map(module, inputs, outputs):
    noise = torch.randn(outputs.shape).to(outputs.device)
    # noise = torch.normal(mean=0, std=3, size=outputs.shape).to(outputs.device)

    outputs += noise

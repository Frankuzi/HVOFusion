import torch

class RandomSamples(object):
    # 用于图像像素的随机采样
    modes = ['random', 'importance_sampling']
    
    def __init__(self, views, num_views, mode='random', percentage=.5):
        if not mode in self.modes:
            raise ValueError(f"Unknown mode '{mode}'. Available modes are {', '.join(self.modes)}")
        self.mode = mode                        # 采样模式
        self.num_views = num_views
        self.img_size = views[0].color.shape[0] * views[0].color.shape[1]
        self.percentage = percentage
        self.loss_dict = {}
    
    def update_probabilities(self, losses, index):
        probabilities = losses.view(-1)
        probabilities = probabilities / torch.sum(probabilities)
        self.loss_dict[index] = probabilities

    def __call__(self, tensor, index):
        """ Select samples from a tensor.

        Args:
            tensor: Tensor to select samples from (HxWxC or NxC)
        """
        if self.mode == 'random':
            self.idx = torch.randperm(tensor.size(0))[:int(tensor.size(0)*self.percentage)]
        if self.mode == 'importance_sampling':
            if index not in self.loss_dict:
                self.idx = torch.randperm(tensor.size(0))   # 全采样
        return tensor.view(-1, tensor.shape[-1])[self.idx], self.idx
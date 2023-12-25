from argparse import ArgumentParser
import numpy as np
import random

class ViewSampler:
    modes = ['random', 'sequential', 'sequential_shuffled', 'importance_sampling']

    def __init__(self, views, mode, views_per_iter):
        if not mode in self.modes:
            raise ValueError(f"Unknown mode '{mode}'. Available modes are {', '.join(self.modes)}")
        self.mode = mode                        # 采样模式
        self.views_per_iter = views_per_iter    # 每次迭代用几个视角
        self.num_views = len(views)             # 一共有多少个视角

        self.current_index = 0
        self.index_buffer = list(range(self.num_views))
        self.probabilities = None               # importance_sampling的采样概率
        self.sampled_list = []                  # 记录进行一次importance_sampling的结果

    # @staticmethod
    # def add_arguments(parser: ArgumentParser):
    #     group = parser.add_argument_group("View Sampling")
    #     group.add_argument('--view_sampling_mode', type=str, choices=ViewSampler.modes, default='random', help="Mode used to sample views.")    # 选择数据集中不同视角的方法 可选"random" "sequential"
    #     group.add_argument('--views_per_iter', type=int, default=1, help="Number of views used per iteration.")     # 每次迭代选择几个视角
        
    @staticmethod
    def get_parameters(args):
        return { 
            "mode": args['view_sampling_mode'],        # 采样模式
            "views_per_iter": args['views_per_iter']   # 每次迭代用几个视角
        }
    
    def update_probabilities(self, losses):
        if self.current_index < len(self.sampled_list):
            return
        probabilities = np.array(losses)
        self.probabilities = probabilities / np.sum(probabilities) 

    def __call__(self, views):
        if self.mode == 'random':
            # Randomly select N views
            sampled_index = np.random.choice(len(views), self.views_per_iter, replace=False)
            sampled_views = [views[index] for index in sampled_index]
            return sampled_views, sampled_index
        elif self.mode == 'sequential':
            # Select N views by traversing the full set of views sequentially
            # (After the last view, start again from the first)
            sampled_views = []
            sampled_index = []
            for _ in range(self.views_per_iter):
                sampled_views += [views[self.current_index]]
                sampled_index += [self.current_index]
                self.current_index = (self.current_index + 1) % self.num_views
            return sampled_views, sampled_index
        elif self.mode == 'sequential_shuffled':
            # Select N views by traversing the full set of views sequentially, but in random order
            # (Each time the full set of views is traversed, randomly shuffle to create a new order)
            sampled_views = []
            sampled_index = []
            for _ in range(self.views_per_iter):
                view_index = self.index_buffer[self.current_index]
                sampled_views += [views[view_index]]
                sampled_index += [view_index]
                self.current_index = (self.current_index + 1) 
                if self.current_index >= self.num_views:
                    random.shuffle(self.index_buffer)
                    self.current_index = 0
            return sampled_views, sampled_index
        elif self.mode == 'importance_sampling':
            def sample_once():
                if self.probabilities is None:
                    self.current_index = 0
                else:
                    self.current_index = 0
                    self.sampled_list = np.random.choice(len(views), size=len(views), replace=True, p=self.probabilities)   # 更新概率表

            if len(self.sampled_list) == 0:
                self.sampled_list = np.arange(len(views))       # 初始采用顺序采样
                self.current_index = 0                          # 设置初始采样位置为0
            if self.current_index >= len(self.sampled_list):
                sample_once()
            sampled_index = self.sampled_list[self.current_index]
            sampled_views = [views[sampled_index]]
            self.current_index += 1
            return sampled_views, sampled_index
from argparse import ArgumentParser
import numpy as np
import random

class ViewSampler:
    modes = ['all', 'several']

    def __init__(self, views, mode='all', views_per_iter=1, picked_views=[]):
        if not mode in self.modes:
            raise ValueError(f"Unknown mode '{mode}'. Available modes are {', '.join(self.modes)}")
        if mode == 'several' and len(picked_views) == 0:
            raise ValueError(f"Empty picked_views when mode == several")
        
        self.mode = mode
        self.views_per_iter = views_per_iter


        if mode =='several':
            self.views = []
            for i, view in enumerate(views):
                if i in picked_views:
                    self.views.append(view)

        elif mode == 'all':
            self.views = views
        
        print(f"Selected {len(self.views)} views.")


    def __call__(self):
        return np.random.choice(self.views, self.views_per_iter, replace=False)

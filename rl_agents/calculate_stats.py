import torch


class stats:
    def __init__(self):
        self.stats = None

    def add_stats(self, additional_stats):
        if self.stats is None:
            self.stats = additional_stats
        else:
            self.stats['world_loss'] = torch.vstack((self.stats['world_loss'], additional_stats['world_loss']))
            self.stats['time_til_first_view'] = torch.hstack(
                (self.stats['time_til_first_view'], additional_stats['time_til_first_view']))
            self.stats['views_vel'] = torch.vstack((self.stats['views_vel'], additional_stats['views_vel']))
            self.stats['seen'] = torch.hstack((self.stats['seen'], additional_stats['seen']))
            self.stats['in_view']= torch.hstack((self.stats['in_view'], additional_stats['in_view']))
            self.stats['reinit']=torch.hstack((self.stats['reinit'], additional_stats['reinit']))

    def stats_analysis(self):
        stats = {
            'avg_time_til_first_view': float(self.stats['time_til_first_view'].mean()),
            'views_vel_corr': float(torch.corrcoef(self.stats['views_vel'].t())[0, 1]),
            'avg_world_loss': self.stats['world_loss'].mean(),
            'percent_targets_seen': self.stats['seen'].mean() * 100,
            'targets_in_view': self.stats['in_view'].mean() * 100,
            'reinitialized_targets' : self.stats['reinit'].mean()
        }
        return stats

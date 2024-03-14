import torch


class stats:
    def __init__(self):
        self.stats = None
    def add_stats(self, additional_stats):

        if self.stats is None:
            self.stats = additional_stats
        else:
            self.stats['world_loss'] = torch.vstack((self.stats['world_loss'], additional_stats['world_loss']))
            self.stats['time_til_first_view'] = torch.hstack((self.stats['time_til_first_view'], additional_stats['time_til_first_view']))
            self.stats['views_vel'] = torch.vstack((self.stats['views_vel'] , additional_stats['views_vel']))
    def stats_analysis(self):
        stats = {
           'avg_time_til_first_view': float(self.stats['time_til_first_view'].mean()),
           'views_vel_corr': float(torch.corrcoef(self.stats['views_vel'].t())[0,1]),
            'avg_world_loss': self.stats['world_loss'].mean()}
        print(stats)
        return stats

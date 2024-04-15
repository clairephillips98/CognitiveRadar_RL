import torch
from statistics import mean

class stats:
    def __init__(self):
        self.stats = None

    def add_stats(self, additional_stats,actions):
        if self.stats is None:
            self.stats = additional_stats
            if type(actions[0])==list:

                self.stats['actions'] = [len(set(map(tuple, actions)))]
            else:
                self.stats['actions'] = [len(set(actions))]
        else:
            self.stats['world_loss'] = torch.vstack((self.stats['world_loss'], additional_stats['world_loss']))
            self.stats['time_til_first_view'] = torch.hstack(
                (self.stats['time_til_first_view'], additional_stats['time_til_first_view']))
            self.stats['views_vel'] = torch.vstack((self.stats['views_vel'], additional_stats['views_vel']))
            self.stats['seen'] = torch.hstack((self.stats['seen'], additional_stats['seen']))
            if type(actions[0])==list:
                self.stats['actions'].append(len(set(map(tuple, actions))))

            else:
                self.stats['actions'].append(len(set(actions)))
    def stats_analysis(self):
        # slope of line
        x_mean = torch.mean(self.stats['views_vel'].t()[0])
        y_mean = torch.mean(self.stats['views_vel'].t()[1])
        dev_x = self.stats['views_vel'].t()[0] - x_mean
        dev_y = self.stats['views_vel'].t()[1] - y_mean
        slope = torch.sum(dev_x * dev_y) / torch.sum(dev_x ** 2)
        correlation = torch.corrcoef(self.stats['views_vel'].t())
        stats = {
            'avg_time_til_first_view': float(self.stats['time_til_first_view'].mean()),
            'views_vel_corr': float(correlation[0, 1]),
            'views_doppler_corr': float(correlation[0, 2]),
            'veiws_vel_slope': slope,
            'avg_world_loss': self.stats['world_loss'].mean(),
            'percent_targets_seen': self.stats['seen'].mean() * 100,
            'unique_actions': mean(self.stats['actions']),
            'average_view_rate': self.stats['views_vel'].t()[0].mean()
        }
        return stats

import torch
from statistics import mean

class stats:
    def __init__(self):
        self.stats = None

    def add_stats(self, additional_stats,actions):
        if self.stats is None:
            self.stats = {}
            for key in additional_stats.keys():
                self.stats[key] = [additional_stats[key]]
            if type(actions[0])==list:

                self.stats['actions'] = [len(set(map(tuple, actions)))]
            else:
                self.stats['actions'] = [len(set(actions))]
        else:
            for key in additional_stats.keys():
                self.stats[key].append(additional_stats[key])
            if type(actions[0])==list:
                self.stats['actions'].append(len(set(map(tuple, actions))))

            else:
                self.stats['actions'].append(len(set(actions)))
    def stats_analysis(self):
        # slope of line
        views_vel = torch.vstack(self.stats['views_vel']).t()
        time_til_first_view = torch.hstack(self.stats['time_til_first_view'])
        world_loss = torch.vstack(self.stats['world_loss'])
        seen = torch.hstack(self.stats['seen'])
        both_viewed =torch.hstack(self.stats['both_viewed'])
        x_mean = torch.mean(views_vel[0])
        y_mean = torch.mean(views_vel[1])
        dev_x =views_vel[0] - x_mean
        dev_y = views_vel[1] - y_mean
        slope = torch.sum(dev_x * dev_y) / torch.sum(dev_x ** 2)
        correlation = torch.corrcoef(views_vel)
        print(views_vel[0].mean())
        stats = {
            'avg_time_til_first_view': float(time_til_first_view.mean()),
            'views_vel_corr': float(correlation[0, 1]),
            'views_doppler_corr': float(correlation[0, 2]),
            'veiws_vel_slope': slope,
            'avg_world_loss': world_loss.mean(),
            'percent_targets_seen': seen.mean() * 100,
            'unique_actions': mean(self.stats['actions']),
            'average_view_rate': views_vel[0].mean(),
            'average_rate_of_viewed_by_both_radars': both_viewed.t()[0].mean()

        }
        return stats



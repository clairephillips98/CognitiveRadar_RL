import torch

def radar_stats(prev_stats, additional_stats):
    if 'time_til_first_view' not in prev_stats.keys():
        prev_stats['time_til_first_view']= additional_stats[0]
        prev_stats['views_vel'] = additional_stats[1]
    else:
        prev_stats['time_til_first_view'] += additional_stats[0]
        prev_stats['views_vel'] = torch.vstack((prev_stats['views_vel'] , additional_stats[1]))

    return prev_stats

def radar_stats_analysis(stats):
    stats = {
       'avg_time_til_first_view': float(stats['time_til_first_view'].mean()),
       'views_vel_corr': float(torch.corrcoef(stats['views_vel'].t())[0,1])}
    return stats

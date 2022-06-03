
import numpy as np
import time
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import os


class random_agent():

    def __init__(self, args, env, metrics, results_dir, randompose=False, num_beams=15):
        self.env=env
        self.metrics=metrics
        self.results_dir=results_dir

    def reset(self):
        self.counter=0

    def make_action(self, obs):
        
        actions=self.env.actions
        action=np.random.random_integers(actions)-1

        return action


    def store_entropy(self,entropy, T):
        self.metrics['steps'].append(T)
        self.metrics['entropy'].append(entropy)
        self._plot_line(self.metrics['steps'], self.metrics['entropy'], 'Entropy', path=self.results_dir)

            


    def _plot_line(self, xs, ys_population, title, path=''):
        max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

        #ys = torch.tensor(ys_population, dtype=torch.float32)
        #ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
        #ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

        trace_max = Scatter(x=xs, y=ys_population, line=Line(color=max_colour, dash='dash'), name='Max')
        #trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
        #trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
        #trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
        #trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

        plotly.offline.plot({
            'data': [trace_max],
            'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
        }, filename=os.path.join(path, title + '.html'), auto_open=False)
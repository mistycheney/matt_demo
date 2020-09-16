import os
import sys
import numpy as np
import pandas as pd
import functools
from IPython.display import clear_output
import ipywidgets as widgets
import matplotlib.pyplot as plt
import time

def train_model(data, output_model):
    print(f'Training {output_model} using provided data...')
    time.sleep(4)
    print('Training finished.')

def load_model(model_name):
    print(f'Loaded {model_name}')

# def retrieve_explanations(query, num):
#     return [c[0] for c in pd.read_csv('matt_demo_data/retrieved_explanations.txt', header=None).values]

# def retrieve_time_series(query, num):
    
#     ss = []
#     for i in range(1, 5):
#         s = pd.read_csv('matt_demo_data/retrieved_time_series_%d.csv' % i).values.T
#         l, r = pd.read_csv('matt_demo_data/retrieved_time_series_%d_range.csv' % i).values.T[0]
#         ss.append((s, (l, r), ['TSLA', 'APPL', 'GOOG', 'IBM'][i-1]))
        
#     return ss

def display_search_widgets(model):
    
    """Descriptive time series search"""

    def search_cb(_):
        text_query = text.value
        if len(text_query) == 0:
            return

#         global time_series_list
#         time_series_list = retrieve_time_series(text_query, num=4)
#         time_series_list = model.retrieve_time_series(text_query, num=4)

        time_series_list = []
        text_query_ind = ['pump severe leakage'].index(text_query)
        for i in range(5):
            retrieved_time_series = pd.read_csv('matt_demo_hydraulic_data/retrieved_time_series_%d_textquery_%d.csv' % (i, text_query_ind), index_col=0).values.T
            time_series_list.append(retrieved_time_series)

        with out_ts_res:
            out_ts_res.clear_output();
            fig, axes = plt.subplots(2, 2, figsize=(20,10), squeeze=True);
            axes = axes.flatten();
            for i in range(4):
#                 axes[i].plot(time_series_list[i]);
                plot_mv(time_series_list[i], ax=axes[i], highlight_sensor_indices=[text_query_ind])
                axes[i].set_title('Example %d' % i);
            plt.show();

    text = widgets.Text(value='', placeholder='e.g. pump severe leakage', disabled=False)
    search_button = widgets.Button(description='Click Here to Get Examples from MATT',
                                 button_style='primary',
                                 layout=widgets.Layout(width='500px'))
    search_button.on_click(search_cb)
    out_ts_res = widgets.Output()

    search_widgets = widgets.VBox([
        widgets.HBox([widgets.Label('Describe what kind of time series you want to retrieve:'), text]), 
        search_button,
        out_ts_res])

    display(search_widgets)
    

def plot_mv(mv, highlight_sensor_indices=[], title='', ax=None):
    
    sensors = ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 'FS1', 'FS2',
       'TS1', 'TS2', 'TS3', 'TS4', 'VS1', 'CE', 'CP', 'SE']
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10,6));
    for si in range(17):
        if si not in highlight_sensor_indices:
            ax.plot(mv[si], label=sensors[si], alpha=0.15);
    for si in highlight_sensor_indices:
        ax.plot(mv[si], alpha=1, linewidth=2)
    ax.set_title(title);
    ax.set_xlabel('Millisecond');
    ax.set_ylabel('Normalized value');
    ax.legend(ncol=6, loc='upper right');
#     plt.show();
    return ax
    
def display_expl_widgets(model):
    """Explaining time series with natural language"""

    bi = None
    time_series_query = None

    def update_plot(self):
        global time_series_query
        global bi
        bi = buttons.index(self)

        with out_ts:
            out_ts.clear_output()
            out_res.clear_output()

#             fig, ax = plt.subplots(1,1);
#             time_series_query = np.random.random(size=(100,1))
#             time_series_query = pd.read_csv('matt_demo_data/time_series_query_%d.csv' % bi).values.T
            time_series_query = pd.read_csv('matt_demo_hydraulic_data/time_series_query_%d.csv' % bi, index_col=0).values.T
            ax = plot_mv(time_series_query)
#             ax.plot(time_series_query);
#             ax.set_xlabel('Time');
#             ax.set_ylabel('Sensor');
#             ax.set_title('Time series sample %d' % bi);
            plt.show();

    buttons = []
    for i in range(5):
        b = widgets.Button(description='Time series %d' % i)
        buttons.append(b)
        b.on_click(update_plot)

    def retrieve_explanations_cb(_):
#         global time_series_query
#         explanation_list = retrieve_explanations(time_series_query, num=5)
#         explanation_list = model.retrieve_explanations(ts_query, num=5)
        global bi
        explanation_list = list(map(str, np.loadtxt('matt_demo_hydraulic_data/retrieved_explanations_%d.txt' % bi, dtype=str, delimiter='\n')))
        with out_res:
            clear_output()
            for i, e in enumerate(explanation_list):
                print('Explanation %d:\n%s\n' % (i, e))

    expl_button = widgets.Button(description='Click Here to Get Explanations from MATT',
                                 button_style='primary',
                                 layout = widgets.Layout(width='500px'))
    expl_button.on_click(retrieve_explanations_cb)
    out_ts = widgets.Output()
    out_res = widgets.Output()

    expl_widgets = widgets.VBox([widgets.HBox([widgets.Label('Select Query:')] + buttons), 
                          out_ts,
                          expl_button, 
                          out_res])
    
    display(expl_widgets)
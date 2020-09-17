import os
import sys
import numpy as np
import pandas as pd
import functools
from IPython.display import clear_output
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import textwrap

# n_series = 7
# sensors = np.array(['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 'FS1', 'FS2',
#    'TS1', 'TS2', 'TS3', 'TS4', 'VS1', 'CE', 'CP', 'SE'])[[2,6,7,9,13,15,16]]
# sensors = ['Pressure', 'MotorPower', 'VolumeFlow', 'Temperature', 'Vibration', 'CoolingPower', 'Efficiency']
n_series = 17
sensors = np.array(['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 'FS1', 'FS2',
   'TS1', 'TS2', 'TS3', 'TS4', 'VS1', 'CE', 'CP', 'SE'])

def load_data(data_name):
    time_series_query = pd.read_csv('data/time_series_query_0.csv', index_col=0).values.T
    explanation = list(map(str, np.loadtxt('data/retrieved_explanations_tsquery_0.txt', dtype=str, delimiter='\n')))[0]

    fig = plt.figure(tight_layout=True, figsize=(20,5))
    gs = GridSpec(1, 2, width_ratios=[3,1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    plot_mv(time_series_query, ax=ax1, highlight_sensor_indices=range(n_series))
    ax2.text(x=0, y=1, s=textwrap.fill(explanation, width=40), fontsize=20, verticalalignment='top')
    fig.suptitle('Example (time series, text) pair', fontsize=20)

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
            time_series_query = pd.read_csv('data/time_series_query_%d.csv' % bi, index_col=0).values.T
            ax = plot_mv(time_series_query, title='Time series query %d' % bi)
#             ax.plot(time_series_query);
#             ax.set_xlabel('Time');
#             ax.set_ylabel('Sensor');
#             ax.set_title('Time series sample %d' % bi);
            plt.show();

    buttons = []
    for i in range(2):
        b = widgets.Button(description='Time series %d' % i)
        buttons.append(b)
        b.on_click(update_plot)

    def retrieve_explanations_cb(_):
#         global time_series_query
#         explanation_list = retrieve_explanations(time_series_query, num=5)
#         explanation_list = model.retrieve_explanations(ts_query, num=5)
        global bi
        explanation_list = list(map(str, np.loadtxt('data/retrieved_explanations_tsquery_%d.txt' % bi, dtype=str, delimiter='\n')))
        with out_res:
            clear_output()
            for i, e in enumerate(explanation_list):
                print('Relevant explanation %d:\n%s\n' % (i, e))

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
    
cond0 = lambda s: ('cool' in s or 'efficien' in s)
cond1 = lambda s: ('valve' in s or 'lag' in s)

def get_textquery_index(s):
    if cond0(s) and cond1(s):
        return 2
    elif cond0(s):
        return 0
    elif cond1(s):
        return 1
    
def get_highlighted_indices_by_text(s):
    if cond0(s) and cond1(s):
        return [0,1,14,15]
    elif cond0(s):
        return [14,15]
    elif cond1(s):
        return [0,1]
    
    
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
        text_query_ind = get_textquery_index(text_query)
        if text_query_ind is None:
            return
        
        for i in range(5):
            retrieved_time_series = pd.read_csv('data/retrieved_time_series_%d_textquery_%d.csv' % (i, text_query_ind), index_col=0).values.T
            time_series_list.append(retrieved_time_series)

        with out_ts_res:
            out_ts_res.clear_output();
            fig, axes = plt.subplots(2, 2, figsize=(20,10), squeeze=True);
            axes = axes.flatten();
            for i in range(4):
#                 axes[i].plot(time_series_list[i]);
                plot_mv(time_series_list[i], ax=axes[i], 
                        highlight_sensor_indices=get_highlighted_indices_by_text(text_query),
                       title='Retrieved example %d' % i)
#                 plot_mv(time_series_list[i], ax=axes[i])
            plt.tight_layout();
            plt.show();

    text = widgets.Text(value='', placeholder='e.g. valve severe leakage', disabled=False)
    search_button = widgets.Button(description='Click Here to Get Examples from MATT',
                                 button_style='primary',
                                 layout=widgets.Layout(width='500px'))
    search_button.on_click(search_cb)
    out_ts_res = widgets.Output()

    search_widgets = widgets.VBox([
        widgets.VBox([widgets.Label('Describe search criteria (e.g. cooler has low efficiency, valve showing severe leakage, or their combination):'), text]), 
        search_button,
        out_ts_res])

    display(search_widgets)
    

def plot_mv(mv, highlight_sensor_indices=[], title='', ax=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10,5));
        
    if len(highlight_sensor_indices) > 0:
        for si in range(n_series):
            ax.plot(mv[si], 
                    alpha=1. if si in highlight_sensor_indices else 0.4, 
                    linewidth=2. if si in highlight_sensor_indices else 1., 
                    label=sensors[si])
    else:
        for si in range(n_series):
            ax.plot(mv[si], alpha=1, linewidth=1, label=sensors[si])
            
    ax.set_title(title, fontsize=20);
    ax.set_xlabel('Millisecond', fontsize=15);
    ax.set_ylabel('Normalized value', fontsize=15);
    ax.legend(ncol=1, loc='upper right');
    ax.set_ylim([-0.1,1.1]);
#     plt.show();
    return ax
    

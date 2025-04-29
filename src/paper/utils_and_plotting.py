import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ideology_grouping(df, y_col=None, time_col='week', ideology_col='ideology', quantiles=[0.25, 0.5, 0.75], interpolation='linear'):
    ''''''
    df_grouped = df.groupby( [time_col, ideology_col] ).quantile( quantiles )
    df_grouped = df_grouped.T.stack(level=2).T    
    if y_col is None:
        return df_grouped
    else:
        return df_grouped[y_col]

def ideology_lineplot( df, y_col=0.5, y_low=0.25, y_high=0.75, colors=None, **kw ):
    ''''''
    fig, ax = plt.subplots()
    ideologies = df.index.levels[1]
    
    if colors is None:
        # assign random colors to ideologies
        colors = dict( zip( ideologies, random_rgb(len(ideologies)) ) )
    elif type(colors) == list:
        # assign given colors to ideology
        colors = dict( zip( ideologies, colors ) )
    
    for ideology in ideologies: 
        df_ = df.xs(ideology, level=1)
        ax.plot( 
            df_.index.values, 
            df_[y_col].values, 
            color=colors[ideology],
            label=ideology,
            **kw,
        )
        ax.fill_between( 
            df_.index.values, 
            df_[y_low].astype('float').values, 
            df_[y_high].values, 
            color=colors[ideology],
            alpha=0.5,
#             label=ideology,
        ) 
    return ax

def ideology_boxplot( df, metric, x_col='overlap', ideology_col='ideologies', colors=None, bins=10, binning_func=pd.qcut, **kw ):
    ''''''
    fig, ax = plt.subplots()
    df_binned = overlap_binning( df, metric=metric, overlap_col=x_col, bins=bins, binning_func=binning_func )
    ideologies = df[ideology_col].unique()
    # assign random colors to ideologies
    if colors is None:        
        nideologies = len(ideologies)
        colors = random_rgb(nideologies)
    # assign given colors to ideology
    elif type(colors) == list:        
        colors = dict( zip( ideologies, colors ) )

    ax = sns.boxplot(data=df_binned, x=x_col, y=metric,
                hue=ideology_col, palette=colors, **kw)
        
    plt.xticks( rotation=90 )
    return ax

def random_rgb(n=1):
    return [ tuple( np.random.choice(range(256), size=3)/256 ) for i in range(n) ]

def z_score(df): return (df-df.mean())/df.std(ddof=0)
def identity(df): return df

def strike_dates():
    ''''''
    fmt = '%d-%m-%Y'
    dates_FFF = [
        # Global Climate Strike for Future of 15 March 2019
        (pd.to_datetime( '15-03-2019', format=fmt ), pd.to_datetime( '16-03-2019', format=fmt )),
        # Second Global Climate Strike on 24 May 2019
        (pd.to_datetime( '24-05-2019', format=fmt ), pd.to_datetime( '25-05-2019', format=fmt )),
        # International climate strike in Aachen on 21 June 2019
        (pd.to_datetime( '21-06-2019', format=fmt ), pd.to_datetime( '22-06-2019', format=fmt )),
        # International conference in Lausanne on 5–9 August 2019
        (pd.to_datetime( '05-08-2019', format=fmt ), pd.to_datetime( '09-08-2019', format=fmt )), # -12
        # Global Week of Climate Action on 20–27 September 2019
        (pd.to_datetime( '20-09-2019', format=fmt ), pd.to_datetime( '27-09-2019', format=fmt )),
        # 4t Global Climate Strike on 24 May 2019 + COP25
        (pd.to_datetime( '29-11-2019', format=fmt ), pd.to_datetime( '30-11-2019', format=fmt )),
    ]
    dates_XR = [
        # Ocuppy London, 14-19 April
        (pd.to_datetime( '14-04-2019', format=fmt ), pd.to_datetime( '20-04-2019', format=fmt )),
        # Impossible Rebellion, 23 August to 4 September
        (pd.to_datetime( '23-08-2019', format=fmt ), pd.to_datetime( '04-09-2019', format=fmt )),
        # # International Rebellion, 7-19 October
#         (pd.to_datetime( '10-07-2019' ), pd.to_datetime( '10-19-2019' )),
    ]
    return dates_FFF, dates_XR

## COLORS 
def ideology_colors():
    # define base colors
    g = (0.1843137254901957, 0.5843137254901961, 0.02352941176470591) # believers: green    
    r =  (0.7784414118360964, 0.16020424291569396, 0.0646958430658644) # skeptics : red    
    w =  (0.4534507223942208, 0.45048052115583076, 0.44949045407636745) # other : gray
    # define color dictionary with base and interaction colors
    color_dict = {
        'believers': g,
        'skeptics': r,
        'other': w,
        'believers-other': tuple([sum(x)/2 for x in zip(g,w)]),
        'believers-skeptics': tuple([sum(x)/2 for x in zip(g,r)]),
        'other-skeptics': tuple([sum(x)/2 for x in zip(w,r)]),
    }
    return color_dict

def plot_strike_dates(dates=None, colors=None, ax=None, **kw):
    ''''''
    if dates is None:
        dates = strike_dates()
    if colors is None:
        colors = random_rgb( len(dates) )
    
    for (i,dates_) in enumerate(dates):
        if ax is None:
            [plt.axvspan(*date_window, color=colors[i], **kw) for date_window in dates_]
        else:
            [ax.axvspan(*date_window, color=colors[i], **kw) for date_window in dates_]
# helpers 
def overlap_binning( metrics_differences, metric, overlap_col='overlap', bins=10, binning_func=pd.qcut ):
    '''Return a dataframe with the overlaps binned according to `binning_func`.
    For binning_func, try `pd.qcut` (default) or `pd.cut`.
    '''
    # bin overlaps
    q_bins_ = binning_func(metrics_differences[ overlap_col ], bins).values
    # populate dataframe with the overlap values
    metrics_vs_overlap_df = pd.DataFrame()
    metrics_vs_overlap_df.loc[:,'overlap'] = [ np.round(qb.mid,3) for qb in q_bins_]
    # add other relevant columns
    metrics_vs_overlap_df.loc[:,'ideologies'] = metrics_differences['ideologies'].values
    metrics_vs_overlap_df.loc[:,metric] = metrics_differences[metric].values
    return metrics_vs_overlap_df
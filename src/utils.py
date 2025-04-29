import os
import pandas as pd
import numpy as np
import networkx as nx
from datetime import timedelta
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde

# reading
def get_filenames_from_path(path, extension='.csv'):
    files = []
    for file in os.listdir( path ):
        if file.endswith(extension):
            files.append(file)
    return files

def read_all_edgelists(path):
    files = sorted( get_filenames_from_path(path) )
    return [pd.read_csv(path+file) for file in files]

# conversion
def timestamps_from_date( week_nums, start_date='01-Mar-2019', date_format='%d-%m-%Y' ):
    '''Starting from `start_date`, return dates according the the weeks passed according to `week_nums:array[int]`.    
    '''
    t0 = pd.to_datetime( start_date )
    t_array = []
    for week in range(week_nums):        
        dt = timedelta(weeks = week)
        t_array.append( (t0 + dt).strftime( date_format ) )
    return pd.to_datetime(t_array, dayfirst=True)#, format=date_format)

def edgelist_to_network( edgelist, create_using=nx.DiGraph ):
    return nx.from_pandas_edgelist( edgelist, create_using=create_using, edge_attr='weight' )

def network_from_edgelist( edgelist, edge_attr='weight', create_using=nx.DiGraph ):
    return nx.from_pandas_edgelist( edgelist, edge_attr=edge_attr, create_using=create_using )

def network_from_similarity( similarity_matrix, create_using=None ):
    return nx.from_pandas_adjacency( similarity_matrix, create_using=create_using )
    
# def edgelist_from_adjacency( adjacency ): 
#     edgelist = adjacency.copy(deep=True)
#     edgelist.values[ [ np.arange( len(edgelist) ) ]*2] = np.nan # FIX 
#     edgelist = edgelist.stack().reset_index()
#     edgelist.rename( columns={'level_0':'source', 'level_1':'target', 0:'weight'}, inplace=True  )
#     return edgelist

def edgelist_from_adjacency( adjacency ): 
    # Suppose df is your weighted adjacency matrix
    edgelist = adjacency.stack().reset_index()
    edgelist.columns = ['source', 'target', 'weight']
    # Remove zero-weight or missing edges
    edgelist = edgelist[edgelist['weight'] != 0].dropna()
    return edgelist

# random
def shuffle_dict(dictionary, random_state=None):
    '''Shuffle values in a dictionary'''
    shuffled_values = list(dictionary.values()) 
    np.random.shuffle( shuffled_values ) 
    return dict( zip( list(dictionary.keys()), shuffled_values ) )

def flatten_array_of_dicts( array ):
    '''Concatenates all the dicts of array:array[dict].'''
    flattened_array = []
    for dic in array:
        for a in dic.values():
            flattened_array.append(a)
    return np.array(flattened_array)
    
def flatten_array_of_arrays( array ):
    '''Concatenates all the arrays of array:array[array].'''
    return [a for arr in array for a in arr]

def get_empirical_pmf(x, bins=10, **kw ):
    '''Get histogram of x and normalize it to get the empirical pmf.'''
    freqs, bins = np.histogram(x, bins=bins, **kw )
    return freqs/freqs.sum(), bins[:-1]

def get_pdf_cutoff( similarity_vector, bw_method='scott', sort=False, return_kde_dist=False ):
    ''' Get the bimodal cutoff of the similairty distribution using a Gaussian KDE and bandwidth selection defaulting to Scott's rule.
    '''    
    if sort:
        similarity_vector.sort()
    
    KDE = gaussian_kde( similarity_vector, bw_method=bw_method )
    kde_fit = KDE.pdf( similarity_vector ) 
    try:
        bimodal_separation = argrelextrema( kde_fit, np.less, order=5 )[0][0]
    except:
        bimodal_separation = 0
        
    if return_kde_dist:
        return similarity_vector[bimodal_separation], kde_fit
    else:
        return similarity_vector[bimodal_separation]
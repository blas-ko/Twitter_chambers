import numpy as np
import pandas as pd
import os 
from collections import Counter

def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

## NETwORK STUFF
def get_in_degree(edgelist, target='target', weight='weight'):
    return edgelist.groupby( target ).sum().sort_values( weight )

def get_out_degree(edgelist, source='source', weight='weight'):
    return get_in_degree(edgelist, target=source, weight=weight)

def cumulative_degrees( edgelist, target='target', weight='weight' ):
    """ Appends cumulative mass functions columns for population and impact.
    """
    in_degrees = get_in_degree( edgelist, target, weight )
        
    # Getting cumulatives
    in_degrees['cumweight'] = in_degrees['weight'].cumsum()
    in_degrees['cumpop'] = range(1, len(in_degrees)+1)

    # Normalising
    in_degrees['cumweight'] /= in_degrees['cumweight'].iloc[-1]
    in_degrees['cumpop'] /= in_degrees['cumpop'].iloc[-1]
    
    return in_degrees

## POPULAR USERS STUFF 
def cut_population( cumulative_degs, influential_pop_size, return_users=True, pop_column='cumpop', cum_weight_column='cumweight', weight_column='weight'):
    '''If influential_pop_size is int, then cut absolute numbers, if float, cut percentage.
    '''
    
    cumulative_degs_wohi = cumulative_degs[ [weight_column, pop_column, cum_weight_column] ].dropna()
    
    if 0 < influential_pop_size < 1:
        
        total_pop_size = len(cumulative_degs) 
        wohi_pop_size = len(cumulative_degs[cumulative_degs[ cum_weight_column ] < influential_pop_size])
        influential_pop_size_ = total_pop_size - wohi_pop_size
        
        cumulative_degs_wohi = cumulative_degs_wohi.iloc[:-influential_pop_size_]
        cumulative_degs_wohi.loc[:, pop_column] /= cumulative_degs_wohi[pop_column].iloc[-1]
        cumulative_degs_wohi.loc[:, cum_weight_column] /= cumulative_degs_wohi[cum_weight_column].iloc[-1]
        
        if return_users:
            tmp = cumulative_degs[ [weight_column, pop_column, cum_weight_column] ].dropna() 
            users = tmp.iloc[-influential_pop_size_:].index.values
            return cumulative_degs_wohi, users
        else:
            return cumulative_degs_wohi

    elif type(influential_pop_size) == int:
        cumulative_degs_wohi = cumulative_degs_wohi.iloc[:-influential_pop_size]
        cumulative_degs_wohi.loc[:, pop_column] /= cumulative_degs_wohi[pop_column].iloc[-1]
        cumulative_degs_wohi.loc[:, cum_weight_column] /= cumulative_degs_wohi[cum_weight_column].iloc[-1]
        
        if return_users:
            tmp = cumulative_degs[ [weight_column, pop_column, cum_weight_column] ].dropna() 
            users = tmp.iloc[-influential_pop_size:].index.values
            return cumulative_degs_wohi, users
        else:
            return cumulative_degs_wohi

    else:
        if return_users:
            return cumulative_degs, []
        else:
            return cumulative_degs

def process_set_of_edgelists( edgelists, influential_pop_size):
    '''Obtain impact vectors and leading users for the given set of edgelists
    '''
    
    impact_vector         = [] 
    impact_vector_nonleading     = []
    # ginis             = []
    leading_users_vector = []

    for el in edgelists:

        indegs = cumulative_degrees(el)
        indegs_nonleading, leading_users = cut_population( indegs, influential_pop_size, return_users=True )

        impact_vector.append( indegs )
        impact_vector_nonleading.append(  indegs_nonleading )
        leading_users_vector.append( leading_users )

    return impact_vector, impact_vector_nonleading, leading_users_vector

def influential_users_popularity(influential_users):#, num_users=None):
    '''Return a dictionary of all the users in `influential_users:list[list]` as well as their persistence.
    '''
    
    all_influential_users = np.array([])
    for users in influential_users:
        
        all_influential_users = np.append(all_influential_users, np.array(users))
        
    all_influential_users = Counter( all_influential_users )
    all_influential_users = {k: v for k, v in sorted(all_influential_users.items(), key=lambda item: item[1], reverse=True)}
    
#     if num_users is not None:
#         all_influential_users = take( all_influential_users, num_users )
    
    return Counter(all_influential_users)

def influential_users_per_window(influential_users, num_popular_users=20):
    ''' 
        influential users: array of arrays of users
        num_popular_users:int. Minimum persistence to consider users
    '''

    users_popularity = influential_users_popularity(influential_users)

    most_populars = users_popularity.most_common( num_popular_users )
    most_populars = [user for (user, popularity) in most_populars]

    most_populars_per_window = [ set(most_populars).intersection( set(users) ) for users in influential_users ]

    return most_populars_per_window

def get_ginis( cumulative_degs ):
    '''Compute gini coefficient for an array of cumulative degrees.
    '''
    ginis = []
    for cumdeg in cumulative_degs:
        ginis.append( gini_coefficient( cumdeg['weight'].values ) )

    return ginis

## HELPERS 
def get_filenames_from_path(path, extension='.csv'):
    files = []
    for file in os.listdir( path ):
        if file.endswith(extension):
            files.append(file)

    return files

def read_edgelist(fn):
    return pd.read_csv(fn)

def read_edgelists(path):
    files = sorted( get_filenames_from_path(path) )
    return [read_edgelist(path+file) for file in files]

def edgelist_from_adjacency( adjacency ): 
    
    edgelist = adjacency.copy(deep=True)
    edgelist.values[ [ np.arange( len(edgelist) ) ]*2] = np.nan # FIX 
    edgelist = edgelist.stack().reset_index()
    edgelist.rename( columns={'level_0':'source', 'level_1':'target', 0:'weight'}, inplace=True  )
    
    return edgelist
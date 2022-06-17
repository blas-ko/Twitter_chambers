import numpy as np
import pandas as pd
import networkx as nx
import powerlaw
# from community import best_partition
from datetime import timedelta  

## CHAMBER HETEROGENEITY METRICS
def disconnectedness( edgelist, connectivity_type='strongly', weighted=True ):
    '''Ratio of |Cc|/|V|. It's 1 when every node is isolated and 0 when they form a giant component.
    '''
    if type(edgelist) == pd.DataFrame:
        G = edgelist_to_network( edgelist )
    elif type(edgelist) == nx.DiGraph:
        G = [edgelist][0]
    
    if connectivity_type == 'strongly':
        connected_components = nx.strongly_connected_components(G)
    elif connectivity_type == 'weakly':
        connected_components = nx.weakly_connected_components(G)
    else:
        pass
#         Exception

    return len(list(connected_components))/len(G)

def strong_disconnectedness( edgelist, weighted=True ):
    '''Ratio of (|Cc|-1)/|V|. It's 1 when every node is isolated and 0 when they form a giant component.
    '''
    return disconnectedness( edgelist, connectivity_type='strongly' )

def weak_disconnectedness( edgelist, weighted=True ):
    '''Ratio of (|Cc|-1)/|V|. It's 1 when every node is isolated and 0 when they form a giant component.
    '''
    return disconnectedness( edgelist, connectivity_type='weakly' )

def heterogeneity_index( edgelist, weighted=True ):
    '''Extension of Estrada (2010) and Ye (2013) to weighted directed networks. (normalised)
    '''
    
    if type(edgelist) == pd.DataFrame:
        G = edgelist_to_network( edgelist )
    elif type(edgelist) == nx.DiGraph:
        G = [edgelist][0]
    
    if weighted:
        in_degrees = dict(G.in_degree(weight='weight'))
        out_degrees = dict(G.out_degree(weight='weight'))
    else:
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())

    heterogeneity = 0
    for i,j,w in G.edges(data=True):

#         if (in_degrees[i] == 0) | (out_degrees[j] == 0):
#             continue
        
        if weighted:
            w_ij = w['weight']
        else:
            w_ij = 1
        
        heterogeneity += w_ij*( 1/in_degrees[j] + 1/out_degrees[i] - 2/np.sqrt( in_degrees[j]*out_degrees[i] ) )

    heterogeneity /= (len(G))# * 2) #the 2 factor is not in the paper, it should be a correction
    return heterogeneity
    

def voneumann_entropy_weakly( edgelist, weighted=True  ):
    ''' Entropy of the eigenvalues of the normalized Laplacian of the graph, extended for directed, weighted graphs.
    We make a quadratic approximation of the entropy as in Ye (2013).
    '''
    
    if type(edgelist) == pd.DataFrame:
        G = edgelist_to_network( edgelist )
    elif type(edgelist) == nx.DiGraph:
        G = [edgelist][0]
    
    if weighted:
        in_degrees = dict(G.in_degree(weight='weight'))
        out_degrees = dict(G.out_degree(weight='weight'))
    else:
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())

    entropy_vn = 0
    for i,j,w in G.edges(data=True):

#         if (in_degrees[i] == 0) | (out_degrees[j] == 0):
#             continue
        
        if weighted:
            w_ij = w['weight']
        else:
            w_ij = 1
        
#         entropy_vn += w_ij**2*( 1/(out_degrees[i]*in_degrees[j]) + 1/(out_degrees[j]*in_degrees[i]) )
        entropy_vn += w_ij**2/(out_degrees[i]*in_degrees[j])
    
        if (out_degrees[i] != 0) & (out_degrees[j] != 0):
            entropy_vn *= ( in_degrees[i]/out_degrees[i] + in_degrees[j]/out_degrees[j] )

    entropy_vn /= ( 2*len(G)**2 )
    entropy_vn = 1 - 1/len(G) - entropy_vn
    return entropy_vn

def voneumann_entropy_strongly( edgelist, weighted=True ):
    ''' Entropy of the eigenvalues of the normalized Laplacian of the graph, extended for directed, weighted graphs.
    We make a quadratic approximation of the entropy as in Ye (2013).
    '''
    
    if type(edgelist) == pd.DataFrame:
        G = edgelist_to_network( edgelist )
    elif type(edgelist) == nx.DiGraph:
        G = [edgelist][0]
    
    if weighted:
        in_degrees = dict(G.in_degree(weight='weight'))
        out_degrees = dict(G.out_degree(weight='weight'))
    else:
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())

    entropy_vn = 0
    for i,j,w in G.edges(data=True):

#         if (in_degrees[i] == 0) | (out_degrees[j] == 0):
#             continue
        
        if weighted:
            w_ij = w['weight']
        else:
            w_ij = 1
        
        entropy_vn += w_ij**2*( in_degrees[i]/(out_degrees[i]**2*in_degrees[j]) )

    entropy_vn /= ( 2*len(G)**2 )
    entropy_vn = 1 - 1/len(G) - entropy_vn
    return entropy_vn

## TODO: RIGHT NOW IS NOT COMPLETE
def voneumann_entropy_full( edgelist, weighted=True ):
    ''' Entropy of the eigenvalues of the normalized Laplacian of the graph, extended for directed, weighted graphs.
    We make a quadratic approximation of the entropy as in Ye (2013).
    '''

    if type(edgelist) == pd.DataFrame:
        G = edgelist_to_network( edgelist )
    elif type(edgelist) == nx.DiGraph:
        G = [edgelist][0]
    
    if weighted:
        in_degrees = dict(G.in_degree(weight='weight'))
        out_degrees = dict(G.out_degree(weight='weight'))
    else:
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())

    entropy_vn = 0
    for i,j,w in G.edges(data=True):

        
        if weighted:
            w_ij = w['weight']
        else:
            w_ij = 1
        
        entropy_vn += w_ij**2*( in_degrees[i]/(out_degrees[i]**2*in_degrees[j]) )
        
        if (in_degrees[i] == 0) | (out_degrees[j] == 0):
            entropy_vn += w_ij**2/( out_degrees[j] * out_degrees[i] )

    entropy_vn /= ( 2*len(G)**2 )
    entropy_vn = 1 - 1/len(G) - entropy_vn
    return entropy_vn

## DEGREE BASED
def indegree_average( edgelist, weighted=True ):
    '''
    '''
    
    if type(edgelist) == pd.DataFrame:
        G = edgelist_to_network( edgelist )
    elif type(edgelist) == nx.DiGraph:
        G = [edgelist][0]
    
    if weighted:
        in_degrees = dict(G.in_degree(weight='weight'))        
    else:
        in_degrees = dict(G.in_degree())

    return np.mean( list(in_degrees.values()) )

def indegree_variance( edgelist, weighted=True ):
    '''
    '''
    
    if type(edgelist) == pd.DataFrame:
        G = edgelist_to_network( edgelist )
    elif type(edgelist) == nx.DiGraph:
        G = [edgelist][0]
    
    if weighted:
        in_degrees = dict(G.in_degree(weight='weight'))        
    else:
        in_degrees = dict(G.in_degree())
        
    return np.std( list(in_degrees.values()) )**2

def indegree_exponent( edgelist, weighted=True ):
    '''
    '''
    
    if type(edgelist) == pd.DataFrame:
        G = edgelist_to_network( edgelist )
    elif type(edgelist) == nx.DiGraph:
        G = [edgelist][0]
    
    if weighted:
        in_degrees = dict(G.in_degree(weight='weight'))        
    else:
        in_degrees = dict(G.in_degree())

    results = powerlaw.Fit( list(in_degrees.values()), verbose=False )
    return results.alpha

def outdegree_average( edgelist, weighted=True ):
    '''
    '''
    
    if type(edgelist) == pd.DataFrame:
        G = edgelist_to_network( edgelist )
    elif type(edgelist) == nx.DiGraph:
        G = [edgelist][0]
    
    if weighted:
        out_degrees = dict(G.out_degree(weight='weight'))        
    else:
        out_degrees = dict(G.out_degree())

    return np.mean( list(out_degrees.values()) )

def outdegree_variance( edgelist, weighted=True ):
    '''
    '''
    
    if type(edgelist) == pd.DataFrame:
        G = edgelist_to_network( edgelist )
    elif type(edgelist) == nx.DiGraph:
        G = [edgelist][0]
    
    if weighted:
        out_degrees = dict(G.out_degree(weight='weight'))        
    else:
        out_degrees = dict(G.out_degree())
        
    return np.std( list(out_degrees.values()) )**2

def outdegree_exponent( edgelist, weighted=True ):
    '''
    '''
    
    if type(edgelist) == pd.DataFrame:
        G = edgelist_to_network( edgelist )
    elif type(edgelist) == nx.DiGraph:
        G = [edgelist][0]
    
    if weighted:
        out_degrees = dict(G.out_degree(weight='weight'))        
    else:
        out_degrees = dict(G.out_degree())

    results = powerlaw.Fit( list(out_degrees.values()), verbose=False )
    return results.alpha

def global_clustering( edgelist, weighted=True  ):
    
    if type(edgelist) == pd.DataFrame:
        G = edgelist_to_network( edgelist )
    elif type(edgelist) == nx.DiGraph:
        G = [edgelist][0]
        
    return nx.transitivity(G)

def local_clustering( edgelist, weighted=True  ):
    
    if type(edgelist) == pd.DataFrame:
        G = edgelist_to_network( edgelist )
    elif type(edgelist) == nx.DiGraph:
        G = [edgelist][0]
        
    if weighted:
        return np.mean( list( nx.clustering(G, weight='weight').values() ) )
    else:
        return np.mean( list( nx.clustering(G).values() ) )

## DYNAMIC MEASURES 
def network_metric_dynamics( temporal_chamber_edgelists, metric=heterogeneity_index, leading_users=None ):
    
    if leading_users is not None:
        metric_df = pd.DataFrame( columns=leading_users )
    
    for (t, chamber_edgelists_at_t) in enumerate(temporal_chamber_edgelists):
        
        for (user, edgelist) in chamber_edgelists_at_t.items():
            
            metric_df.loc[t, user] = metric( edgelist, weighted=True )
            
    return metric_df.astype('float')

def network_metrics_dynamics(temporal_chamber_edgelists, metrics, leading_users=None):
    dict_of_metrics = dict()
    for metric_name, metric in metrics.items():
        dict_of_metrics[metric_name] = network_metric_dynamics( 
        temporal_chamber_edgelists, 
        metric,
        leading_users)

    return dict_of_metrics

# ToDo: automatise this
# def from_user_to_ideology( metric_df, partition ):

#     believerss_mask = metric_df.columns.isin( [user for (user,stance) in partition.items() if stance== 'believers'] )
#     skepticss_mask = metric_df.columns.isin( [user for (user,stance) in partition.items() if stance== 'skeptics'] )
#     others_mask = metric_df.columns.isin( [user for (user,stance) in partition.items() if stance== 'other'] )

#     metric_by_ideology_df = pd.DataFrame()

#     metric_by_ideology_df['believers'] = metric_df.loc[ :, metric_df.columns[ believerss_mask ]].mean(axis=1)
#     metric_by_ideology_df['skeptics'] = metric_df.loc[ :, metric_df.columns[ skepticss_mask ]].mean(axis=1)
#     metric_by_ideology_df['other'] = metric_df.loc[ :, metric_df.columns[ others_mask ]].mean(axis=1)
    
#     return metric_by_ideology_df

def concatenate_network_measures(dict_of_metrics, ideological_partition=None, times=None):
    ''' Concatenate all network metrics from `dict_of_metrics` into a dataframe where each column is a network metric.
    '''
        
    metric_0 = list(dict_of_metrics.keys())[0]
    network_metrics_df = dict_of_metrics[ metric_0 ].stack().reset_index().rename( {
            'level_0':'week',
            'level_1':'user',
            0:metric_0
        }, axis=1 )

    if times is None:
        times = range( len( dict_of_metrics[metric_0] ) )
    
    for metric in list( dict_of_metrics.keys() )[1:]:
        tmp = dict_of_metrics[metric].stack().reset_index().rename( {
            'level_0':'week',
            'level_1':'user',
            0:metric
        }, axis=1 )

        network_metrics_df[ metric ] = tmp[ metric ]
    
    if ideological_partition is None:
        network_metrics_df['ideology'] = np.nan
    else:
        network_metrics_df['ideology'] = network_metrics_df['user'].map( ideological_partition )
        
    network_metrics_df['week'] = network_metrics_df['week'].map( dict( zip(range(len(times)), times) ) )
    
    return network_metrics_df

def metrics_differences(network_metrics, similarities_dynamic, return_deviation=True):
    ''' From network_metrics:pandas.dataframe and similarities_dynamic:list[pandas.dataframe], compute the differences in all network metrics for 
    the overlap between all pair of users.
    '''
    
    ## USEFUL QUANTITIES ##
    users = network_metrics['user'].unique()
    weeks = network_metrics['week'].unique()
    ideologies_dict = pd.Series(network_metrics['ideology'].values ,index=network_metrics['user']).to_dict()
    metrics = network_metrics.columns[ ~network_metrics.columns.isin(['week','user','ideology']) ]
    
    ## INITIALISE OUTPUT DATAFRAME ## 
    metrics_difference_df = pd.DataFrame()

    counter = 0
    for (t, week) in enumerate(weeks):

        remaining_users = set( network_metrics['user'].unique() )

        for user_i in users:

            remaining_users.remove(user_i)        

            for user_j in remaining_users:

                if (user_i in similarities_dynamic[t].index) and (user_j in similarities_dynamic[t].index):

                    q_ij = similarities_dynamic[t].loc[user_i, user_j]
                    m_i = network_metrics[ (network_metrics['user'] == user_i) & (network_metrics['week'] == week) ][metrics].values[0]
                    m_j = network_metrics[ (network_metrics['user'] == user_j) & (network_metrics['week'] == week) ][metrics].values[0]
                    Δm = np.abs(m_i - m_j)

                    metrics_difference_df.loc[counter, 'overlap'] = q_ij
                    metrics_difference_df.loc[counter, metrics] = Δm

                    metrics_difference_df.loc[counter, 'users'] = '{}_{}'.format(user_i, user_j)

                    # order ideologies alphabetically
                    id_i, id_j = (ideologies_dict[user_i], ideologies_dict[user_j])
                    if id_i < id_j:
                        metrics_difference_df.loc[counter, 'ideologies'] = '{}_{}'.format( ideologies_dict[user_i], ideologies_dict[user_j] )
                    else:
                        metrics_difference_df.loc[counter, 'ideologies'] = '{}_{}'.format( ideologies_dict[user_j], ideologies_dict[user_i] )
                    metrics_difference_df.loc[counter, 'week'] = week

                    counter += 1
    
    if return_deviation:
        metrics_difference_df[metrics] /= network_metrics[metrics].mean()
    
    return metrics_difference_df

## HELPERS
def edgelist_to_network( edgelist, create_using=nx.DiGraph ):
    return nx.from_pandas_edgelist( edgelist, create_using=create_using, edge_attr='weight' )

def network_from_edgelist( edgelist, edge_attr='weight', create_using=nx.DiGraph ):
    return nx.from_pandas_edgelist( edgelist, edge_attr=edge_attr, create_using=create_using )

def network_from_similarity( similarity_matrix, create_using=None ):
    return nx.from_pandas_adjacency( similarity_matrix, create_using=create_using )

def connected_component_size_distribution( graph, weakly=True, normalise=True ):
    
    if weakly:
        connected_components = nx.weakly_connected_components( graph )
    else:
        connected_components = nx.strongly_connected_components( graph )
    
    N = len(graph)
    if normalise:
        return [ len(cc)/N for cc in connected_components ]
    else: 
        return [ len(cc) for cc in connected_components ]
    
def edgelist_from_adjacency( adjacency ): 
    
    edgelist = adjacency.copy(deep=True)
    edgelist.values[ [ np.arange( len(edgelist) ) ]*2] = np.nan # FIX 
    edgelist = edgelist.stack().reset_index()
    edgelist.rename( columns={'level_0':'source', 'level_1':'target', 0:'weight'}, inplace=True  )
    
    return edgelist

def timestamps_from_date( weeks, date='01-Mar-2019', date_format='%d-%m-%Y' ):
    '''Transforms set of week numbers:array[int] into date objects starting from date `date`.    
    '''
    t0 = pd.to_datetime( date )
    t_array = []
    
    for week in range(weeks):
        
        dt = timedelta(weeks = week)
        t_array.append( (t0 + dt).strftime( date_format ) )
        
    return pd.to_datetime(t_array, dayfirst=True)#, format=date_format)
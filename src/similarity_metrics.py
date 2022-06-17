import itertools
import numpy as np
import pandas as pd
# local
import chambers_and_audiences as ca

from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde

###################### OVERLAP SIMILARITY BETWEEN LEADING (PERSISTENT) USERS ######################

def jaccard_similarity(U, V):
    """Return Jaccard similarity between sets U and V.
    """
    
    UintV = len( U.intersection(V) )
    if UintV == 0:
        return np.nan
    else:
        return UintV/(len(U) + len(V) - UintV) # ~len( U.union(V) ) 

def szymkiewicz_simpson_similarity(U, V):
    """Return Szymkiewicz–Simpson similarity between sets U and V.
    """
    
    UintV = len( U.intersection(V) )
    if UintV == 0:
        return np.nan
    else:
        return UintV/(min(len(U),len(V))) # ~len( U.union(V) ) 


def similarity_matrix(chambers, similarity_func=jaccard_similarity ):
    """ Returns the similarity matrix of the chambers:list[set] given a similarity_func:function.
    If no similarity function is specified, it return the Jaccard similarity of the chambers.
    If a similarity function is passed, it should receive two sets as arguments and return a number.
    """
    
    similarities = np.zeros( [len(chambers), len(chambers)] )

    for (i,C_i) in enumerate(chambers.values()):
        for (j,C_j) in enumerate(chambers.values()):
            
            # the matrix is symetric, so no need to run through all i and j.
            if i > j:
                pass
            
            elif i == j:
                similarities[i,j] = 0
            else:
                similarities[i,j] = similarity_func(C_i, C_j)
                similarities[j,i] = similarities[i,j]

    return pd.DataFrame( similarities, index=chambers.keys(), columns=chambers.keys() )

def temporal_similarity_matrices(temporal_chambers, similarity_func=jaccard_similarity, order_by_communities=True, resolution=1, partition=None):
    
    similarity_matrices = []
    for chambers in temporal_chambers:

        Q = similarity_matrix(chambers, similarity_func=similarity_func)
        
        if order_by_communities:
            if partition is None:
                pass
            else:
                partition = {user:comm for (user,comm) in partition.items() if user in Q.columns.values }
                partition = {user: community for (user, community) in sorted( partition.items(), key=lambda item: item[1]) }

            Q = reorder_similarity_matrix(Q, partition)

        similarity_matrices.append( Q )

    return similarity_matrices

def aggregate_similarity_matrices( temporal_similarities ):
    ''' Obtain average similarities between all pairs of users aggregated over all weeks.
    '''

    Q_aggregate = pd.concat( temporal_similarities )
    Q_aggregate = Q_aggregate.groupby(Q_aggregate.index).mean()
    Q_aggregate = Q_aggregate[ Q_aggregate.index ]

    return Q_aggregate


## AUDIENCE VS CHAMBER DEDICATED METHOD
def temporal_subchambers_overlaps(audiences, edgelists, users_excluded=False, removal_ratio='intersection', order_by_communities=True, partition=None, resolution=1, source='source', target='target'):
    """Returns overlap matrices for every week in edgelists based on the subchambers with some members of the audience removed. 
    By default, for every pairs of users i & j, this method removes their common audience members to construct their chambers.
    """
    assert len(audiences) == len(edgelists), "audiences and edgelists are not of the same size"

    # preallocation
    temporal_subchambers_similarities = [] 

    # check if there is a list of excluded users. If there is, check if it's a temporal (leading) or a static (persistent) list 
    list_of_lists = False
    if users_excluded != False:
        list_of_lists = (type(users_excluded[0]) == list) | (type(users_excluded[0]) == set)

    # loop through every week
    for (t, edgelist) in enumerate(edgelists):

        users = audiences[t].keys()
        Q = np.zeros( [ len(users), len(users) ] )
    
        # loop through every pair of users
        for (i,u) in enumerate( users ):
            for (j,v) in enumerate( users ):
                if i < j:

                    Au = audiences[t][u]
                    Av = audiences[t][v]

                    if removal_ratio == 'intersection':
                        #remove intersection
                        Au -= Av 
                        Av -= Au
                    else: # removal ratio is a number in (0,1)
                        #TODO: pick random sample based on ratio
                        pass
                
                    if list_of_lists:
                        Cu = ca.get_chamber_from_audience( u, edgelist, Au, users_excluded=users_excluded[t], source=source, target=target )
                        Cv = ca.get_chamber_from_audience( v, edgelist, Av, users_excluded=users_excluded[t], source=source, target=target )
                    else:
                        Cu = ca.get_chamber_from_audience( u, edgelist, Au, users_excluded=users_excluded, source=source, target=target )
                        Cv = ca.get_chamber_from_audience( v, edgelist, Av, users_excluded=users_excluded, source=source, target=target )

                    Q[i,j] = jaccard_similarity(Cu,Cv)
                    Q[j,i] = Q[i,j]
                    
            
        # transform similarity matrix intro a named dataframe
        Q = pd.DataFrame( Q, index=users, columns=users )
        
        # order entries of similarity matrix by community membership
        if order_by_communities:
            if partition is None:
                pass
            else:
                P = {user:comm for (user,comm) in partition.items() if user in Q.columns.values }
                P = {user: community for (user, community) in sorted( P.items(), key=lambda item: item[1]) }

            Q = reorder_similarity_matrix(Q, P)
        
        temporal_subchambers_similarities.append( Q )

    return temporal_subchambers_similarities


## COMMUNITY RELATED ## 
def reorder_similarity_matrix(similarity_matrix, partition=None):
    """ Reorder users in similarity_matrix:dataframe in terms of their communities in partition:(dict or list). 
    """

    if partition is None:
        return similarity_matrix
    else:
        P = []
        if type(partition) == dict:
            P = list( partition.keys() )
        else:
            P = list( partition )

        return similarity_matrix.loc[ P, P ]

### SIMILARITY METRICS ### 

# peristing users_ per week
def weekly_num_persisting_users( similarities, ideological_partition, times=None ):
    
    df_num_users = []

    if times is None:
        times = range( len(similarities) )
    
    for (t,Q) in enumerate(similarities):

        df_num_users.append( Q.rename(index=ideological_partition).index.value_counts() )

    df_num_users = pd.concat(df_num_users, axis=1)

    df_num_users = df_num_users.T.replace(np.nan, 0)#.set_index(times)

    df_num_users.index = times
    df_num_users.index.name = 'week'

    return df_num_users

# autooverlap

# chamber size

# overlap dynamics 
def weekly_overlaps_by_ideology(similarities, ideological_partition, times=None):
    
    df_weekly_overlaps = []

    if times is None:
        times = range( len(similarities) )

    for (t,Q) in enumerate(similarities):
        
        Q_by_ideology = Q.replace(0, np.nan).rename(index=ideological_partition, columns=ideological_partition).stack().reset_index()
        Q_by_ideology['week'] = times[t] #local variable
        Q_by_ideology['ideology'] = Q_by_ideology['level_0']+'-'+Q_by_ideology['level_1']
#         Q_by_ideology.apply(lambda row: "{}-{}".format(row['level_0'], row['level_1']), axis=1)
        
        df_weekly_overlaps.append(Q_by_ideology[[0,'week','ideology']])

    df_weekly_overlaps = pd.concat( df_weekly_overlaps )
    
    df_weekly_overlaps.replace( {
        "other-believers": "believers-other",
        "other-skeptics": "skeptics-other",
        "skeptics-believers": "believers-skeptics",
    }, inplace=True  )

    df_weekly_overlaps.rename({0:'overlap'}, axis=1, inplace=True)
    return df_weekly_overlaps


### OVERLAP DISTRIBUTIONS ### 
def flatten_similarity_matrix( similarity_matrix ):
    """Obtain sorted list with similarity values from similarity_matrix.
    """
    Q_flattened = similarity_matrix.values[np.triu_indices( len( similarity_matrix ), k=1 )]
    Q_flattened = Q_flattened[ ~np.isnan(Q_flattened) ]
    Q_flattened.sort()
    
    return Q_flattened

def flatten_similarity_matrices( similarity_matrices ):
    """Obtain sorted concatenated list with similarity values from all the matrices in similarity_matrices.
    """
    Q_flattened = ( [ Q.values[ np.triu_indices( len( Q ), k=1)] for Q in similarity_matrices ] )
    Q_flattened = np.array( list( itertools.chain.from_iterable(Q_flattened) ) )
    Q_flattened = Q_flattened[ ~np.isnan(Q_flattened) ]
    Q_flattened.sort()
    
    return Q_flattened

def get_empirical_pmf( similarity_vector, bins=10, **kw ):
    
    freqs, bins = np.histogram( similarity_vector, bins=bins, **kw )
    
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

def edgelist_from_adjacency( adjacency ): 
    
    edgelist = adjacency.copy(deep=True)
    edgelist.values[ [ np.arange( len(edgelist) ) ]*2] = np.nan # FIX 
    edgelist = edgelist.stack().reset_index()
    edgelist.rename( columns={'level_0':'source', 'level_1':'target', 0:'weight'}, inplace=True  )
    
    return edgelist
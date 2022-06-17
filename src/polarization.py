import numpy as np 
from operator import itemgetter

"""General notation: p_in_a : frac of edges inside community alpha. p_in: aggregate frac of in-edges.
p_out: frac of edges goin from one ocmmunity to the other"""
## POLARIZATION FUNCTIONS ## 
def polarisation1( n_11, n_22, n_12, n_21 ):
    '''Returns p_out'''
    connectivity = (n_12 + n_21)/(n_11 + n_22 + n_12 + n_21)
    return 1 - connectivity

def polarisation2( n_11, n_22, n_12, n_21 ):
    '''Returns p_out/p_in'''
    connectivity = (n_12 + n_21)/(n_11 + n_22)
    return 1 - connectivity

def polarisation3( n_11, n_22, n_12, n_21 ):
    '''Returns  p_out( 1/p_in_1 + 1/p_in_2)'''
    connectivity = (n_12/n_11 + n_21/n_22)/2
    return 1 - connectivity

def polarisation4( n_11, n_22, n_12, n_21 ):
    '''Returns 1 - p_out = p_in?'''
    p_11 = n_11/(n_11 + n_12)
    p_12 = n_12/(n_11 + n_12)
    p_21 = n_21/(n_22 + n_21)
    p_22 = n_22/(n_22 + n_21)
    return 1 - (p_12 + p_21)/2

def polarisation5( n_11, n_22, n_12, n_21 ):
    ''' p_in_1*p_in_2 + p_out^2 (EI index?)'''
    p_11 = n_11/(n_11 + n_12)
    p_12 = n_12/(n_11 + n_12)
    p_21 = n_21/(n_22 + n_21)
    p_22 = n_22/(n_22 + n_21)
    return p_11*p_22 - p_12*p_21

def polarisation6( n_11, n_22, n_12, n_21 ):
    '''p_out - p_in (Adaptive EI index)'''
    in_strength = n_11+n_22
    out_strength = n_12 + n_21
    return (in_strength - out_strength)/(in_strength + out_strength)


### MAIN FUNCTIONS ### 
def community_polarisation( edgelist, users_C1, users_C2, polarisation_func=polarisation6, source='source', target='target', weight='weight' ):
    """ Returns the polarization value between communities C1 and C2 for a given `polarization_func`.
    """
    
    mask_sources_in_C1 = edgelist[ source ].isin( users_C1 )
    mask_sources_in_C2 = edgelist[ source ].isin( users_C2 )
    mask_targets_in_C1 = edgelist[ target ].isin( users_C1 )
    mask_targets_in_C2 = edgelist[ target ].isin( users_C2 )
    
    inner_edge_strength_in_C1 = edgelist[ mask_sources_in_C1 & mask_targets_in_C1 ][ weight ].sum()
    inner_edge_strength_in_C2 = edgelist[ mask_sources_in_C2 & mask_targets_in_C2 ][ weight ].sum()
    
    edge_strength_C1_to_C2 = edgelist[ mask_sources_in_C1 & mask_targets_in_C2 ][ weight ].sum()
    edge_strength_C2_to_C1 = edgelist[ mask_sources_in_C2 & mask_targets_in_C1 ][ weight ].sum()
    
    total_strength = inner_edge_strength_in_C1 + inner_edge_strength_in_C2 + edge_strength_C1_to_C2 + edge_strength_C2_to_C1
    
#     connectivity = ( edge_strength_C1_to_C2 + edge_strength_C2_to_C1 )/total_strength
#     polarization = 1 - connectivity
    
    polarization = polarisation_func( 
        inner_edge_strength_in_C1, 
        inner_edge_strength_in_C2, 
        edge_strength_C1_to_C2,
        edge_strength_C2_to_C1
    )
    
    return polarization

def network_polarization( edgelist, partition, polarisation_func=polarisation6, return_polarization_array=False, source='source', target='target', weight='weight' ):
    ''' Given an edgelist (network) and its partition, compute the mean polarization between all pairs of communities.
    '''

    names_of_communities = np.unique(list(partition.values()))
    
    polarization_of_community_pairs = []
    community_pairs = []
    for (i,comm_i) in enumerate(names_of_communities):
        for (j,comm_j) in enumerate(names_of_communities):
            if i < j:                
                
                Ci = get_users_in_community(partition, comm_i)
                Cj = get_users_in_community(partition, comm_j)
                
                Cij = comm_i+'_'+comm_j
                polarization_ij = community_polarisation( edgelist, Ci, Cj, source=source, target=target, weight=weight, polarisation_func=polarisation_func )
                
                community_pairs.append( Cij )
                polarization_of_community_pairs.append( polarization_ij )
                    
    if return_polarization_array:
        return np.mean(polarization_of_community_pairs), dict( zip(community_pairs, polarization_of_community_pairs) )
    else: 
        return np.mean(polarization_of_community_pairs)
    

# not optimal, this should go through partition once rather than once per community
def separate_partition_dict(partition):
    """
    Transform partition into a dict with the users in each community
    """

    community_names = set( partition.values() )
    
    users_per_comm = dict()
    for comm_name in community_names:
        
        users_per_comm[comm_name] = [users for (users, comm) in partition.items() if comm == comm_name] 
        
    return users_per_comm



### RANDOM ENSEMBLES FUNCTIONS 
def shuffle_dict(dictionary, random_state=None):
    '''Shuffle values in a dictionary
    '''
    shuffled_values = list(dictionary.values()) 
    np.random.shuffle( shuffled_values ) 
    return dict( zip( list(dictionary.keys()), shuffled_values ) )

def community_polarisation_shuffled( edgelist, users_C1, users_C2, polarisation_func=polarisation6, source='source', target='target', weight='weight' ):
    '''
    '''
    pass

def community_polarisation_shuffled_ensemble(
    edgelists, 
    partition, 
    C1_name, 
    C2_name, 
    ensemble_size,
    polarisation_func=polarisation6,
    source='source',
    target='target',
    weight='weight'
):
    """
    """
    
    polarization_ensemble = []
    for i in range(ensemble_size):
        
        pass
    
    pass

## HELPERS 
def get_users_in_community( partition, community ):
    return [user for (user, comm) in partition.items() if comm == community]

def get_users_in_partition(partition, *comms):
    """ Retrieve users from partition
    """
    return itemgetter( *comms )( separate_partition_dict(partition) )

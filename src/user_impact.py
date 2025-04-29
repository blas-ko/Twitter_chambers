import numpy as np
from collections import Counter

###################### IMPACT AND LEADING PERSISTENT USERS ######################
def temporal_leading_impacts( edgelists, num_leading_users, target='target', weight='weight'):
    """Return an array with the leading impact vector over time as well as the array of leading users.
    
    Inputs:
        - edgelists: array of edgelists for each time with the number of retweets from user i to user j.
    """ 
    temporal_leading_impact_array = []
    temporal_leading_users_array = []
    for edgelist in edgelists:
        leading_impact_vector = get_leading_impact_vector( edgelist, num_leading_users, target=target, weight=weight )
        temporal_leading_impact_array.append( leading_impact_vector ) 
        temporal_leading_users_array.append( set( leading_impact_vector.index ) )
    return temporal_leading_impact_array, temporal_leading_users_array 

def temporal_persistent_impacts(edgelists, num_leading_users, num_persistent_users, target='target', weight='weight'):
    """ Return impact vector for leading (N) persistent (M) users in edgelist.

    If num_leading_users:float in [0,1], it is treated as a percentage. If num_leading_users:int > 0, it is treated as absolute number of users.

    Inputs:
        - edgelists: array of edgelists for each time with the number of retweets from user i to user j.
        - num_leading_users: number of users considered as leading.
        - num_persistent_users: number of leading users considered as persistent.
    Outputs:
        - persistent_impact_vec: array of impact vectors of the leading persistent users 
        - persistent_users_vec: array of the leading persistent users 
        - users_persistence: persistence of the leading users
    """
    leading_impact_vec, leading_users_vec = temporal_leading_impacts( edgelists, num_leading_users, target=target, weight=weight )
    users_persistence = get_users_persistence( leading_users_vec )
    users_persistence = users_persistence.most_common( num_persistent_users )
    persistent_users = get_users_from_users_persistence_dict( users_persistence ) 

    persistent_impact_vec = []
    persistent_users_vec  = []
    for leading_impact in leading_impact_vec:
        persistent_impacts = leading_impact[ leading_impact.index.isin( persistent_users ) ]
        persistent_impact_vec.append( persistent_impacts  )
        persistent_users_vec.append( set( persistent_impacts.index )  )
    return persistent_impact_vec, persistent_users_vec, users_persistence

def get_impact_vector(edgelist, target='target', weight='weight'):
    """ Returns impact vector for all users in edgelist.
    """
    return edgelist.groupby( target ).sum().sort_values( weight )

# In the article, N = # leading weekly users, M = # of persistent users
def get_leading_impact_vector(edgelist, num_leading_users, target='target', weight='weight'):
    """ Return impact vector for leading (N) persistent (M) users in edgelist.

    If num_leading_users:float in [0,1], it is treated as a percentage. If num_leading_users:int > 0, it is treated as absolute number of users.
    """
    impact_vec = get_impact_vector( edgelist, target=target, weight=weight )

    if 0 < num_leading_users < 1:
        ## based on percentage of retweets
        impact_cum_dist = impact_vec[ weight ].cumsum()
        impact_cum_dist /= impact_cum_dist.iloc[-1]
        return impact_vec[ impact_cum_dist > num_leading_users]
    elif (type(num_leading_users) == int) & (num_leading_users > 0):
        return impact_vec.iloc[ -num_leading_users: ]
    else:
        print('Some error for {}'.format(num_leading_users))

def get_users_persistence(users_array):#, num_users=None):
    '''Return a dictionary of all the users in `influential_users:list[set]` as well as their persistence.
    '''
    # append all users into a single array
    users_persistence = np.array([])
    for users in users_array:        
        users_persistence = np.append(users_persistence, np.array(list(users)))
    
    users_persistence = Counter( users_persistence )
    users_persistence = {k: v for k, v in sorted(users_persistence.items(), key=lambda item: item[1], reverse=True)}
    return Counter(users_persistence)

def get_users_from_users_persistence_dict( users_persistence ):
    return [ users for (users, persistence) in users_persistence ]

# TODO: audiences:array[dict], chambers:array[dict], array of overlap matrices, chambers/audiences networks 
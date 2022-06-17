import numpy as np
from collections import Counter

###################### IMPACT AND LEADING PERSISTENT USERS ######################
def temporal_highimpact_impacts( edgelists, num_highimpact_users, target='target', weight='weight'):
    """Return an array with the leading impact vector over time as well as the array of leading users.
    
    Inputs:
        - edgelists: array of edgelists for each time with the number of retweets from user i to user j.
    """ 
    
    temporal_highimpact_impact_array = []
    temporal_highimpact_users_array = []
    for edgelist in edgelists:

        leading_impact_vector = get_leading_impact_vector( edgelist, num_highimpact_users, target=target, weight=weight )

        temporal_highimpact_impact_array.append( leading_impact_vector ) 
        temporal_highimpact_users_array.append( set( leading_impact_vector.index ) )

    return temporal_highimpact_impact_array, temporal_highimpact_users_array 

def temporal_leading_impacts(edgelists, num_highimpact_users, num_persistent_users, target='target', weight='weight'):
    """ Return impact vector for leading (N) persistent (M) users in edgelist.

    If num_highimpact_users:float in [0,1], it is treated as a percentage. If num_leading_users:int > 0, it is treated as absolute number of users.

    Inputs:
        - edgelists: array of edgelists for each time with the number of retweets from user i to user j.
        - num_highimpact_users: number of high-impact users (N in the paper).
        - num_persistent_users: number of leading users considered as persistent (M in the paper).
    Outputs:
        - persistent_impact_vec: array of impact vectors of the leading persistent users 
        - persistent_users_vec: array of the leading persistent users 
        - users_persistence: persistence of the leading users
    """

    leading_impact_vec, leading_users_vec = temporal_highimpact_impacts( edgelists, num_highimpact_users, target=target, weight=weight )

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

# In the article, N = 50 leading weekly users, M = 50 of persistent users
def get_leading_impact_vector(edgelist, num_leading_users, target='target', weight='weight'):
    """ Return impact vector for leading (N) persistent (M) users in edgelist.
    If num_leading_users:float in [0,1], it is treated as a percentage. If num_leading_users:int > 0, it is treated as absolute number of users.
    """

    impact_vec = get_impact_vector( edgelist, target=target, weight=weight )

    if 0 < num_leading_users < 1:

        ## Option 1: based on percentage of retweets
        impact_cum_dist = impact_vec[ weight ].cumsum()
        impact_cum_dist /= impact_cum_dist.iloc[-1]

        return impact_vec[ impact_cum_dist > num_leading_users]

        ## Option 2: based on percentage of population
        # population_cum_dist = pd.Series(range(1 , len(impact_vec)+1), index=impact_vec.index)    
        # population_cum_dist /= population_cum_dist.iloc[-1]
        
        # return impact_vec[ population_cum_dist > num_leading_users]
        

    elif (type(num_leading_users) == int) & (num_leading_users > 0):

        return impact_vec.iloc[ -num_leading_users: ]

    else:
        print('Some error for {}'.format(num_leading_users))


def get_users_persistence(users_array):#, num_users=None):
    '''Return a dictionary of all the users in `influential_users:list[set]` as well as their persistence.
    '''
    
    users_persistence = np.array([])

    # append all users into a single array
    for users in users_array:
        
        users_persistence = np.append(users_persistence, np.array(list(users)))
    
    users_persistence = Counter( users_persistence )
    users_persistence = {k: v for k, v in sorted(users_persistence.items(), key=lambda item: item[1], reverse=True)}
    
    return Counter(users_persistence)

def get_users_from_users_persistence_dict( users_persistence ):
    return [ users for (users, persistence) in users_persistence ]


# TODO: audiences:array[dict], chambers:array[dict], array of overlap matrices, chambers/audiences networks 

###################### CHAMBERS & AUDIENCES ###################### 

def get_audience( user, edgelist, source='source', target='target' ):
    '''Returns the audience:set of the user given an edgelist.
    '''
    return set( edgelist[ edgelist[ target ] == user ][ source ].unique() )

def get_chamber( user, edgelist, users_excluded=False, source='source', target='target'):
    """Returns the chamber:set of the user given an edgelist.

    users_excluded can be either False or a list of users to exclude from chamber.
    """
    
    audience = get_audience( user, edgelist, source=source, target=target )
    audience_out_ego_network = edgelist[ edgelist[ source ].isin( audience ) ]
    # remove the leading user from the audience ego network
    audience_out_ego_network = audience_out_ego_network[ audience_out_ego_network[ target ] != user ]
    
    # exclude specific users from chamber.
    if users_excluded != False:
        audience_out_ego_network = audience_out_ego_network[ ~audience_out_ego_network[ target ].isin( users_excluded ) ]

    chamber = set( audience_out_ego_network[ target ].unique() )
    return chamber

def get_chamber_from_audience(user, edgelist, audience, users_excluded=False, source='source', target='target'):
    """Returns the chamber:set of the user given an edgelist and a given audience.

    users_excluded can be either False or a list of users to exclude from chamber.
    """

    audience_out_ego_network = edgelist[ edgelist[ source ].isin( audience ) ]
    # remove the leading user from the audience ego network
    audience_out_ego_network = audience_out_ego_network[ audience_out_ego_network[ target ] != user ]
    
    # exclude specific users from chamber.
    if users_excluded != False:
        audience_out_ego_network = audience_out_ego_network[ ~audience_out_ego_network[ target ].isin( users_excluded ) ]

    chamber = set( audience_out_ego_network[ target ].unique() )
    return chamber

def get_audience_edgelist( user, edgelist, source='source', target='target' ):
    '''Builds audience network of leading user `user`.
    ''' 

    audience = get_audience( user, edgelist, source, target )

    audience_edgelist = get_edgelist_from_audience( audience, edgelist, source, target ) 
    return audience_edgelist


def get_chamber_edgelist( user, edgelist, users_excluded=False, source='source', target='target'):
    '''Builds chamber network of leading user `user`.
    ''' 

    chamber = get_chamber( user, edgelist, users_excluded, source, target )

    chamber_edgelist = get_edgelist_from_chamber( chamber, edgelist, source, target ) 
    return chamber_edgelist

# helper
def get_edgelist_from_chamber(chamber, edgelist, source='source', target='target'):
    chamber_edgelist = edgelist[ (edgelist[ source ].isin( chamber )) & (edgelist[ target ].isin( chamber )) ]
    return chamber_edgelist

def get_edgelist_from_audience(audience, edgelist, source='source', target='target'):
    audience_edgelist = edgelist[ (edgelist[ source ].isin( audience )) & (edgelist[ target ].isin( audience )) ]
    return audience_edgelist


def get_chambers_of_users( users, edgelist, users_excluded=False, source='source', target='target', return_network=False ):
    """Get the chamber of all the users in `users`.

    users_excluded can be {False, list:str (list of global excluded users), list:list:str (list of excluded users per week)}
    """

    chambers_dict = {}
    chamber_networks_dict = {}
    for i, user in enumerate(users):

        chamber = get_chamber( user, edgelist, users_excluded=users_excluded, source=source, target=target )
        chambers_dict[user] = chamber

        if return_network:
            chamber_edgelist = get_edgelist_from_chamber( chamber, edgelist, source, target )
            chamber_networks_dict[user] = chamber_edgelist


    if return_network: 
        return chambers_dict, chamber_networks_dict
    else:
        return chambers_dict

def get_audiences_of_users( users, edgelist, source='source', target='target', return_network=False ):
    """Get the audience of all the users in `users`.

    users_excluded can be {False, list:str (list of global excluded users), list:list:str (list of excluded users per week)}
    """

    audiences_dict = {}
    audience_networks_dict = {}
    for i, user in enumerate(users):

        audience = get_audience( user, edgelist, source=source, target=target )
        audiences_dict[user] = audience

        if return_network:
            audience_edgelist = get_edgelist_from_audience( audience, edgelist, source, target )
            audience_networks_dict[user] = audience_edgelist


    if return_network: 
        return audiences_dict, audience_networks_dict
    else:
        return audiences_dict

def temporal_chambers(users, edgelists, users_excluded=False, source='source', target='target', return_networks=False): 
    """Get the chambers of all the users in users for every edgelist in edgelists. It is assumed that edgelists are ordered temporally.
    """

    # preallocation
    temporal_chambers_vec = [{}] * len(edgelists)
    temporal_chamber_networks_vec = [{}] * len(edgelists)

    # check if there is a list of excluded users. If there is, check if it's a temporal (leading) or a static (persistent) list 
    list_of_lists = False
    if users_excluded != False:
        list_of_lists = (type(users_excluded[0]) == list) | (type(users_excluded[0]) == set)

    for (t,edgelist) in enumerate(edgelists): 
        
        if return_networks:
            if list_of_lists:
                chamber_per_user_dict, chamber_network_per_user_dict = get_chambers_of_users( users[t], edgelist, users_excluded[t], source, target, return_networks )
            else:
                chamber_per_user_dict, chamber_network_per_user_dict = get_chambers_of_users( users[t], edgelist, users_excluded, source, target, return_networks )

            # append chamber networks for time t
            temporal_chamber_networks_vec[t] = chamber_network_per_user_dict

        else:
            if list_of_lists:
                chamber_per_user_dict = get_chambers_of_users( users[t], edgelist, users_excluded[t], source, target, return_networks )
            else:
                chamber_per_user_dict = get_chambers_of_users( users[t], edgelist, users_excluded, source, target, return_networks )
        
        # append chambers for time t 
        temporal_chambers_vec[t] = chamber_per_user_dict

    if return_networks:
        return temporal_chambers_vec, temporal_chamber_networks_vec
    else:
        return temporal_chambers_vec

def temporal_audiences( users, edgelists, source='source', target='target', return_networks=False ):
    """Get the audiences of all the users in users for every edgelist in edgelists. It is assumed that edgelists are ordered temporally.
    """

    # preallocation
    temporal_audiences_vec = [{}] * len(edgelists)
    temporal_audience_networks_vec = [{}] * len(edgelists)

    for (t,edgelist) in enumerate(edgelists): 
        
        if return_networks:
            audience_per_user_dict, audience_network_per_user_dict = get_audiences_of_users( users[t], edgelist, source, target, return_networks )
            # append audience networks for time t
            temporal_audience_networks_vec[t] = audience_network_per_user_dict

        else:
            audience_per_user_dict = get_audiences_of_users( users[t], edgelist, source, target, return_networks )
        
        # append audiences for time t 
        temporal_audiences_vec[t] = audience_per_user_dict

    if return_networks:
        return temporal_audiences_vec, temporal_audience_networks_vec
    else:
        return temporal_audiences_vec

# helper
def get_chamber_from_audience(user, edgelist, audience, users_excluded=False, source='source', target='target'):
    """Returns the chamber:set of the user given an edgelist and a given audience.

    users_excluded can be either False or a list of users to exclude from chamber.
    """

    audience_out_ego_network = edgelist[ edgelist[ source ].isin( audience ) ]
    # remove the leading user from the audience ego network
    audience_out_ego_network = audience_out_ego_network[ audience_out_ego_network[ target ] != user ]
    
    # exclude specific users from chamber.
    if users_excluded != False:
        audience_out_ego_network = audience_out_ego_network[ ~audience_out_ego_network[ target ].isin( users_excluded ) ]

    chamber = set( audience_out_ego_network[ target ].unique() )
    return chamber
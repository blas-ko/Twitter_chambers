import pandas as pd

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
    chamber = get_chamber_from_audience(user, edgelist, audience, users_excluded=users_excluded, source=source, target=target)
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
    for user in users:
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

### CHAMBER METRICS 
def weekly_chamber_sizes(chambers, ideological_partition, times=None):
    """Returns the size of each chamber at each week with their corresponding user ideological membership.
    """
    if times is None:
        times = range( len(chambers) )

    df_chamber_sizes = []
    for t,c_t in enumerate(chambers):
        row = [ [
                    times[t], 
                    user, 
                    len(chamber), 
                    ideological_partition[user] 
                ] for (user, chamber) in c_t.items() ]
        
        row = pd.DataFrame( row, columns=['week','user','chamber_size','ideology'] )
        df_chamber_sizes.append( row )

    df_chamber_sizes = pd.concat(df_chamber_sizes, axis=0)  
    return df_chamber_sizes

def weekly_audience_sizes(audiences, ideological_partition, times=None):
    """Exactly the same function as `weekly_chamber_sizes`.
    """
    if times is None:
        times = range( len(audiences) )

    df_audience_sizes = []
    for t,a_t in enumerate(audiences):
        row = [ [
                    times[t], 
                    user, 
                    len(audience), 
                    ideological_partition[user] 
                ] for (user, audience) in a_t.items() ]
        
        row = pd.DataFrame( row, columns=['week','user','audience_size','ideology'] )
        df_audience_sizes.append( row )

    df_audience_sizes = pd.concat(df_audience_sizes, axis=0) 
    return df_audience_sizes

def aggregate_chamber_similarities(temporal_chambers):
    pass
    # '''AGGREGATE CHAMBERS: Chambers of each user aggregated over all weeks'''

    # persistent_leading_users = get_users_from_chambers_list()

    # ### initialise chambers dict
    # Chambers = dict()
    # for user in persistent_leading_users:
    #     Chambers[user] = set()
    
    # for chambers_t in temporal_chambers:
    #     for user in chambers_t.keys():
    #         Chambers[user] = Chambers[user].union( chambers_t[user] )

    # return Chambers 

def aggregate_audience_similarities(edgelists): # TODO
    pass
    ## AGGREGATE AUDIENCES: Chambers of each user aggregated over all weeks
    # Audiences = dict()
    # for user in persistent_leading_users:
    #     Audiences[user] = set()
    #     for (t, el) in enumerate(edgelists):
    #         try:
    #             audience_t = ca.get_audience(user, el)
    #             Audiences[user] = Audiences[user].union( audience_t )
    #         except:
    #             pass

# Helpers 
def get_users_from_chambers_list(temporal_chambers):
    '''Get list of total users in the series of chambers'''
    persistent_leading_users = set()

    for chambers_t in temporal_chambers:
        
        users_t = set( list(chambers_t.keys()) )
        persistent_leading_users = persistent_leading_users.union( users_t )
    return persistent_leading_users

def users_from_chambers(chambers): 
    
    total_users = set()
    for c in chambers:
        
        users_c = set(list( c.keys() ))
        total_users = total_users.union( users_c )        
    return total_users
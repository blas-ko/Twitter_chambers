import pandas as pd
import numpy as np

# local
import similarity_metrics as sm

### CHAMBER METRICS 
def autooverlap( chambers, ideological_partition, order=20, times=None ):
    """Returns the auto-overlap decay for a difference of up to `order=20` weeks. 
    This works for chambers and should also work for audiences 
    """
    
    persistent_leading_users = users_from_chambers(chambers)
    
    chambers_auto_overlap = pd.DataFrame()

    if times is None:
        times = range( len(chambers) )

    counter = 0
    for user in persistent_leading_users:
        for (t, chamber_t) in enumerate(chambers):
            for (τ, chamber_τ) in enumerate(chambers):

                if (t < τ) & (τ -t <= order):
                    if (user in chamber_t) & (user in chamber_τ):

                        q = sm.jaccard_similarity( chamber_τ[ user ], chamber_t[ user ]  )

                        chambers_auto_overlap.loc[counter,'week'] = times[t]
                        chambers_auto_overlap.loc[counter,'week_prime'] = times[τ]
                        chambers_auto_overlap.loc[counter,'overlap'] = q
                        chambers_auto_overlap.loc[counter,'user'] = user
                        chambers_auto_overlap.loc[counter,'ideology'] = ideological_partition[user]

                        counter += 1

    if type(times[0]) == int:
        chambers_auto_overlap.loc[:, 'delta_t'] = np.abs(chambers_auto_overlap['week'] - chambers_auto_overlap['week_prime'])
    else:
        chambers_auto_overlap.loc[:, 'delta_t'] = np.abs( (chambers_auto_overlap['week'] - chambers_auto_overlap['week_prime']).dt.days / 7 ).astype(int)
                 
    return chambers_auto_overlap

def weekly_chamber_sizes(chambers, ideological_partition, times=None):
    """Returns the size of each chamber at each week with their corresponding user ideological membership.
    """
    
    df_chamber_sizes = []

    if times is None:
        times = range( len(chambers) )

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
    
    df_audience_sizes = []

    if times is None:
        times = range( len(audiences) )

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
    '''Obtain the set of all leading users from the chambers dictionary.
    '''
    
    total_users = set()
    for c in chambers:
        
        users_c = set(list( c.keys() ))
        total_users = total_users.union( users_c )
        
    return total_users
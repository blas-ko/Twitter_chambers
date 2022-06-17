def echo_chambers_dynamics(audiences, chambers, partition):
    '''Returns the weekly echo chambers (union of all audiences and chambers) for all the groups in partition:dict.
    '''
    
    assert len(audiences) == len(chambers) 
    
    echo_chambers = [ ]
    for t in range(len(audiences)):

        chamber = chambers[t]
        audience = audiences[t]
        
        echo_chambers.append( get_echo_chambers(audience, chamber, partition) )
        
    return echo_chambers
    
def get_echo_chambers(audience, chamber, partition):
    '''Returns the echo chamber (union of audience and chamber) according to the groups in partition.
    The chamber (audience) is a dict of the form user:chamber(audience)
    '''

    assert chamber.keys() == audience.keys(), "the users in the chambers are not the same that those in the audiences"
    
    ideological_groups = set( partition.values() ) 
    
    # echo_audience = dict()
    echo_chamber = dict()
    for group in ideological_groups:
        # echo_audience[group] = set()
        echo_chamber[group] = set()
    
    for user in chamber.keys():
        
        # echo_audience[ partition[user] ] = echo_audience[ partition[user] ].union( audience[user] )
        echo_chamber[ partition[user] ] = echo_chamber[ partition[user] ].union( audience[user] )
        echo_chamber[ partition[user] ] = echo_chamber[ partition[user] ].union( chamber[user] )
    
    return echo_chamber

### HIGH-IMPACT USER SCORES ### 

## Score Functions
def score_1(audience_i, echo_chamber_β, echo_chamber_α):
    '''Returns the score in [-1,1] of the audience of i. 
    A score of 1 means that i leans towards the ideology of β, -1 to the audience of α, 0 to neither.
    All the inputs are sets of users.
    '''
    
    n_iβ = len( audience_i.intersection( echo_chamber_β ) )
    n_iα = len( audience_i.intersection( echo_chamber_α ) )
    
    if n_iα + n_iβ == 0:
        return 0
    else:
        return (n_iβ - n_iα)/(n_iα + n_iβ)

def score_2(audience_i, echo_chamber_β, echo_chamber_α):
    '''Returns the score in [-1,1] of the audience of i. 
    A score of 1 means that i leans towards the ideology of β, -1 to the audience of α, 0 to neither.
    All the inputs are sets of users.
    '''
    
    n_iβ = len( audience_i.intersection( echo_chamber_β ) )
    n_iα = len( audience_i.intersection( echo_chamber_α ) )
    
    n_α = len( echo_chamber_α )
    n_β = len( echo_chamber_β )
    
    if n_α == 0:
        return n_iβ/n_β
    if n_β == 0:
        return - n_iα/n_α
    if n_α + n_β == 0:
        return 0
    else:
        return n_iβ/n_β - n_iα/n_α

def score_3(audience_i, echo_chamber_β, echo_chamber_α):
    '''Returns the score in [-1,1] of the audience of i. 
    A score of 1 means that i leans towards the ideology of β, -1 to the audience of α, 0 to neither.
    All the inputs are sets of users.
    '''
    
    n_iβ = len( audience_i.intersection( echo_chamber_β ) )
    n_iα = len( audience_i.intersection( echo_chamber_α ) )
    
    n_i = len( audience_i )
    
    return (n_iβ - n_iα)/n_i

def ideology_scores( audiences, echo_chamber, users_excluded=[], score_func=score_1 ):
    '''Compute the ideology scores for all the users in audiences:dict{user:audience} according to score_func:function 
    based on the common audience members in echo_chamber:dict{ideology:users}.
    '''
    
    groups = list( echo_chamber.keys() ) # we assume there are 2 groups
    
    scores = dict()
    for (user, audience) in audiences.items():

        if user not in users_excluded:
            scores[user] = score_func( audience, echo_chamber[ groups[1] ], echo_chamber[ groups[0] ] ) 
            
    return scores

def ideology_scores_dynamics( audiences, echo_chambers, users_excluded=[], score_func=score_1 ):
    '''Compute weekly ideology scores where audiences:list, echo_chambers:list. See ?ideology_scores for details. 
    '''
    
    scores_dynamic = []
    
    for (t, audiences_t) in enumerate( audiences ):
        
        if type( echo_chambers ) == list: # sequence of weekly echo chambers
            scores_dynamic.append( ideology_scores(audiences_t, echo_chambers[t], users_excluded, score_func) )
        elif type( echo_chambers ) == dict: # global echo_echamber
            scores_dynamic.append( ideology_scores(audiences_t, echo_chambers, users_excluded, score_func) )
        else:
            Exception( "Type of echo_chamber not understood. Can be a `list` of echo_chambers or a dictionary." )
        
    return scores_dynamic

### ECHO CHAMBER AUGMENTATION ###
# TODO: include high-impact chambers
def augment_echo_chambers( scores, audiences_of_scored, echo_chambers, thresh=0.75, users_excluded=[]):
    '''Augment `echo_chambers`:list[set] from the `scores`:list[dict] >= `thresh` of the high-impact users using the `audiences_of_scored`:list[dict].
    '''
    
    assert len(scores) == len(audiences_of_scored) # & len(echo_chambers)
    
    augmented_echo_chambers = []
    for t in range(len(scores)):
        augmented_echo_chambers.append( augment_echo_chamber(scores[t], audiences_of_scored[t], echo_chambers[t], thresh=thresh, users_excluded=users_excluded) )
        
    return augmented_echo_chambers
    
def augment_echo_chamber( scores, audiences_of_scored, echo_chamber, thresh=0.75, users_excluded=[]):
    '''Augment `echo_chamber`:set
    '''
    
    groups = list( echo_chamber.keys() )
    augmented_echo_chamber = dict() 
    
    for group in groups:
        # echo_audience[group] = set()
        augmented_echo_chamber[group] = echo_chamber[group]
    
    # augment_echo_chamber = augment_echo_chamber.union( echo_chamber )
    for (user,score) in scores.items():
        if user not in users_excluded:
            if score > thresh:
                augmented_echo_chamber[ groups[1] ] = augmented_echo_chamber[ groups[1] ].union( audiences_of_scored[user] )
            elif score < -thresh:
                augmented_echo_chamber[ groups[0] ] = augmented_echo_chamber[ groups[0] ].union( audiences_of_scored[user] )
            else:
                pass
            
    return augmented_echo_chamber

## Helpers
def flatten_array_of_dicts(x):
    '''Concatenates all the dicts of array:array[dict].'''
    
    flattened_array = []
    for dic in x:
        for a in dic.values():
            flattened_array.append(a)
    
    return flattened_array
    
def flatten_array_of_arrays(x):
    '''Concatenates all the arrays of x:array[array].'''
    return [a for arr in x for a in arr]
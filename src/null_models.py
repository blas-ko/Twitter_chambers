import numpy as np

#TODO: Instead of taking the top 50 users, take those that correspond with the persisting leading users.
# My hypothesis is that this will lower the average overlap, making it more similar to the empirical second peak.

def prob_user_in_chamber(k1,k2,N):
    '''Probability that a user with degree k1 is in the chamber of a user with degree k2 in a network of N nodes.
    '''
    return 1-(1-k1*k2/N**2)**N

def prob_user_in_audience(k1,N):
    '''Probability that a user is in the audience of user with degree k1 in a network of N nodes.
    '''
    return k1/N

def audience_overlap_null(k1,k2,N):
    '''expected overlap between users with degree k1 and k2 respectively in a network of N nodes. 
    '''
    return k1 * k2 / ( N*(k1 + k2) )

# not used
# def prob_user_in_chamber_approx(k1,k2,N):
#     '''Probability that a user with degree k1 is in the chamber of a user with degree k2 in a network of N nodes.
#     '''
#     return k1*k2/N


def chamber_overlap_distribution_null(kseq, leader_indexes=50):
    '''APPROXIMATION of the expected chamber overlap of the M leading users for a given degree sequence.'''

    ix_sort = np.argsort(-kseq)
    kseq = kseq[ix_sort]
    qlist = list()
    num_users = len(kseq)

    if type(leader_indexes) == int:
        leader_indexes = range( min(leader_indexes, num_users) )
    else: # rearrange leader indexes according to sorting of users
        leader_indexes = rearrange_indexes_according_to_array_perm(ix_sort, leader_indexes)

    nonleader_indexes = not_in_y( ix_sort, leader_indexes )
    
    for i in leader_indexes:
        for j in leader_indexes:
            if i<j:

                num = 0
                den = 0
                
                for t in nonleader_indexes: #range(leaders +1, num_users):
                    if not(i == t or j == t):

                        num += prob_user_in_chamber(kseq[t],kseq[i], num_users) * prob_user_in_chamber(kseq[t],kseq[j], num_users)
                        den += prob_user_in_chamber(kseq[t],kseq[i], num_users) + prob_user_in_chamber(kseq[t],kseq[j], num_users)
                
                qlist.append(num/den)

    return qlist

def audience_overlap_distribution_null(kseq, leader_indexes=50):
    '''Expected audience overlap of the M leading users for a given degree sequence.'''

    ix_sort = np.argsort(-kseq)
    kseq = kseq[ix_sort]
    qlist = list()
    num_users = len(kseq)

    # print(leader_indexes)

    if type(leader_indexes) == int:
        # print('number')
        leader_indexes = range( min(leader_indexes, num_users) )
    else: # rearrange leader indexes according to sorting of users
        leader_indexes = rearrange_indexes_according_to_array_perm(ix_sort, leader_indexes)

    # print( len(kseq), kseq[0:5], ix_sort[0:5] )    
    
    for i in leader_indexes:
        for j in leader_indexes:
            if i<j:

                prob = audience_overlap_null(kseq[i], kseq[j], num_users)
                qlist.append( prob )

    return qlist


## HELPERS ## 
def get_degree_sequence(edgelist, target='target', weight='weight'):
    '''Obtain *weigthed* degree sequence (by number of retweets from A to B)
    '''
    return edgelist.groupby( target ).sum().sort_values( weight )[weight].values

def get_binary_degree_sequence(edgelist, source='source', target='target', weight='weight'):
    '''Obtain *unweigthed* degree sequence (by number of users from A to B)
    '''
    return edgelist.groupby( target ).count().sort_values( source )[source].values

def get_binary_degree_sequence_with_zeros(edgelist, source='source', target='target', weight='weight'):
    '''Obtain *unweigthed* degree sequence (by number of users from A to B)
    '''

    deg_seq = edgelist.groupby( target ).count().sort_values( source )[source]
    users_with_retweets = set( deg_seq.index )
    total_users = set( edgelist[source].unique() )
    num_users_without_retweets = len( total_users - users_with_retweets )

    deg_seq = np.append( [0] * num_users_without_retweets, deg_seq.values )

    return deg_seq

def get_binary_degree_sequences_and_leading_indexes(edgelists, IΔ, source='source', target='target', weight='weight'):
    '''Obtain *unweigthed* degree sequences (by number of users from A to B) per week and the indexes of every leading user.
    '''

    deg_seqs = []
    leader_indexes_per_week = []

    for (t, edgelist) in enumerate(edgelists):
        deg_seq = edgelist.groupby( target ).count().sort_values( source )[source].reset_index()
        # get leader indexes; only the ones in IΔ
        leader_indexes = deg_seq[ deg_seq[target].isin( IΔ[t] ) ].index.values

        deg_seqs.append( deg_seq[source].values )
        leader_indexes_per_week.append( leader_indexes )

    return deg_seqs, leader_indexes_per_week

def rearrange_indexes_according_to_array_perm(permution_indexes, original_indexes):
    """Relabels original_indexes so they match to a permutated_array according to `permutation_indexes`. 
    We assume `original_indexes` correspond to a certain selected on an array *before* permutation.
    `permutation_indexes` is the ordering of the permuted array.
    """
    # obs: in1d tests whether each element of a 1-D array is also present in a second array.
    return np.where( np.in1d(permution_indexes, original_indexes))[0]

def not_in_y(x, y):
    ''' Return elements in `x` *not* in the index list `y`
    '''
    
    mask= np.full( len(x),True,dtype=bool)
    mask[y] = False
    
    return x[mask]

### LEGACY 
# def probChamber(k1,k2,N):
# #     probability of a node with degree k1 of being
# #     in the chamber of a node of degree k2
#     return 1-(1-k1*k2/N*2)*N

# def overlaps_noleaders(kseq):
# #     APPROXIMATION of the average value of the overlap for a given degree sequence
#     kseq = -np.sort(-kseq)
#     qlist = list()
#     N = len(kseq)
#     leaders = min(50,N)
#     for i in range(leaders):
#         for j in range(leaders):
#             if i<j:
#                 num = 0
#                 den = 0
#                 for t in range(leaders +1, N):
#                     if not(i == t or j == t):
#                         num += probChamber(kseq[t],kseq[i],N)*probChamber(kseq[t],kseq[j],N)
#                         den += probChamber(kseq[t],kseq[i],N) + probChamber(kseq[t],kseq[j],N)
#                 qlist.append(num/den)
#     return qlist

# def chamber_overlap_distribution_null(kseq, M=50):
#     '''APPROXIMATION of the expected chamber overlap of the M leading users for a given degree sequence.'''
#     kseq = -np.sort(-kseq)
#     qlist = list()
#     num_users = len(kseq)
#     leaders = min(M, num_users)
#     for i in range(leaders):
#         for j in range(leaders):
#             if i<j:
#                 num = 0
#                 den = 0
#                 for t in range(leaders +1, num_users):
#                     if not(i == t or j == t):
#                         num += prob_user_in_chamber(kseq[t],kseq[i], num_users) * prob_user_in_chamber(kseq[t],kseq[j], num_users)
#                         den += prob_user_in_chamber(kseq[t],kseq[i], num_users) + prob_user_in_chamber(kseq[t],kseq[j], num_users)
#                 qlist.append(num/den)
#     return qlist

# def audience_overlap_distribution_null(kseq, M=50):
#     '''Expected audience overlap of the M leading users for a given degree sequence.'''
#     kseq = -np.argsort(-kseq)
#     qlist = list()
#     num_users = len(kseq)
#     leaders = min(M, num_users)
#     for i in range(leaders):
#         for j in range(leaders):
#             if i<j:
#                 prob = audience_overlap_null(kseq[i], kseq[j], num_users)
            
#                 qlist.append( prob )

#     return qlist

import numpy  as np
import pandas as pd
from itertools import combinations
from functools import cache # TODO: needed?

from src.chambers_and_audiences import temporal_chambers
from src.similarity_metrics import temporal_similarity_matrices

####### CM OVERLAP THEORETICAL APPROXIMATION #######
# @cache
def null_model_temporal_similarities_approx(temporal_leaders, temporal_edgelists):
    ''' '''
    Q_dynamic_shuffled = []
    for t,edgelist in enumerate(temporal_edgelists):                
        # Get list of users 
        userlist = set(edgelist['source']).union(edgelist['target'])
        # Get degree sequence
        out_degrees = edgelist['source'].value_counts().reindex(userlist, fill_value=0)
        in_degrees = edgelist['target'].value_counts().reindex(userlist, fill_value=0) # this        
        # Get leaders
        leaders = list(temporal_leaders[t])
        # Get necessary statistics
        # k_out_var = out_degrees.var()
        k_out_sq = (out_degrees**2).mean()
        k_mean = out_degrees.mean()
        m = len(edgelist)
        N = len(userlist)

        out_degree_counts = out_degrees.value_counts()
        out_degree_counts = out_degree_counts[out_degree_counts.index > 0]
        in_degree_counts = in_degrees.value_counts() 

        # TODO: remove when not necessary anymore
        # print(f"{k_mean=:.2f}, {k_out_var=:.2f}, {k_out_sq=:.2f}, {m=:.0f}, {N=:.0f}")
        # print( N, int(in_degree_counts.sum()) )

        Q = pd.DataFrame(index=leaders, columns=leaders)
        for u, v in combinations(leaders, 2):        
            # Compute the expected chamber overlap
            ku, kv = in_degrees[u], in_degrees[v]
            # q_uv = chamber_overlap_approx(ku,kv, in_degrees, k_mean, k_out_sq, users_excluded=None)
            # q_uv = chamber_overlap_approx(ku,kv, in_degrees, k_mean, k_out_sq, users_excluded=None)
            q_uv = chamber_overlap_approx(ku,kv, in_degree_counts, k_mean, out_degree_counts, users_excluded=None)
            # q_uv = chamber_overlap_approx(ku,kv, in_degree_counts, k_mean, out_degree_counts, users_excluded=None)
            Q.loc[u, v] = q_uv
            Q.loc[v, u] = q_uv                    

        Q_dynamic_shuffled.append(Q)
    return Q_dynamic_shuffled

# chamber overlap approximation (Eq 22)
# @cache
def prob_user_in_chamber(k1,k2, k_mean, k_out_counts, N):
    '''Probability that a user with degree k1 is in the chamber of a user with degree k2 in a network of N nodes.
    '''
    K = k1*k2/(N*k_mean)**2
    ks = k_out_counts.index.to_numpy()
    nks = k_out_counts.to_numpy()
    x = 1 - K * ks*(ks-1)    
    valid = x > 0
    log_p = np.sum(nks[valid] * np.log(x[valid]))
    return 1 - np.exp(log_p)

    # p = np.prod((1 - K * ks**2) ** nks)
    # return 1 - np.exp(p)
    # k_out_sq = np.sum( ( (k_out_counts.index ** 2) * k_out_counts.values ) / k_out_counts.sum() )
    # return 1 - np.exp( - k_out_sq*k1*k2/(N*k_mean**2) )
    # return 1 - (1 - (k1*k2)/N**2)**N
    # p = 1
    # for k, nk in k_out_counts.items():
    #     p *= (1 - K*k**2)**nk
    # return 1 - p

# counts = k_out_seq.value_counts()
# ks = counts.index.to_numpy()
# nks = counts.to_numpy()
# p = np.prod((1 - K * ks**2) ** nks)

# @cache
def chamber_overlap_approx(ki, kj, k_in_counts, k_mean, k_out_counts, N=None, users_excluded=None):
    '''Approximation of the expected chamber overlap between users with degrees ki and kj in a network of N nodes.'''
    # ki, kj = k_seq[ui], k_seq[uj]
    if N is None:
        N = int( k_in_counts.sum() )

    # if users_excluded is not None:
    #     k_seq = k_seq.drop( users_excluded )
    #     k_seq_out = k_seq_out.drop( users_excluded )

    # Compute average across all network
    # Note on second quantization: sum terms with equal degree and do way less operations
    num, den = 0, 0
    for (k, nk) in k_in_counts.items():
        pCki = prob_user_in_chamber(k, ki, k_mean, k_out_counts, N)
        pCkj = prob_user_in_chamber(k, kj, k_mean, k_out_counts, N)
        num += nk * pCki * pCkj
        den += nk * (pCki + pCkj - pCki * pCkj)

    q_ij = num / den if den != 0 else 0
    return q_ij

#### APPROX ####
# @cache
# def prob_user_in_chamber(k1,k2, k_mean, k_out_sq, N):
#     '''Probability that a user with degree k1 is in the chamber of a user with degree k2 in a network of N nodes.
#     '''    
#     return 1 - np.exp( - k_out_sq*k1*k2/(N*k_mean**2) )
#     # return 1 - np.exp( - (k_out_sq - k_mean)*k1*k2/(N*k_mean**2) )
#     # return 1 - (1 - k_out_sq*k1*k2/(N*k_mean)**2 )**N
#     # return 1 - (1 - (k1*k2)/N**2)**N

# @cache
# def prob_user_in_chamber(k1,k2,N):
#     '''Probability that a user with degree k1 is in the chamber of a user with degree k2 in a network of N nodes.
#     '''
#     return 1 - (1 - (k1*k2)/N**2)**N

# @cache
# # TODO: Change ui uj to ki kj?
# def chamber_overlap_approx(ki, kj, k_seq, k_mean, k_out_sq, N=None, users_excluded=None):
#     '''Approximation of the expected chamber overlap between users with degrees ki and kj in a network of N nodes.'''
#     # ki, kj = k_seq[ui], k_seq[uj]
#     if N is None:
#         N = len(k_seq)

#     if users_excluded is not None:
#         k_seq = k_seq.drop( users_excluded )

#     # Compute average across all network
#     # Note on second quantization: sum terms with equal degree and do way less operations
#     num, den = 0, 0
#     for (k, nk) in k_seq.value_counts().items():
#         pCki = prob_user_in_chamber(k, ki, k_mean, k_out_sq, N)
#         pCkj = prob_user_in_chamber(k, kj, k_mean, k_out_sq, N)
#         num += nk * pCki * pCkj
#         den += nk * (pCki + pCkj - pCki * pCkj)

#     q_ij = num / den if den != 0 else 0
#     return q_ij
##########

####### CM OVERLAP EMPIRICAL APPROXIMATION #######
def null_model_temporal_similarities_ensemble(temporal_leaders, temporal_edgelists, n_experiments=10):
    ''' '''
    Q_dynamics = null_model_temporal_similarities(temporal_leaders, temporal_edgelists)
    if n_experiments > 1:
        for _ in range(n_experiments-1):
            tmp = null_model_temporal_similarities(temporal_leaders, temporal_edgelists)
            Q_dynamics = [Q_dynamics[t] + tmp[t] for t in range(len(Q_dynamics))]

    Q_dynamics = [Q_dynamics[t] / n_experiments for t in range(len(Q_dynamics))]
    return Q_dynamics

def null_model_temporal_similarities(temporal_leaders, temporal_edgelists):
    ''' '''
    # Sample configuration model for each of the networks
    temporal_edgelists_shuffled = [sample_degseq(el) for el in temporal_edgelists]
    # temporal_edgelists_shuffled = [sample_in_degseq(el) for el in temporal_edgelists]
    # Compute the chambers for the shuffled networks
    chambers_dynamic = temporal_chambers(temporal_leaders, temporal_edgelists_shuffled, users_excluded=temporal_leaders, return_networks=False)
    # Compute the temporal similarity matrices for the shuffled networks
    Q_dynamics = temporal_similarity_matrices(chambers_dynamic)
    return Q_dynamics

# Chamber overlap configuration model
def sample_degseq(edgelist, source='source', target='target', seed=None):
    """Sample an instance of the configuration model according to `edgelist`.
    """
    # Get degree sequence
    np.random.seed(seed)
    out_degrees = edgelist[source].value_counts()
    in_degrees = edgelist[target].value_counts()
    
    # Generate stubs
    out_stubs = np.repeat(out_degrees.index.to_numpy(), out_degrees.values)
    in_stubs = np.repeat(in_degrees.index.to_numpy(), in_degrees.values)

    # Check that the total number of edges match
    assert len(out_stubs) == len(in_stubs), "Degree sums don't match — invalid input."
    # Shuffle the stubs
    np.random.shuffle(out_stubs)
    np.random.shuffle(in_stubs)
    
    # Pair them into a new edgelist
    sampled_edgelist = pd.DataFrame({'source': out_stubs, 'target': in_stubs})
    return sampled_edgelist

def sample_in_degseq(edgelist, source='source', target='target', seed=None):
    """
    Sample a relaxed configuration model where only the in-degree sequence is preserved.
    Source nodes are drawn uniformly at random.

    Parameters:
        edgelist: pd.DataFrame — original edge list
        source: str — column name for source nodes
        target: str — column name for target nodes
        seed: int or None — for reproducibility

    Returns:
        pd.DataFrame — sampled edgelist
    """
    np.random.seed(seed)
    
    # Get in-degree sequence
    in_degrees = edgelist[target].value_counts()
    in_stubs = np.repeat(in_degrees.index.to_numpy(), in_degrees.values)
    
    # Draw random sources for each edge
    unique_sources = edgelist[source].unique()
    out_stubs = np.random.choice(unique_sources, size=len(in_stubs), replace=True)
    
    sampled_edgelist = pd.DataFrame({source: out_stubs, target: in_stubs})
    return sampled_edgelist

######### CM OVERLAP BAD APPROXIMATION ##########
# Chamber overlap bad approximation (Eq 24) -- this is a terrible approximation for the given degree distributions
def chamber_overlap_loose_approx(ki, kj, N, k_var, k_mean):
    '''Approximation of the expected chamber overlap between users with degrees ki and kj in a network of N nodes.'''
    return ki * kj / (N * (ki + kj)) * k_var / k_mean

def null_model_temporal_similarities_loose_approx(temporal_leaders, temporal_edgelists):
    ''' '''
    Q_dynamic_shuffled = []
    for t,edgelist in enumerate(temporal_edgelists):        
        
        # Get list of users 
        userlist = set(edgelist['source']).union(edgelist['target'])
        N = len(userlist)
        # Get degree sequence
        in_degrees = edgelist['target'].value_counts().reindex(userlist, fill_value=0)
            
        # Get leaders
        leaders = list(temporal_leaders[t])

        # Compute degree statistics without leaders
        k_var  = np.var( in_degrees.drop( leaders ) )
        k_mean = np.mean( in_degrees.drop( leaders ) )

        Q = pd.DataFrame(index=leaders, columns=leaders)
        for u, v in combinations(leaders, 2):        
            # Compute the expected chamber overlap
            ku = in_degrees[u]
            kv = in_degrees[v]
            q_uv = chamber_overlap_loose_approx(ku, kv, N, k_var, k_mean)
            Q.loc[u, v] = q_uv
            Q.loc[v, u] = q_uv                    

        Q_dynamic_shuffled.append(Q)
    return Q_dynamic_shuffled

########## EXPERIMENT ##########
def intersections(A,B): return len(A.intersection(B))
def unions(A,B): return len(A.union(B))
def experiment_null_model_temporal_similarities(temporal_leaders, temporal_edgelists):
    ''' '''
    # Sample configuration model for each of the networks
    temporal_edgelists_shuffled = [sample_degseq(el) for el in temporal_edgelists]
    # Compute the chambers for the shuffled networks
    chambers_dynamic = temporal_chambers(temporal_leaders, temporal_edgelists_shuffled, users_excluded=temporal_leaders, return_networks=False)
    # Compute the temporal similarity matrices for the shuffled networks    
    nums = temporal_similarity_matrices(chambers_dynamic, similarity_func=intersections)
    dems = temporal_similarity_matrices(chambers_dynamic, similarity_func=unions)
    return nums, dems
################################
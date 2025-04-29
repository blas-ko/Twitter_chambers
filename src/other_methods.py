import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds
from sklearn.preprocessing import minmax_scale

def estimate_latent_ideologies(edgelist, k=1):
    """
    Estimate latent ideological positions using SVD of standardized residuals.

    Parameters:
    - edgelist: pd.DataFrame with columns ['source', 'target', 'weight'] (optional)
    - k: number of singular vectors to compute (default = 1)

    Returns:
    - ideologies: pd.Series of ideology scores indexed by user
    """

    # Encode nodes as integer indices
    users = pd.Index(edgelist['source'].unique(), name='user')
    elites = pd.Index(edgelist['target'].unique(), name='elite')
    user_idx = {u: i for i, u in enumerate(users)}
    elite_idx = {e: i for i, e in enumerate(elites)}

    # Create sparse adjacency matrix A
    rows = edgelist['source'].map(user_idx)
    cols = edgelist['target'].map(elite_idx)
    weights = edgelist['weight'] if 'weight' in edgelist.columns else 1
    A = csr_matrix((weights, (rows, cols)), shape=(len(users), len(elites)))

    m = A.sum()  # total number of interactions
    P = A / m    # normalized matrix

    # Compute row and column marginals
    r = np.array(P.sum(axis=1)).flatten()  # row sums
    c = np.array(P.sum(axis=0)).flatten()  # column sums
    Dr_inv_sqrt = diags(1.0 / np.sqrt(r + 1e-10))  # avoid division by zero
    Dc_inv_sqrt = diags(1.0 / np.sqrt(c + 1e-10))

    # Compute standardized residual matrix S
    rc_outer = np.outer(r, c)
    R = P - csr_matrix(rc_outer)
    S = Dr_inv_sqrt @ R @ Dc_inv_sqrt

    # Truncated SVD of S (compute k singular values/vectors)
    U, L, VT = svds(S, k=k)
    # Sort components by descending singular values
    idx = np.argsort(-L)
    U = U[:, idx]
    L = L[idx]

    # Compute ideological estimates X
    X = Dr_inv_sqrt @ U[:, 0]  # first singular vector
    # Rescale to [-1, 1]
    ideology = minmax_scale(X, feature_range=(-1, 1))
    return pd.Series(ideology, index=users, name='ideology')

def compare_LI_vs_CS(edgelists, augmented_echo_chambers, echo_chambers):
    """ """
    results = []
    for t in range(len(edgelists)):
        # Define network
        el = edgelists[t]
        ec_aug = augmented_echo_chambers[t]
        ec = echo_chambers[t]
        
        # Preprocess edgelist
        el_pruned = preprocess_edgelist(el)
        # Estimate latent ideologies
        latent_ideologies = estimate_latent_ideologies(el_pruned).sort_values()
        
        # Get our classification within estimated users
        S = latent_ideologies[ latent_ideologies.index.isin( ec_aug['skeptics'] )]
        B = latent_ideologies[ latent_ideologies.index.isin( ec_aug['believers'] )]
        # Get performance of our classification compared to theirs
        Bc = classification_counts(B)
        Sc = classification_counts(S)
        res_performance = performance_metrics(Sc, Bc)
        
        # Get stats on number of users classified
        U = set(el['source']).union(el['target'])
        res_numbers = pd.Series({
            'n_users': len(U),        
            'p_classified_ACS': (len(ec_aug['skeptics'])+len(ec_aug['believers']))/len(U),
            'p_classified_CS': (len(ec['skeptics'])+len(ec['believers']))/len(U),
            'p_classified_LI': len(latent_ideologies)/len(U),
            'p_LI_ACS_intersection': len( (ec_aug['skeptics'].union(ec_aug['believers'])).intersection(latent_ideologies.index) ) / len(latent_ideologies.index),
            'p_LI_CS_intersection': len( (ec['skeptics'].union(ec['believers'])).intersection(latent_ideologies.index) ) / len(latent_ideologies.index),
        })
        
        results.append( pd.concat([res_numbers, res_performance]) )
    # Concatenate resulte
    results = pd.concat(results, axis=1).T
    
    # Remove rows where LI put everyone into the same ideological side (wrong convergence)
    C = ['accuracy','precision','recall','f1']
    faulty_LI = (results[C] == [1.0]*len(C)).any(axis=1)
    results = results[~faulty_LI]

    print(f"Number of faulty weeks for LI: {faulty_LI.sum()}")
    return results

#helpers
def preprocess_edgelist(edgelist, n_influencers = 300, min_consumer_activity = 2):
    """ """
    # Get influencers    
    influencers = edgelist['target'].value_counts().head(n_influencers).index.values    
    # Get interactions towards influencers
    el_pruned = edgelist[ edgelist['target'].isin(influencers) ]    
    # Get active consumers
    consumers = el_pruned['source'].value_counts()
    consumers = consumers[consumers >= min_consumer_activity].index.values    
    # Prune network
    el_pruned = el_pruned[ el_pruned['source'].isin(consumers) ]
    return el_pruned

# performance helpers
def performance_metrics(class_0_counts, class_1_counts):
    TN = class_0_counts.get(True, 0)
    FP = class_0_counts.get(False, 0)
    TP = class_1_counts.get(True, 0)
    FN = class_1_counts.get(False, 0)
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return pd.Series({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

def classification_counts(preds, thresh=0):
    """ """
    # Classify preds
    class_counts = (preds >= thresh).value_counts().reindex([True, False], fill_value=0)
    # Assign True to majority class 
    class_counts.index = class_counts.index[::-1] if class_counts.get(True,0) < class_counts.get(False,0) else class_counts.index
    return class_counts
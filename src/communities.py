
import numpy as np
import pandas as pd

import networkx as nx
from community import best_partition

def communities_spectral( Q, mode=3, cutoff=0, return_spectra=False, anchor_believer=None ):
    '''Unsupervised community detection based on the spectral clustering of the Laplacian of the similarity matrix Q.'''
    L = laplacian_matrix( Q.replace(np.nan,0) )
    eigvals, eigvecs = np.linalg.eig(L)

    # Associate users to their leading eigenvector
    leading_eigvec = eigvecs[:,mode-1]
    users_ordering = np.argsort( leading_eigvec )    
    leading_eigvec = dict( zip(Q.index.values[users_ordering], leading_eigvec[users_ordering]) )
    
    # Force cutoff to have the same sign as the leading eigenvector of some anchor believer
    if anchor_believer is not None:
        anchor_cutoff = next(leading_eigvec[anchor] for anchor in anchor_believer if anchor in leading_eigvec.keys())    
        leading_eigvec = {u: np.sign(anchor_cutoff) * v for (u, v) in leading_eigvec.items()}    

    P_spectral = dict()
    for user, val in leading_eigvec.items():        
        if val < cutoff:
            P_spectral[user] = 'skeptics'
        elif val > cutoff:
            P_spectral[user] = 'believers'
        else:
            P_spectral[user] = 'other'    
    # order communities
    # P_spectral = {user: ideology for user, ideology in sorted(P_spectral.items(), key=lambda item: item[1], reverse=True)}  

    if return_spectra:
        return P_spectral, (eigvals, eigvecs)
    else:
        return P_spectral

def communities_louvain( Q, resolution=1):#, sort_by_community=True ):
    """Obtain communities of the leading users from similarity_matrix:dataframe using louvain best_partition algorithm.
    If sort_by_community:Bool=True, sorts similarity_matrix in place to order it by users in the same community.
    """
    # create wieghted network from similarity matrix
    G = nx.from_pandas_adjacency( Q.replace(np.nan, 0) )
    # obtain communities and sort them
    P = best_partition(G, resolution=resolution)
    P = {user: community for (user, community) in sorted( P.items(), key=lambda item: item[1]) }
    # if sort_by_community:
    #     similarity_matrix = similarity_matrix.loc[P.keys(),P.keys()]
    return P

## Helpers
def laplacian_matrix( Q ):
    '''Returns the names Laplacian matrix of Q.
        input: A:dataframe where index==columns:names of users
    '''
    # degree vector
    D = Q.sum(axis=1)
    D = pd.DataFrame( 
            np.diag(D),
            index=D.index,
            columns=D.index
            )
    return D - Q

def spectrum( L ):
    return np.linalg.eig(L) 
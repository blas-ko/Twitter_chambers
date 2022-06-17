import numpy as np
import pandas as pd

# Example names
COMMUNITY_1 = 'The foos'
COMMUNITY_2 = 'The bars'

def communities_spectral( Q, mode=3, cutoff=0, return_spectra=False ):
    '''Unsupervised community detection based on the spectral clustering of the Laplacian of the similarity matrix Q.'''

    L = laplacian_matrix( Q.replace(np.nan,0) )
    eigvals, eigvecs = np.linalg.eig(L)

    leading_eigvec = eigvecs[:,mode-1]
    users_ordering = np.argsort( leading_eigvec )

    users = Q.index.values[users_ordering]

    P_spectral = dict()
    for i,u in enumerate( leading_eigvec[users_ordering] ):
        
        user = users[i]
        if u > cutoff:
            P_spectral[user] = COMMUNITY_1 
        elif u < cutoff:
            P_spectral[user] = COMMUNITY_2
        else:
            P_spectral[user] = 'other'

    if return_spectra:
        return P_spectral, (eigvals, eigvecs)
    else:
        return P_spectral

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
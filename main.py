#!/usr/bin/env python3
# coding: utf-8

#### LIBRARIES ####
import numpy as np
import pandas as pd
from time import time
# LOCAL
import src.utils as ut
import src.user_impact as ui
import src.chambers_and_audiences as ca
import src.network_metrics as nm
import src.communities as cm
import src.similarity_metrics as sm
import src.polarization as pol
import src.echo_chambers as ec
import src.null_model as null
import src.other_methods as om
from src.communities_manual_labelling import communities_manual_labelling_anonymized

print('Read all libraries.')
tic = time()

## PARAMETERS ##
# PATH_NETWORKS = './data/networks/'
PATH_NETWORKS = './data/networks_anonymized/'
NUM_TOP_USERS = 50 # number of popular users per week
NUM_PERSISTENT_USERS = 50 # number of persistent users
IDEOLOGY_THRESH = 0.5 # threshold for ideology assignment
POLARIZATION_FUNC = pol.polarisation6 # function to compute polarization
IDEOLOGY_SCORE_FUNC = ec.score_1 # function to compute ideology

## WEEKLY EDGELISTS ##
# PATH_NETWORKS = './data/networks_24-08-2021/'
edgelists = ut.read_all_edgelists(PATH_NETWORKS)
network_dates = ut.timestamps_from_date( len(edgelists) , date_format='%d-%m-%Y')

elapsed_time = np.round( (time() - tic) / 60, 2 )
print('Read all edgelists. {} minutes passed'.format(elapsed_time))

# Get activity over time
num_rts_weekly = [el['weight'].sum() for el in edgelists]
num_rts_weekly = pd.Series(num_rts_weekly, index=network_dates)
num_users_weekly = np.array( [ len( set( el.source.unique() ).union( set( el.target.unique() ) ) ) for el in edgelists ] )
num_users_weekly = pd.Series( num_users_weekly, network_dates )

## GET PERSISTENT USERS ## 

# impact, users, users' frequencies
w_IΔ, IΔ, users_persistence = ui.temporal_persistent_impacts(edgelists, NUM_TOP_USERS, NUM_PERSISTENT_USERS)
_, I_leading = ui.temporal_leading_impacts(edgelists, NUM_TOP_USERS)

users_leading_frequencies = ui.get_users_persistence( I_leading )
persistent_leading_users = ui.get_users_from_users_persistence_dict(users_persistence)

leading_voices_dynamics = pd.DataFrame( columns=persistent_leading_users, index=network_dates )

for (t, w_t) in enumerate( w_IΔ ):    
    leading_voices_dynamics.loc[ network_dates[t] ] = w_t['weight']

elapsed_time = np.round( (time() - tic) / 60, 2 )
print('Computed (persistent) leading users. {} minutes passed'.format(elapsed_time))

### CHAMBERS AND AUDIENCES ### 

chambers_dynamic, chambers_edgelists_dynamic = ca.temporal_chambers( IΔ, edgelists, users_excluded=I_leading, return_networks=True )
audiences_dynamic, audiences_edgelists_dynamic = ca.temporal_audiences( IΔ, edgelists, return_networks=True )

## AGGREGATE CHAMBERS: Chambers of each user aggregated over all weeks
chambers_aggregate = dict()
for user in persistent_leading_users:
    
    chambers_aggregate[user] = set()
    for (t, chamber_t) in enumerate(chambers_dynamic):
        try:
            chambers_aggregate[user] = chambers_aggregate[user].union( chamber_t[user] )
        except:
            pass

## AGGREGATE AUDIENCES: Audiences of each user aggregated over all weeks
audiences_aggregate = dict()
for user in persistent_leading_users:
    audiences_aggregate[user] = set()
    for (t, el) in enumerate(edgelists):
        try:
            audience_t = ca.get_audience(user, el)
            audiences_aggregate[user] = audiences_aggregate[user].union( audience_t )
        except:
            pass

elapsed_time = np.round( (time() - tic) / 60, 2 )
print('Computed all chambers, audiences, and their edgelists. {} minutes passed'.format(elapsed_time))

### OVERLAPS ###   

### Chambers
## Weekly overlap matrices
Q_dynamic = sm.temporal_similarity_matrices(chambers_dynamic)
Q_edgelists_dynamic = [ ut.edgelist_from_adjacency ( Q ) for Q in Q_dynamic ]

## Aggregate overlap matrics
Q_static = sm.aggregate_similarity_matrices( Q_dynamic )

### Audiences
## Weekly overlap matrices
Q_audiences_dynamic = sm.temporal_similarity_matrices(audiences_dynamic)
Q_audiences_edgelists_dynamic = [ ut.edgelist_from_adjacency ( Q ) for Q in Q_audiences_dynamic ]

## Aggregate overlap matrics
Q_audiences_static = sm.aggregate_similarity_matrices( Q_audiences_dynamic )

### SUB CHAMBERS (COMPUTED WITHOUT INTERSECTION OF AUDIENCES)
Q_subchambers_dynamic = sm.temporal_subchambers_overlaps(audiences_dynamic, edgelists, users_excluded=I_leading, removal_ratio='intersection')

## Agreggate overlap matrices
Q_subchambers_static = sm.aggregate_similarity_matrices( Q_subchambers_dynamic )

elapsed_time = np.round( (time() - tic) / 60, 2 )
print('Computed overlap similarities. {} minutes passed'.format(elapsed_time))

### COMMUNITIES/IDEOLOGIES ### 

# Manual labelling 
# # TODO: make it independent of persistent_leading_users
# if PATH_NETWORKS.__contains__('anonymized'):
#     P_empirical = communities_manual_labelling_anonymized(persistent_leading_users, M=NUM_PERSISTENT_USERS)
# else:
#     P_empirical = communities_manual_labelling(persistent_leading_users, M=NUM_PERSISTENT_USERS)
P_empirical = communities_manual_labelling_anonymized(M=NUM_PERSISTENT_USERS)
# print( P_empirical )

# Unsupervised
P_spectral, (spectral_eigvals, spectral_eigvecs) = cm.communities_spectral( Q_static.replace(np.nan,0), mode=3, cutoff=0, return_spectra=True, anchor_believer=('PaulEDawson', 2085114))
# P_spectral, (l,u) = cm.communities_spectral(Q_static_binary, mode=3, cutoff=0, return_spectra=True)
# print( P_spectral )

elapsed_time = np.round( (time() - tic) / 60, 2 )
print('Computed supervised and unsupervised communities. {} minutes passed'.format(elapsed_time))

### POLARIZATION ### TODO: automatise ideology names.
# Polarization of similarity matrices
polarization_dynamic_spectral = [ 
    pol.network_polarization( 
        E,
        P_spectral, 
        polarisation_func=POLARIZATION_FUNC
    ) for E in Q_edgelists_dynamic 
]
polarization_dynamic_spectral = pd.Series( polarization_dynamic_spectral, index=network_dates )

polarization_dynamic_empirical = [ 
    pol.network_polarization( 
        E, 
        P_empirical, 
        polarisation_func=POLARIZATION_FUNC, 
        return_polarization_array=True
    )[1]['believers_skeptics'] for E in Q_edgelists_dynamic 
]
polarization_dynamic_empirical = pd.Series( polarization_dynamic_empirical, index=network_dates )

## Polarization with randomized labels
ensemble_size = 50
polarization_dynamic_shuffled = [ 
    [ 
        pol.community_polarisation( 
            E, 
            *pol.get_users_in_partition( ut.shuffle_dict(P_spectral), 'believers', 'skeptics' ), 
            polarisation_func=POLARIZATION_FUNC 
            ) for E in Q_edgelists_dynamic         
    ] for _ in range(ensemble_size) 
]
polarization_dynamic_shuffled_mean = np.mean( polarization_dynamic_shuffled, axis=0 )
polarization_dynamic_shuffled_mean = pd.Series( polarization_dynamic_shuffled_mean, index=network_dates )

elapsed_time = np.round( (time() - tic) / 60, 2 )
print('Computed polarization metrics. {} minutes passed'.format(elapsed_time))

### OVERLAP METRICS ### 
num_persisting_users_dynamic = sm.weekly_num_persisting_users(Q_dynamic, P_spectral, times=network_dates)
overlaps_dynamic = sm.weekly_overlaps_by_ideology(Q_dynamic, P_spectral, times=network_dates)

elapsed_time = np.round( (time() - tic) / 60, 2 )
print('Computed overlap metrics. {} minutes passed'.format(elapsed_time))

### CHAMBER METRICS ### 
# Chambers 
chamber_sizes_dynamic = ca.weekly_chamber_sizes( chambers_dynamic, P_spectral, times=network_dates )
# Audience
audience_sizes_dynamic = ca.weekly_audience_sizes( audiences_dynamic, P_spectral, times=network_dates )

elapsed_time = np.round( (time() - tic) / 60, 2 )
print('Computed chamber metrics. {} minutes passed'.format(elapsed_time))

### CHAMBER NETWORK METRICS ###  
network_metrics = {
    'strong_disconnectedness': nm.strong_disconnectedness,
    # 'weak_disconnectedness': nm.weak_disconnectedness,
    'heterogeneity_index': nm.heterogeneity_index,
    # 'voneumann_entropy_strongly': nm.voneumann_entropy_strongly,
    # 'global_clustering': nm.global_clustering,
    # 'local_clustering': nm.local_clustering,
    'indegree_average': nm.indegree_average,
    # 'indegree_variance': nm.indegree_variance,
    # 'indegree_exponent': nm.indegree_exponent,
    # 'outdegree_average': nm.outdegree_average,
    # 'outdegree_variance': nm.outdegree_variance,
    # 'outdegree_exponent': nm.outdegree_exponent,
}

dict_of_metrics = nm.network_metrics_dynamics( chambers_edgelists_dynamic, network_metrics, persistent_leading_users )
network_metrics_df = nm.concatenate_network_measures(dict_of_metrics, P_spectral, network_dates)

elapsed_time = np.round( (time() - tic) / 60, 2 )
print('Computed chamber network metrics. {} minutes passed'.format(elapsed_time))

# metrics = network_metrics_df.columns[ ~network_metrics_df.columns.isin(['week','user','ideology']) ]
metrics_difference_df = nm.metrics_differences( network_metrics_df, Q_dynamic )

elapsed_time = np.round( (time() - tic) / 60, 2 )
print("Computed difference between network metrics and users' overlap. {} minutes passed".format(elapsed_time))

### ECHO CHAMBERS & AUGMENTED ECHO CHAMBERS ###
# The  echo chamber is the union of audiences and chambers of leading users
echo_chambers = ec.echo_chambers_dynamics( audiences_dynamic, chambers_dynamic, P_spectral )

# Get impact of high-impact users (not leading)
high_impact_audiences = []
for t, high_impacts_t in enumerate( I_leading ):
    high_impact_audiences.append( ca.get_audiences_of_users( high_impacts_t, edgelists[t] ) )

# Get ideology scores of high-impact users
high_impact_scores = ec.ideology_scores_dynamics( high_impact_audiences, echo_chambers, users_excluded=persistent_leading_users, score_func=IDEOLOGY_SCORE_FUNC )
high_impact_scores_flat = - ut.flatten_array_of_dicts( high_impact_scores ) # minus is to set negatives to skeptics

high_impact_scores_believers = high_impact_scores_flat[ high_impact_scores_flat > IDEOLOGY_THRESH ]
high_impact_scores_skeptics = high_impact_scores_flat[ high_impact_scores_flat < -IDEOLOGY_THRESH ]
high_impact_scores_neutral = high_impact_scores_flat[ 
    (high_impact_scores_flat >= -IDEOLOGY_THRESH) & 
    (high_impact_scores_flat <= IDEOLOGY_THRESH) 
]

# Compute augmented echo chambers
augmented_echo_chambers = ec.augment_echo_chambers( 
    high_impact_scores, 
    high_impact_audiences, 
    echo_chambers, 
    thresh=IDEOLOGY_THRESH, 
    users_excluded=persistent_leading_users 
)

## Echo chamber sizes
# augmented
size_augmented_echo_chambers_intersection = [ len( augmented_echo_chambers[t]['skeptics'].intersection( augmented_echo_chambers[t]['believers'] ) ) for t in range(len(augmented_echo_chambers)) ]
size_augmented_echo_chambers_intersection = pd.Series( size_augmented_echo_chambers_intersection, network_dates )

size_augmented_echo_chambers_skeptics = [ len( augmented_echo_chambers[t]['skeptics'] ) for t in range(len(augmented_echo_chambers)) ]
size_augmented_echo_chambers_skeptics = pd.Series( size_augmented_echo_chambers_skeptics, network_dates )

size_augmented_echo_chambers_believers = [ len( augmented_echo_chambers[t]['believers'] ) for t in range(len(augmented_echo_chambers)) ]
size_augmented_echo_chambers_believers = pd.Series( size_augmented_echo_chambers_believers, network_dates )

# normal
size_echo_chambers_intersection = [ len( echo_chambers[t]['skeptics'].intersection( echo_chambers[t]['believers'] ) ) for t in range(len(echo_chambers)) ]
size_echo_chambers_intersection = pd.Series( size_echo_chambers_intersection, network_dates )

size_echo_chambers_skeptics = [ len( echo_chambers[t]['skeptics'] ) for t in range(len(echo_chambers)) ]
size_echo_chambers_skeptics = pd.Series( size_echo_chambers_skeptics, network_dates )

size_echo_chambers_believers = [ len( echo_chambers[t]['believers'] ) for t in range(len(echo_chambers)) ]
size_echo_chambers_believers = pd.Series( size_echo_chambers_believers, network_dates )

elapsed_time = np.round( (time() - tic) / 60, 2 )
print("Computed echo chambers and augmented echo chambers. {} minutes passed".format(elapsed_time))

## Auto-overlap dynamics
autooverlaps_chambers = sm.chamber_autooverlap( chambers_dynamic, P_spectral, times=network_dates )
autooverlaps_echo_chambers = sm.echo_chambers_autooverlap(echo_chambers, order=20, times=network_dates)
autooverlaps_aug_echo_chambers = sm.echo_chambers_autooverlap(augmented_echo_chambers, order=20, times=network_dates)

elapsed_time = np.round( (time() - tic) / 60, 2 )
print("Computed auto-overlap metrics. {} minutes passed".format(elapsed_time))

## NULL MODELS ##
# Compute overlaps averaged over 10 realizations of the configuration model
n_experiments = 10
Q_config_model = null.null_model_temporal_similarities_ensemble(IΔ, edgelists, n_experiments=n_experiments)

elapsed_time = np.round( (time() - tic) / 60, 2 )
print("Computed null model overlaps. {} minutes passed".format(elapsed_time))

## METHODS COMPARISON##
# Compare our echo chamber classification against latent ideologies (Falkenberg, 2022) across weeks
performance_against_LI = om.compare_LI_vs_CS(edgelists, augmented_echo_chambers, echo_chambers)

elapsed_time = np.round( (time() - tic) / 60, 2 )
print("Computed latent ideologies and model performance against ours. {} minutes passed".format(elapsed_time))

elapsed_time = np.round( (time() - tic) / 60, 2 )
print('\nRAN EVERYTHING SUCCESSFULLY!!! It took a total of {} minutes.'.format( elapsed_time ))
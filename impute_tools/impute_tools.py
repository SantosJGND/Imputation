from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import estimate_bandwidth

import numpy as np
import itertools as it

def rand_wdDist(genotype, nan_coords,
               wind_sizes= 100,
                Nreps= 400,
                ncomps= 5,
                dimN= 2,
                nan_idx= 0,
                metric= 'euclidean'):
    '''
    given gentoype array, a sngle coordinate of which the column to avoid, 
    analyse Nrep windows of wind_sizes number of contiguous features. 
    calculate distances to observation provided in nan_coords in PCA feature space. 
    Number of dimensions for Dim reduction and distance calculated is kept separate. for reasons. 
    '''
    nan_obs= nan_coords[nan_idx]
    nan_acc= nan_obs[0]
    nan_pos= nan_obs[1]

    other_obs= [x for x in range(genotype.shape[0]) if x != nan_acc]

    dist_store= []

    for idx in range(Nreps):

        st= -1
        while st == -1:
            stp= np.random.randint(0,genotype.shape[1]-wind_sizes)
            dinc= nan_pos - stp
            if dinc > wind_sizes or dinc < 0:
                st= stp

        nwind= genotype[:,stp:(stp+wind_sizes)]

        pca2 = PCA(n_components=ncomps, whiten=False,svd_solver='randomized')
        featw= pca2.fit_transform(nwind)

        obsn= featw[nan_acc,:dimN].reshape(1,-1)
        dist_vec= pairwise_distances(obsn, featw[other_obs,:dimN],
                                                    metric=metric)

        dist_store.extend(dist_vec)
    
    dist_store= np.array(dist_store)
    
    return dist_store


def get_bg_grid(Quanted_set, P= 20, dimN= 2):
    '''
    return coordinates for grid encompassing coordinates in Quanted data set.
    - P: grid number,
    - dimN: number of features from Quanted_set to use. 
    '''
    to_mesh= [np.linspace(min(Quanted_set[:,x]),max(Quanted_set[:,x]),P) for x in range(dimN)]

    coords_net = np.meshgrid(*to_mesh, indexing= 'ij')

    pprod= [list(range(P)) for y in range(dimN)]
    traces= [x for x in it.product(*pprod)]

    background= np.array(coords_net)

    background= background.reshape(dimN,np.prod(background.shape[1:])).T
    
    return background



def kde_likes_extract(dist_grid,dist_ref,pca_obj= 0, 
                     Bandwidth_split = 30,dist_comps= 4):
    """
    i) Dr fit using the first data set, ii) transformation of both, iii) likelihood extraction for first using KDE.
    """

    if not pca_obj:
        pca2 = PCA(n_components=dist_comps, whiten=False,svd_solver='randomized')
        pca_obj= pca2.fit(dist_ref)
    #pca_dists= pca2.fit(dist_grid)
    featw= pca_obj.transform(dist_grid)
    featref= pca_obj.transform(dist_ref)

    params = {'bandwidth': np.linspace(np.min(featref), np.max(featref),Bandwidth_split)}
    grid = GridSearchCV(KernelDensity(algorithm = "ball_tree",breadth_first = False), params,verbose=0,cv= 3,iid= False)

    grid.fit(featref)
    kde = grid.best_estimator_

    grid_likes= kde.score_samples(featw)
    grid_likes= np.exp(grid_likes)
    
    return grid_likes

####
#### GRID search
####


def nBg_MS(subset, lb= 0.05,up= 0.8,kernel= 'gaussian',N_samps= 50):
    band_qtl = [estimate_bandwidth(subset, quantile=0.05),estimate_bandwidth(subset, quantile=0.8)]
    params = {'bandwidth': np.linspace(*band_qtl, 20)}
    grid = GridSearchCV(KernelDensity(kernel= kernel), params, cv=5, iid=False)
    grid.fit(subset)

    kde = grid.best_estimator_

    background= kde.sample(N_samps, random_state=0)
    
    return background

def nBg_grid(subset,P= 20, dimN= 2):
    
    background= get_bg_grid(subset, P= P, dimN= dimN)
    
    return background


def gridWalk(featl,dist_ref, BG_func, BG_args= {}, std_gp_use= 0,
            P= 20,
            dimN= 2,
            N_samps= 50,
            dist_comps= 10,
            Bandwidth_split = 30,
            metric= 'euclidean',
            kernel= 'gaussian',
            min_samp= 5):
    '''
    grid narrowing using MeanShift.
    '''
    
    ###
    Quanted_set= np.array(featl)
    
    if std_gp_use== 0:
        std_gp_use= list(range(Quanted_set.shape[0]))
    ##
    background= get_bg_grid(Quanted_set, P= P, dimN= dimN)
    ###
    workfeat= Quanted_set[std_gp_use,:dimN]

    granted= []
    d= 0

    max_like= [0]

    while d == 0:
        ###
        dist_grid= pairwise_distances(background, workfeat,
                                                    metric=metric)

        ####

        grid_likes= kde_likes_extract(dist_grid,dist_ref,dist_comps= dist_comps)

        ####

        lm= np.mean(grid_likes)
        lsd= np.std(grid_likes)
        lmax= np.max(grid_likes)

        diff_max= max_like[-1]

        if lm + lsd * 2 > diff_max:
            which= [x for x in range(len(grid_likes)) if grid_likes[x] >= lm]
            
            if len(which) < min_samp:
                granted.extend(background)
                max_like+= [lmax]
                d += 1

            else:
                granted.extend(background)
                
                subset= background[which]
                #
                background= BG_func(subset,**BG_args)
                #
                max_like+= [lmax]

        else:
            granted.extend(background)
            max_like+= [lmax]
            d+= 1

    granted= np.array(granted)
    dist_grid= pairwise_distances(granted, workfeat,
                                                metric=metric)

    ####
    grid_likes= kde_likes_extract(dist_grid,dist_ref,dist_comps= dist_comps)
    
    return granted, grid_likes

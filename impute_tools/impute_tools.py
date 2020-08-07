from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import pairwise_distances
from scipy.stats import norm

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
                     Bandwidth_split = 30,dist_comps= 4, lnorm= False):
    """
    i) Dr fit using the first data set, ii) transformation of both, iii) likelihood extraction for first using KDE.
    """

    if not pca_obj:
        pca2 = PCA(n_components=dist_comps, whiten=False,svd_solver='randomized')
        pca_obj= pca2.fit(dist_ref)
    #pca_dists= pca2.fit(dist_grid)
    featw= pca_obj.transform(dist_grid)
    featref= pca_obj.transform(dist_ref)

    params = {'bandwidth': np.linspace(np.min(dist_ref), np.max(dist_ref),Bandwidth_split)}
    grid = GridSearchCV(KernelDensity(algorithm = "ball_tree",breadth_first = False), params,verbose=0,cv= 3,iid= False)

    grid.fit(featref)
    kde = grid.best_estimator_

    grid_likes= kde.score_samples(featw)

    if lnorm: 
        lrefs= kde.score_samples(featref)
        grid_likes= norm.cdf(grid_likes,loc= np.mean(lrefs),scale= np.std(lrefs))

    else:
        grid_likes= np.exp(grid_likes)
    
    return grid_likes

###
###


def store_differences(genotype, Seq_store, select_same, dr_obj, 
                        dimN= 2, wind_sizes= 50, tf= [0,1],
                        nan_char= [1,9],
                        metric= 'euclidean',
                        process_tools= {},
                        keep_tools= {},
                        std= True,
                        avoid_range= 0):
    """
    Use vector of array positions to extract windows;
    Calculate distances between between target observation and select_same distances at each window.
    exclude observations at each windows
    """
    nan_acc= tf[0]
    nan_pos= tf[1]
    
    dist_store= []

    mask_pos= list(range(nan_pos-avoid_range,nan_pos+avoid_range+1))

    for stp in Seq_store:
        
        if stp < wind_sizes/2:
            continue
        
        nwind= lwind_extractv2(genotype,idx= stp, wind_sizes= wind_sizes,mask_pos= mask_pos)
        
        ## local keep 
        keep= keep_tools[0](nwind, **keep_tools[1])
        #
        
        ### process local window
        nwind= process_tools[0](nwind,**process_tools[1])
        
        if sum(keep) <= 5:
            continue
        
        if not keep[nan_acc]:
            continue
        #
        
        pcah= dr_obj.fit(nwind[keep])
        featw= pcah.transform(nwind)

        obsn= featw[nan_acc,:dimN].reshape(1,-1)
        dist_vec= pairwise_distances(obsn, featw[select_same,:dimN],
                                                    metric=metric)

        if std:
            dist_vec= (dist_vec.T - np.mean(dist_vec,axis= 1)) / np.std(dist_vec,axis= 1)
            dist_vec= dist_vec.T

        dist_store.extend(dist_vec)
    
    dist_store= np.array(dist_store)
    
    return dist_store





def grid_likelihood(dist_grid,dist_store,dist_tools,labelf_select= {},std_gp_use= [],
                   correct_dist= {}, ncomps= 4):
    '''
    classify distance vectors in dist_grid using reference dlik_func. Dlik_func proocesses dist_grid observations.
    refereence observations are extracted from the dist_store array and can be subsetted.
    optionally, use std_gp_use to use only a subset of features.
    - likelihoods are weighed for the number of observations in each reference group. 
    - likelihoods are corrected for the inverse of their distance.
    '''
    
    ###
    if not std_gp_use:
        std_gp_use= list(range(dist_store.shape[1]))
    
    if not labelf_select:
        labelf_select= {0: list(range(dist_store.shape[0]))}
    
    if not correct_dist:
        correct_dist= {z: 1 for z in labelf_select}
    ###
    
    pca_dists = PCA(n_components=ncomps, whiten=False,svd_solver='randomized')

    pca_dists= pca_dists.fit(dist_store[:,std_gp_use])
    
    likes_array= []
    supp_prop= []
    dist_prop= []
    
    ##    
    for dist_ref_select,g in labelf_select.items():
        if dist_ref_select == -1:
            continue
        
        if len(g) > 5:
            dist_ref= dist_store[g,:]
            dist_ref= dist_ref[:,std_gp_use]
            
            grid_likes= dist_tools[0](dist_grid,dist_ref,pca_obj= pca_dists,**dist_tools[1])
            
            ###
            likes_array.append(grid_likes)
            supp_prop.append(len(g))
            
            dist_prop.append(min(correct_dist[dist_ref_select]))
    
    
    supp_prop= np.array(supp_prop) / sum(supp_prop)
    dist_prop= 1 / np.array(dist_prop)**2
    dist_prop= dist_prop / sum(dist_prop)

    likes_array= np.array(likes_array)
    
    likes_array= likes_array * supp_prop.reshape(-1,1)
    likes_array= likes_array * dist_prop.reshape(-1,1)
    #likes_array= likes_array / np.nansum(likes_array,axis= 1).reshape(-1,1)
    #
    
    like_diet= np.nansum(likes_array,axis= 0) 
    #
    return like_diet



 ###
 ###




#######################################################
#######################################################

from impute_tools.genome_adapt import (
    target_wdDist, lwind_extractv2
)

def get_likes_engine(genotype, wst, tf, 
                     process_tools, keep_tools, varFilt_tools, dist_tools,
                     wind_sizes= 50, Nreps= 100, ncomps= 3, nan_char= [1,9], ind_min= 50,
                     dimN= 3, metric= 'euclidean', comps_dists= 5,
                     std_diffs= True,avoid_range= 0):
    '''
    '''
    tf_acc= tf[0]
    tf_pos= tf[1]
    dr_obj = PCA(n_components=ncomps, whiten=False,svd_solver='randomized')
    
    ### get array of similar windows
    select_same, Seq_store= target_wdDist(genotype, keep_tools, avail_coords= wst,
                      nan_obs= tf,
                       wind_sizes= wind_sizes,
                        Nrep= Nreps,
                        ncomps= ncomps,
                         ind_min= ind_min,
                         avoid_range= avoid_range)


    ### get distances array from extracted windows windows
    dist_store= store_differences(genotype, Seq_store, select_same, dr_obj, 
                            dimN= dimN, wind_sizes= wind_sizes, tf= tf,
                            nan_char= nan_char,
                            metric= metric,
                            process_tools= process_tools,
                            keep_tools= keep_tools,
                            std= std_diffs,
                            avoid_range= avoid_range)


    ### Variance in distances across reference windows
    dist_var= np.std(dist_store,axis= 0)**2

    # subselect features by variance in distances:
    std_gp_use= varFilt_tools[0](dist_var,**varFilt_tools[1])

    ### Clustering windows
    ###
    pca_cl = PCA(n_components=comps_dists, whiten=False,svd_solver='randomized')
    featd= pca_cl.fit_transform(dist_store[:,std_gp_use])
    bandwidth = estimate_bandwidth(featd, quantile=0.2)

    if bandwidth == 0:
        labelf_select= {0: list(range(featd.shape[0]))}
    else:
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=False, min_bin_freq=15)
        ms.fit(featd)
        labelsf = ms.labels_
        labelf_select = {y:[x for x in range(len(labelsf)) if labelsf[x] == y] for y in sorted(list(set(labelsf))) if y != -1}



    ### 
    ### Distances to focal position by cluster, in number of inds.
    correct_dist= {
        z: [abs(Seq_store[x] - tf_pos) for x in g] for z,g in labelf_select.items()
    }

    ### 
    
    return dist_store, labelf_select, correct_dist, select_same, std_gp_use


from impute_tools.impute_cofactors import (
    index_convert2L, expand_grid
)



def window_exam(featl, samp_keep, select_same, std_gp_use, dist_store, dist_tools, 
                labelf_select= {},correct_dist= {},
               P= 25, dimN= 3, metric= "euclidean",expand= 1,
                std_diffs= True):
    '''
    Local window, parse for select_same inds from target_wdDist
    retain only the observations without avoid_data at this window.
    '''
    #### subselect samples to use at window given missingness
    sample_select, std_gp_use= index_convert2L(samp_keep, select_same, std_gp_use)
    
    #### define grid to use
    Quanted_set= np.array(featl)
    background= get_bg_grid(Quanted_set , P= P, dimN= dimN)
    background= expand_grid(background, expand= expand)


    ###
    ### DISTS for this grid
    workfeat= featl[sample_select,:dimN]
    dist_grid= pairwise_distances(background, workfeat,
                                                metric=metric)

    ###
    if std_diffs:

        dist_grid= (dist_grid.T - np.mean(dist_grid,axis= 1)) / np.std(dist_grid,axis= 1)
        dist_grid= dist_grid.T

    like_diet= grid_likelihood(dist_grid,dist_store,dist_tools,
                               labelf_select= labelf_select,
                               std_gp_use= std_gp_use,
                              correct_dist= correct_dist)
    
    return background, like_diet



###############################################################################
###############################################################################
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




#######
#######


def haplotype_impute(genotype,tf,
                     process_tools, keep_tools, varFilt_tools, dist_tools, 
                     comps_dists= 3,
                     wind_sizes= 20, Nreps= 50, ncomps= 3, nan_char= [9], ind_min= 50,
                     dimN= 3, metric= 'euclidean',std_diffs= False, predict= 'haplotype'):
    '''
    impute function 
    '''

    tf_pos= tf[1]
    tf_acc= tf[0]

    if predict== 'haplotype':
        avoid_range= int(wind_sizes / 2) - 1
    else:
        avoid_range= predict

    ###
    ###
    wst= wparse_func(summary,centre= tf_pos,**wparse_args)
    print('# pos: {}'.format(len(wst)))

    ###
    ### construct dist ref

    dist_store, labelf_select, correct_dist, select_same, std_gp_use= get_likes_engine(genotype, wst, tf, 
                         process_tools, keep_tools, varFilt_tools, dist_tools,comps_dists= comps_dists,
                         wind_sizes= wind_sizes, Nreps= Nreps, ncomps= ncomps, nan_char= nan_char, ind_min= ind_min,
                         dimN= dimN, metric= metric,std_diffs= std_diffs, avoid_range= avoid_range)


    ##
    ##
    ##
    ## local window - get and process

    local_l= wind_extract_func(genotype, idx= tf_pos, wind_sizes= wind_sizes,mask_pos= [])

    ## samp keep - samples without het or nan at this window
    samp_keep= keep_tools[0](local_l, **keep_tools[1])

    ### process local window - convert to haplotypes - keep only code_keep
    local_l= process_tools[0](local_l,**process_tools[1])

    ##
    ## local PCA and PCA transform
    ##
    pca_special= dr_obj.fit(local_l[samp_keep])
    featl= pca_special.transform(local_l)

    ## grid and likelihood.

    background, like_diet= window_exam(featl, samp_keep, select_same, std_gp_use, dist_store, dist_tools, 
                    labelf_select= labelf_select,correct_dist= correct_dist,
                   P= P, dimN= ncomps, metric= metric,expand= expand, std_diffs= std_diffs)

    ###


    tf_rec= rechap_func(background,like_diet,pca_special,
                    scale= 1, round_t= True)

    tf_proj= pca_special.transform(tf_rec.reshape(1,-1))[0]

    return background, like_diet, local_l, featl, tf_rec, tf_proj


    
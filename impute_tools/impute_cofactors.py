import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.cluster import estimate_bandwidth


def code_find(nwind,code_v= [1,9],binned= True,axis= 1):
    '''
    return obs or feats carrying codes in code_v vector.
    - returns binary array, value 1 indicating absence of code_v codes.
    '''
    code_check= np.zeros(nwind.shape)

    for cn in code_v:
        cd_m= nwind == cn
        cd_m= np.array(cd_m,dtype= int)
        code_check+= cd_m

    code_check= np.sum(code_check,axis= axis)

    if binned:
        code_check= code_check == 0
    
    return code_check




def bin_keep(lwind, keep= [2]):
    """
    turn data into binary array of presence of chosen code.
    """
    
    nl= np.zeros(lwind.shape)
    for char in keep:
        lt=  lwind == keep
        nl+= lt

    lwind= np.array(nl,dtype= int)
    
    return lwind


def sg_varSel(dist_var,proc= 'cluster',stt= 2, min_ind= 10):
    '''
    filter observations by variance. methods:
    - cluster: return cluster of observations with smallest mean.
    - standard: observations with inlier std value.
    - none: return full list.
    '''
    
    if proc == 'cluster':
        
        bandwidth = estimate_bandwidth(dist_var.reshape(-1,1), quantile=0.2)

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=False, min_bin_freq=35)

        ms.fit(dist_var.reshape(-1,1))
        labels_std = ms.labels_
        std_select = {y:[x for x in range(len(labels_std)) if labels_std[x] == y] for y in sorted(list(set(labels_std)))}
        
        std_gpmeans= {z: np.std([dist_var[x] for x in g]) for z,g in std_select.items() if z != -1}

        std_gp_use= sorted(std_gpmeans,key= std_gpmeans.get)
        d= 0
        idx = 0

        while d != 1:
            g=std_select[std_gp_use[idx]] 

            if len(g) >= min_ind:
                std_gp_use= list(g)
                d= 1

            idx+= 1
        
        return std_gp_use

        
    elif proc == 'standard':
        std_gp_use= [x for x in range(len(dist_var)) if (dist_var[x] - np.mean(dist_var)) / np.std(dist_var) < stt]
        return std_gp_use
        
    else:
        std_gp_use= list(range(len(dist_var)))

        return std_gp_use

    
def expand_grid(grid_array, expand= 1):
    
    centre= np.mean(grid_array,axis= 0)
    grid_array= np.array(grid_array) - centre 
    grid_array= grid_array * expand
    grid_array= grid_array + centre
    
    return grid_array


def index_convert2L(fsamp_keep, idx_l1, idx_l2, 
                    min_miss= 5):
    '''
    subselect two layers if indicies to use are positive in fsamp_keep boolean vector.
    '''
    sample_select= [idx_l1[x] for x in idx_l2]
    samp_exc= fsamp_keep[sample_select]
    
    select_miss= list(range(len(sample_select)))
    if len(samp_exc) - sum(samp_exc) > min_miss:
        select_miss= [x for x in range(len(sample_select)) if samp_exc[x]]
    
    sample_select= [sample_select[x] for x in select_miss]
    idx_l2= [idx_l2[x] for x in select_miss]
    
    return sample_select, idx_l2




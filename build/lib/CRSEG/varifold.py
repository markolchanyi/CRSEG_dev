import numpy as np
import math
from scipy.spatial import distance
from skimage import measure
from scipy.stats import entropy
import numexpr as ne


def hilbert_distance(cntr1,norm1,cntr2,norm2,sigma):
    return inner_hilbert_w_fast(cntr1,norm1,cntr1,norm1,sigma) + inner_hilbert_w_fast(cntr2,norm2,cntr2,norm2,sigma) - 2*inner_hilbert_w_fast(cntr1,norm1,cntr2,norm2,sigma)


def inner_hilbert_w(cntr1,norm1,cntr2,norm2,sigma):
    if np.asarray(cntr1).ndim == 3:
        cntr1 = np.asarray(cntr1)[0,...]
    else:
        cntr1 = np.asarray(cntr1)
    if np.asarray(norm1).ndim == 3:
        norm1 = np.asarray(norm1)[0,...]
    else:
        norm1 = np.asarray(norm1)
    if np.asarray(cntr2).ndim == 3:
        cntr2 = np.asarray(cntr2)[0,...]
    else:
        cntr2 = np.asarray(cntr2)
    if np.asarray(norm2).ndim == 3:
        norm2 = np.asarray(norm2)[0,...]
    else:
        norm2 = np.asarray(norm2)

    dw = 0
    for p in range(0,max(cntr1.shape)):
        for q in range(0,max(cntr2.shape)):
            if (np.dot(norm1[p,:],norm2[q,:]))**2 < 0.0005:
                continue
            else:
                dw += RBF_kernel(cntr1[p,:],cntr2[q,:],sigma)*((np.dot(norm1[p,:],norm2[q,:]))**2)
                # fast implementation of RBF kernel
                #dw += fast_RBF_kernel(((np.dot(norm1[p,:],norm2[q,:]))**2),cntr1[p,:],cntr2[q,:],sigma)
    return dw



def inner_hilbert_w_fast(cntr1,norm1,cntr2,norm2,sigma):
    if np.asarray(cntr1).ndim == 3:
        cntr1 = np.asarray(cntr1)[0,...]
    else:
        cntr1 = np.asarray(cntr1)
    if np.asarray(norm1).ndim == 3:
        norm1 = np.asarray(norm1)[0,...]
    else:
        norm1 = np.asarray(norm1)
    if np.asarray(cntr2).ndim == 3:
        cntr2 = np.asarray(cntr2)[0,...]
    else:
        cntr2 = np.asarray(cntr2)
    if np.asarray(norm2).ndim == 3:
        norm2 = np.asarray(norm2)[0,...]
    else:
        norm2 = np.asarray(norm2)

    # fill normals and precompute by matmul to avoid overhead from calling dp
    # a bajillion times
    norm_mat = (norm1 @ np.transpose(norm2))**2

    dw = 0
    for p in range(0,max(cntr1.shape)):
        for q in range(0,max(cntr2.shape)):
            dw += RBF_kernel(cntr1[p,:],cntr2[q,:],sigma)*norm_mat[p,q]
            # fast implementation of RBF kernel
            #dw += fast_RBF_kernel(((np.dot(norm1[p,:],norm2[q,:]))**2),cntr1[p,:],cntr2[q,:],sigma)
    return dw



def RBF_kernel(pt1,pt2,sigma):
    sqdist = np.linalg.norm(pt1-pt2)
    return math.exp(-sqdist/(2*(sigma**2)))

def fast_RBF_kernel(signal_var,pt1,pt2,sigma):
    s1 = 0
    s2 = 0
    for i in range(pt1.shape[0]):
        s1 += pt1[i]**2
        s2 += pt2[i]**2
    w = ne.evaluate('v * exp(-g * (A + B - 2 * C))', {
        'A': np.sqrt(s1),
        'B': np.sqrt(s2),
        'C': np.dot(pt1, pt2.T),
        'g': 1 / (2 * sigma**2),
        'v': signal_var
    })
    return w


def mesh(vol,step_size=1):
    verts, faces, normals, vals = measure.marching_cubes(vol,level=0,step_size=step_size,allow_degenerate=False)

    cts_normals = np.zeros((verts[faces].shape[0],3),dtype=np.float32)
    cts = np.zeros((verts[faces].shape[0],3),dtype=np.float32)

    for i in range(0,max(verts[faces].shape)):
        norm_vec = np.cross((verts[faces][i,1,:]-verts[faces][i,0,:]),(verts[faces][i,2,:]-verts[faces][i,0,:]))
        norm_vec /= np.linalg.norm(norm_vec)
        cts_normals[i,:] = norm_vec
        cts[i,:] = (verts[faces][i,0,:] + verts[faces][i,1,:] + verts[faces][i,2,:])/3

    return cts, cts_normals

def mesh_explicit(vol,step_size=1):
    verts, faces, normals, vals = measure.marching_cubes(vol,level=0,step_size=step_size,allow_degenerate=False)
    return verts, faces


def update_mesh(verts,faces,displacement):
    if np.asarray(verts).ndim == 3:
        verts = np.asarray(verts)[0,...]
    else:
        verts = np.asarray(verts)
    if np.asarray(faces).ndim == 3:
        faces = np.asarray(faces)[0,...]
    else:
        faces = np.asarray(faces)
    verts_updated = np.zeros_like(verts,dtype=np.float32)
    cts_normals = np.zeros((verts[faces].shape[0],3),dtype=np.float32)
    cts = np.zeros((verts[faces].shape[0],3),dtype=np.float32)

    for i in range(0,max(verts.shape)):
        verts_updated[i,:] = verts[i,:] + displacement[0,int(verts[i,0]),int(verts[i,1]),int(verts[i,2]),:]

    for i in range(0,max(verts_updated[faces].shape)):
        norm_vec = np.cross((verts_updated[faces][i,1,:]-verts_updated[faces][i,0,:]),(verts_updated[faces][i,2,:]-verts_updated[faces][i,0,:]))
        norm_vec /= np.linalg.norm(norm_vec)
        cts_normals[i,:] = norm_vec
        cts[i,:] = (verts_updated[faces][i,0,:] + verts_updated[faces][i,1,:] + verts_updated[faces][i,2,:])/3

    return cts, cts_normals



def varifold_distance(vol1,vol2,step_size=3,sigma=0.5,prethresh=True,threshval=0.1):

    if prethresh:
        vol1[vol1 < threshval] = 0
        vol1[vol1 > 0] = 1
        vol2[vol2 < threshval] = 0
        vol2[vol2 > 0] = 1

    cts1, cts_norms1 = mesh(vol1,step_size)
    cts2, cts_norms2 = mesh(vol2,step_size)

    return hilbert_distance(cts1, cts_norms1, cts2, cts_norms2, sigma)



def varifold_distance_nomesh(cts1, cts_norms1, cts2, cts_norms2,sigma=3):
    return hilbert_distance(cts1, cts_norms1, cts2, cts_norms2, sigma)




def dice_loss(vol1,vol2,prethresh=True,threshval=0.1):

    vol1 = vol1[0,0,...]
    vol2 = vol2[0,0,...]

    assert vol1.ndim == 3 and vol2.ndim == 3, "incorrect number of volume dimensions"

    if prethresh:
        vol1[vol1 < threshval] = 0
        vol1[vol1 > 0] = 1
        vol2[vol2 < threshval] = 0
        vol2[vol2 > 0] = 1

    assert vol1.shape == vol2.shape, "volumes are different shapes"

    intersect_count = 0
    union_count = np.count_nonzero(vol1) + np.count_nonzero(vol2)

    for i in range(0,vol1.shape[0]):
        for j in range(0,vol1.shape[1]):
            for k in range(0,vol1.shape[2]):
                if (vol1[i,j,k] > 0) and (vol2[i,j,k] > 0):
                    intersect_count += 1

    return 1 - ((2*intersect_count)/union_count)


def ce(vol1,vol2):
    vol1 = vol1.astype(np.float64)
    vol2 = vol2.astype(np.float64)
    vol2+=1e-15
    loss=-np.sum(vol1*np.log(vol2))
    return loss/float(vol1.shape[0]*vol1.shape[1]*vol1.shape[2])

import tensorflow as tf
from utils import *


#
# calculate lDDT scores and KL-divergence
# for the predicted disto- and angle-grams
#
L = a3m.shape[1]
scores = []

# KL-divergence (from background)
kl = [np.arange(L)]
for prefix in ['dist','omega','theta','phi']:
    p = contacts[prefix]
    p0 = contacts[prefix+'_bkgr']
    scores.append(np.sum(p*np.log(p/p0))/L/L)

# distance lDDT score
d = contacts['dist'][:,:,1:]
lddt = []
for k in [3,5,9,17]:
    kernel = np.array([[[1.]*k]])
    prod = d*correlate(d,kernel,'same')
    lddt.append(np.sum(prod[:,:,:26],axis=-1))
scores.append(0.25*np.sum(np.array(lddt))/np.sum(d[:,:,:26]))

# omega and theta lDDT scores
for ANG in ['omega','theta']:
    a = contacts[ANG][:,:,1:]
    lddt = []
    for k in [1,2,3,4]:
        kernel = np.array([[[1.]*(k*2+1)]])
        if k == 0:
            ang = a
        else:
            ang = np.concatenate([a[:,:,-k:],a,a[:,:,:k]],axis=-1)
        prod = a*correlate(ang,kernel,'valid')
        lddt.append(np.sum(prod,axis=-1))

    scores.append(0.25*np.sum(np.array(lddt))/np.sum(a))

# phi lDDT score
p = contacts['phi'][:,:,1:]
lddt = []
for k in [1,2,3,4]:
    kernel = np.array([[[1.]*(k*2+1)]])
    if k == 0:
        phi = p
    else:
        phi = np.concatenate([np.flip(p[:,:,:k],axis=-1),p,np.flip(p[:,:,-k:],axis=-1)], axis=-1)
        prod = p*correlate(phi,kernel,'valid')
        lddt.append(np.sum(prod,axis=-1))

scores.append(0.25*np.sum(np.array(lddt))/np.sum(p))


# probability of top L contacts
cont=np.sum(contacts['dist'][:,:,1:13],axis=-1)
a = np.arange(L)
mask = np.abs(a.reshape([1,L])-a.reshape([L,1]))
for sep in [6,12,24]:
    mtx = cont*(mask>=sep)
    sel = np.sort(mtx.flatten())[-L*2:]
    scores.append(np.mean(sel))

print(scores)

#
# save distograms & anglegrams
#
np.savez_compressed(npz_file,
    scores=np.array(scores),
    dist=contacts['dist'],
    omega=contacts['omega'],
    theta=contacts['theta'],
    phi=contacts['phi'],
    cont=np.sum(contacts['dist'][:,:,1:13],axis=-1),
    dist_bkgr=contacts['dist_bkgr'],
    omega_bkgr=contacts['omega_bkgr'],
    theta_bkgr=contacts['theta_bkgr'],
    phi_bkgr=contacts['phi_bkgr'])

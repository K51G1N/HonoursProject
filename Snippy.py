import numpy as np
import matplotlib.pyplot as plt
import pdb

def MainMethod(outdir, Train):
    if Train == True:
        datapath = outdir[1]
        savepath = outdir[3]
    else:
        datapath = outdir[2]
        savepath = outdir[6]

    Dirty = np.load(datapath+'/dirtyArray.npy')
    psLocations = np.load(datapath+'/PSLocations.npy')
    esLocations = np.load(datapath+'/NonPSLocations.npy')
    # Locations = np.load("/home/keagan/Documents/Honours/Project/Output/TempTest/meta_info_%d/meta_info/locations_%d.npy"%(i,i))
    for j in range(len(psLocations)):
        cutout = Dirty[psLocations[j][0]:psLocations[j][0]+51, psLocations[j][1]:psLocations[j][1]+51]
        source = savepath+'/PS/PointSource_%d.npy'%(j)
        np.save(source, cutout)
    for k in range(len(esLocations)):
        cutout = Dirty[esLocations[k][0]:esLocations[k][0] + 51, esLocations[k][1]:esLocations[k][1] + 51]
        source = savepath + '/NPS/ExtendedSource_%d.npy' % (k)
        np.save(source, cutout)

import numpy as np


def MainMethod(outdir, nsource, Train):

    if nsource == 'H':
        nsource = 100
    else:
        nsource = 1000

    if Train == True:
        datapath = outdir[1]
        savepath = outdir[3]
    else:
        datapath = outdir[2]
        savepath = outdir[6]
    Dirty = np.load(datapath + '/dirtyArray.npy')
    Ix = np.load(datapath + "/Ix.npy")
    Iy = np.load(datapath + "/Iy.npy")
    # Dirty = np.load("/home/keagan/Documents/Honours/Project/%s/dirtyArray.npy"%folder)

    #51x51
    for i in range(len(Ix)):
        if(i < nsource):
            cutout = Dirty[Ix[i]:Ix[i]+51, Iy[i]:Iy[i]+51]
            source = savepath + "/PS/PS_%d"%i
            np.save(source, cutout)
        else:
            cutout = Dirty[Ix[i]:Ix[i]+51, Iy[i]:Iy[i]+51]
            source = savepath + "/NPS/NPS_%d"%i
            np.save(source, cutout)

#!/usr/bin/env python
import os
from ducc0.wgridder import ms2dirty, dirty2ms
from ducc0.fft import r2c, c2r
import matplotlib.pyplot as plt
import numpy as np


import SnipPS
iFs = np.fft.ifftshift
Fs = np.fft.fftshift
complex_type = np.complex64
real_type = np.float32

class PSF(object):
    def __init__(self, psf, nthreads=1, imsize=None):
        self.nthreads = nthreads
        nx_psf, ny_psf = psf.shape
        if imsize is not None:
            nx, ny = imsize
            if nx > nx_psf or ny > ny_psf:
                raise ValueError("Image size can't be smaller than PSF size")
        else:
            # if imsize not passed in assume PSF is twice the size of image
            nx = nx_psf//2
            ny = ny_psf//2
        npad_xl = (nx_psf - nx)//2
        npad_xr = nx_psf - nx - npad_xl
        npad_yl = (ny_psf - ny)//2
        npad_yr = ny_psf - ny - npad_yl
        self.padding = ((npad_xl, npad_xr), (npad_yl, npad_yr))
        self.ax = (0,1)
        self.unpad_x = slice(npad_xl, -npad_xr)
        self.unpad_y = slice(npad_yl, -npad_yr)
        self.lastsize = ny + np.sum(self.padding[-1])
        self.psf = psf
        psf_pad = iFs(psf, axes=self.ax)
        self.psfhat = r2c(psf_pad, axes=self.ax, forward=True, nthreads=nthreads, inorm=0)


    def convolve(self, x):
        xhat = iFs(np.pad(x, self.padding, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
        xhat = c2r(xhat * self.psfhat, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
        return Fs(xhat, axes=self.ax)[self.unpad_x, self.unpad_y]

lightspeed = 3e8
nthreads = 8

def hogbom(ID, comboLocs, PSF, gamma=0.1, pf=0.1, maxit=10000):
    nx, ny = ID.shape
    PSLocations = []
    NonLocations = []
    x = np.zeros((nx, ny), dtype=ID.dtype) 
    IR = ID.copy()
    IRmax = IR.max()
    tol = pf*IRmax
    for k in range(maxit):
        if IRmax < tol:
            break
        p, q = np.argwhere(IR == IRmax).squeeze()
        if np.size(p) > 1:
            p = p.squeeze()[0]
            q = q.squeeze()[0]
        xhat = IR[p, q]
        x[p, q] += gamma * xhat
        IR -= gamma * xhat * PSF[nx-p:2*nx - p, ny-q:2*ny - q]
        IRmax = IR.max()

        if ([p, q] not in PSLocations):
            if (p, q) not in comboLocs:
                NonLocations.append([p, q])
            else:
                PSLocations.append([p, q])

    return x, IR, np.array(PSLocations), np.array(NonLocations)

def create_output_dirs(outdir, N=1):
    """create the output folders for the simulations"""

    base_path = outdir + "/"
    test_path = outdir + "/Test/"
    folders = [base_path, test_path]
    folders.append(test_path + "/" + str(1))
    folders.append(test_path + "/" + str(1) + "/PointSources")
    folders.append(test_path + "/" + str(1) + "/NonPointSources")
    folders.append(outdir + "/meta_info")

    try:
        os.system("rm -rf %s" % base_path)
    except:
        pass

    # create data folders if they do not exist
    for folder in folders:
        if os.path.isdir(folder):
            print(folder, "already exist")
        else:
            os.mkdir(folder)

    print("Output directory for simulations setup successfully")



def main(basedir, PowerMode, Sources, Train):

    if Train == True:
        path = basedir[1]
    else:
        path = basedir[2]
    combinedLocs = []
    xloc = []
    yloc = []
    # load in uv-coverage
    uvw = np.load("data/uvw.npy")
    nrow, _ = uvw.shape
    uvw[:, 2] = 0.0
    # set frequency (this will determine image size - low freq -> smaller image)
    freq = np.array([1e9])

    # determine required pixel size
    uvmax = np.maximum(np.abs(uvw[:, 0]).max(), np.abs(uvw[:, 1]).max())
    cell_N = 1.0/(2*uvmax*freq.max()/lightspeed)  # Nyquist
    super_resolution_factor = 1.5
    cell_rad = cell_N/super_resolution_factor

    fov = np.deg2rad(1.0)  # 1 degree field of view
    npix = int(fov/cell_rad)
    if npix%2:  # gridder wants and even number of pixels
        npix += 1  

    print("npix = ", npix)

    # create a PSF
    weights = np.ones((nrow, 1), dtype=np.float32)
    psf = ms2dirty(uvw=uvw, freq=freq, ms=weights.astype(np.complex64), 
                   npix_x=2*npix, pixsize_x=cell_rad, npix_y=2*npix, pixsize_y=cell_rad,  # psf assumed to be twice the size of the image
                   epsilon=1e-6, do_wstacking=False, nthreads=nthreads, double_precision_accumulation=True)
    wsum = psf.max()
    psf /= wsum  # needs to be normalised for cleaning
    psfo = PSF(psf, nthreads=nthreads, imsize=(npix, npix))

    # make a model
    model = np.zeros((npix, npix))

    if Sources == 'H':
        nsource = 100
    elif Sources == "T":
        nsource = 1000


    # source locations: Create 200. 100 will be +'ve cases, 100 will be -'ve cases.
    Ix = np.random.randint(int(0.05 * npix), int(0.95 * npix), int(2 * nsource))
    Iy = np.random.randint(int(0.05 * npix), int(0.95 * npix), int(2 * nsource))

    for i in range(nsource):
        combo = (Ix[i], Iy[i])
        combinedLocs.append(combo)
    # create_output_dirs(basedir)
    # outdir = basedir + "/Test"


    #Save Ix and Iy locations
    np.save(path + '/Ix.npy',Ix)
    np.save(path + '/Iy.npy', Iy)

    print('LOCS')
    # print("X: \n", Ix, "\n Y: \n",Iy)
    # First 100 are point sources.'

    #Fluxes
    # ss = np.random.power(0.1, nsource)
    # aflux = -np.sort(-ss)
    # peakflux = 2
    # minflux = 0.1 #1e-4
    # flux = (peakflux - minflux)*(aflux - np.min(aflux))/(np.max(aflux) - np.min(aflux)) + minflux

    # # NORMAL
    # # flux =
    # # POWER
    # flux =

    if PowerMode == 0:
        flux = -np.sort(-(np.random.power(0.1, nsource)))
    else:
        flux = -np.sort(-(0.1 + np.abs(np.random.randn(nsource))))


    # flux = 0.1 + np.abs(np.random.randn(nsource))
    model[Ix[0:nsource], Iy[0:nsource]] = flux  # positive flux

    # model visibilities
    model_vis = dirty2ms(uvw=uvw, freq=freq, dirty=model.astype(real_type),
                         wgt=weights.astype(real_type),
                         pixsize_x=cell_rad, pixsize_y=cell_rad, epsilon=1e-6,
                         do_wstacking=False, nthreads=nthreads)

    # generate noise
    standard_deviation = np.sqrt(1. / weights)
    noise = (standard_deviation * np.random.randn(*weights.shape) / np.sqrt(2) +
             1.0j * standard_deviation * np.random.randn(*weights.shape) / np.sqrt(2))

    data = model_vis + noise.astype(complex_type)

    # now the dirty image
    dirty = ms2dirty(uvw=uvw, freq=freq, ms=data.astype(complex_type),
                     wgt=weights.astype(real_type),
                     npix_x=npix, pixsize_x=cell_rad,
                     npix_y=npix, pixsize_y=cell_rad,
                     epsilon=1e-6, do_wstacking=False,
                     nthreads=nthreads,
                     double_precision_accumulation=True) / wsum
    np.save(path, '/NoPadDirtyImage.npy',dirty)

    dirty_pad = np.pad(dirty, (26,), 'constant', constant_values=0)

    np.save(path + '/dirtyArray.npy', dirty_pad)

    # clean image
    # model_rec, residual = hogbom(dirty, psf)
    pf = 0.01
    model_rec, residual, PSlocations, NonLocs = hogbom(dirty.copy(), combinedLocs, psf, gamma=0.1, pf=pf, maxit=10000)

    source_mod = path + "/model_rec.np"
    source_res = path + "/residual.npy"
    source_ps_loc = path + "/PSLocations.npy"
    source_Nps_loc = path + "/NonPSLocations.npy"
    # sources_Modv.append(source_mod)
    # sources_Resv.append(source_res)
    # sources_Locv.append(source_loc)

    # Save
    np.save(source_mod, model_rec)
    np.save(source_res, residual)
    np.save(source_ps_loc, PSlocations)
    np.save(source_Nps_loc, NonLocs)



    print(xloc, yloc)
    plt.figure('psf')
    plt.imshow(psf, vmax=0.1)
    plt.savefig('%s/PSF_1.png' % (path))
    plt.colorbar()
    plt.savefig('%s/PSF.png'%(path))
    plt.close()

    plt.figure('ground truth')
    plt.imshow(model.T, vmax=0.001)
    plt.savefig("%s/GT_1.png" % (path))
    plt.colorbar()
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("%s/GT.png"%(path))
    plt.close()

    plt.figure('dirty')
    plt.imshow(dirty)
    plt.savefig("%s/DIRT_1.png" % (path))
    plt.colorbar()
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("%s/DIRT.png"%(path))
    plt.close()

    plt.figure('truth recovered')
    plt.imshow(model_rec, vmax=0.001)
    plt.savefig('%s/TruthRecovered_1.png' % (path))
    plt.colorbar()
    plt.savefig('%s/TruthRecovered.png'%(path))
    plt.close()

    plt.figure('residual')
    plt.imshow(residual)
    plt.savefig('%s/Residual_1.png' % (path))
    plt.colorbar()
    plt.savefig('%s/Residual.png'%(path))
    plt.close()

    # SnipPS.MainMethod(basedir, nsource)

    if Train == True:
        print('Data Acquisition Complete')
    else:
        print('Unseen Data Acquisition Complete')

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     sys.exit('Usage: %s output-folder' % sys.argv[0])
    #
    # basedir = sys.argv[1]

    #Set parameters
    PowerMode = 1 #0 Powerlaw, 1 Normal Flux
    Sources = "T" # H hundreds or T thousands
    basedir_Train = "/media/keagan/Digitide 1TB/FINAL/%s_%s/X_%s_%s"%(Sources, PowerMode, Sources, PowerMode)
    np.random.seed(123)
    main(basedir_Train, PowerMode, Sources)
    # MultiModel.MainMethod(basedir_Train)

    #Set parameters
    PowerMode = 1 #0 Powerlaw, 1 Normal Flux
    Sources = "H" # H hundreds or T thousands
    basedir_Train = "/media/keagan/Digitide 1TB/FINAL/%s_%s/X_%s_%s"%(Sources, PowerMode, Sources, PowerMode)
    np.random.seed(123)
    main(basedir_Train, PowerMode, Sources)
    # MultiModel.MainMethod(basedir_Train)


    PowerMode = 1  # 0 Powerlaw, 1 Normal Flux
    Sources = "H"  # H hundreds or T thousands
    basedir_Test = "/media/keagan/Digitide 1TB/FINAL/%s_%s/Y_%s_%s"%(Sources, PowerMode, Sources, PowerMode)
    np.random.seed(1234)
    main(basedir_Test, PowerMode, Sources)
    # MultiModel_Test.MainMethod(basedir_Test, basedir_Train)

    PowerMode = 1  # 0 Powerlaw, 1 Normal Flux
    Sources = "T"  # H hundreds or T thousands
    basedir_Test = "/media/keagan/Digitide 1TB/FINAL/%s_%s/Y_%s_%s" % (Sources, PowerMode, Sources, PowerMode)
    np.random.seed(1234)
    main(basedir_Test, PowerMode, Sources)
    # MultiModel_Test.MainMethod(basedir_Test, basedir_Train)
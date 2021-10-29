#!/usr/bin/env python
import os
import sys
from ducc0.wgridder import ms2dirty, dirty2ms
from ducc0.fft import r2c, c2r
import matplotlib.pyplot as plt
# import cv2 as cv
import numpy as np

iFs = np.fft.ifftshift
Fs = np.fft.fftshift
lightspeed = 3e8
nthreads = 8
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
            nx = nx_psf // 2
            ny = ny_psf // 2
        npad_xl = (nx_psf - nx) // 2
        npad_xr = nx_psf - nx - npad_xl
        npad_yl = (ny_psf - ny) // 2
        npad_yr = ny_psf - ny - npad_yl
        self.padding = ((npad_xl, npad_xr), (npad_yl, npad_yr))
        self.ax = (0, 1)
        self.unpad_x = slice(npad_xl, -npad_xr)
        self.unpad_y = slice(npad_yl, -npad_yr)
        self.lastsize = ny + np.sum(self.padding[-1])
        self.psf = psf
        psf_pad = iFs(psf, axes=self.ax)
        self.psfhat = r2c(psf_pad, axes=self.ax, forward=True, nthreads=nthreads, inorm=0)

    def convolve(self, x):
        xhat = iFs(np.pad(x, self.padding, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
        xhat = c2r(xhat * self.psfhat, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2,
                   nthreads=self.nthreads)
        return Fs(xhat, axes=self.ax)[self.unpad_x, self.unpad_y]

def hogbom(ID, comboLocs, PSF, gamma=0.1, pf=0.1, maxit=10000):
    nx, ny = ID.shape
    locations = []
    esLocations = []
    x = np.zeros((nx, ny), dtype=ID.dtype)
    IR = ID.copy()
    IRmax = IR.max()
    tol = pf * IRmax
    for k in range(maxit):
        if IRmax < tol:
            break
        p, q = np.argwhere(IR == IRmax).squeeze()
        if np.size(p) > 1:
            p = p.squeeze()[0]
            q = q.squeeze()[0]
        xhat = IR[p, q]
        x[p, q] += gamma * xhat
        IR -= gamma * xhat * PSF[nx - p:2 * nx - p, ny - q:2 * ny - q]
        IRmax = IR.max()

        if ([p, q] not in locations):
            if (p, q) not in comboLocs[0:999]:
                esLocations.append([p,q])
            else:
                locations.append([p, q])

    return x, IR, np.array(locations), np.array(esLocations)

def Gaussian2D(xin, yin, GaussPar=(1., 1., 0.), normalise=True, nsigma=5):
    S0, S1, PA = GaussPar
    Smaj = np.maximum(S0, S1)
    Smin = np.minimum(S0, S1)
    A = np.array([[1. / Smin ** 2, 0],
                  [0, 1. / Smaj ** 2]])

    c, s, t = np.cos, np.sin, np.deg2rad(-PA)
    R = np.array([[c(t), -s(t)],
                  [s(t), c(t)]])
    A = np.dot(np.dot(R.T, A), R)
    sOut = xin.shape
    # only compute the result out to 5 * emaj
    extent = (nsigma * Smaj)**2
    xflat = xin.squeeze()
    yflat = yin.squeeze()
    ind = np.argwhere(xflat**2 + yflat**2 <= extent).squeeze()
    idx = ind[:, 0]
    idy = ind[:, 1]
    x = np.array([xflat[idx, idy].ravel(), yflat[idx, idy].ravel()])
    R = np.einsum('nb,bc,cn->n', x.T, A, x)
    # need to adjust for the fact that GaussPar corresponds to FWHM
    fwhm_conv = 2 * np.sqrt(2 * np.log(2))
    tmp = np.exp(-fwhm_conv * R)
    gausskern = np.zeros(xflat.shape, dtype=np.float64)
    gausskern[idx, idy] = tmp

    if normalise:
        gausskern /= np.sum(gausskern)
    return np.ascontiguousarray(gausskern.reshape(sOut),
                                dtype=np.float64)
def place_Gauss2D(xlocs, ylocs, fluxs, nx, ny, emaj0, emin0, pa0):
    x = np.arange(-nx, nx)
    y = np.arange(-ny, ny)

    model = np.zeros((2*nx, 2*ny))

    nsource = len(xlocs)

    for s, I0, l0, m0 in zip(range(nsource), fluxs, xlocs, ylocs):
        emaj = emaj0 * (1 + 0.5*np.random.randn())
        emin = emin0 * (1 + 0.5*np.random.randn())
        pa = pa0 * np.random.randn()

        x0 = nx//2 + x - xlocs[s]
        y0 = ny//2 + y - ylocs[s]

        xx, yy = np.meshgrid(x0, y0)

        model += I0 * Gaussian2D(xx, yy, GaussPar=(emaj, emin, pa),
                                 normalise=True, nsigma=5)

    return model

def create_output_dirs(outdir, N=10):
    """create the output folders for the simulations"""

    base_path = outdir + "/"
    test_path = outdir + "/Test/"
    folders = [base_path, test_path]

    # for i in range(1, N):
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


def main(folders, PowerMode, Sources, Train):
    combinedLocs = []
    xloc = []
    yloc = []
    # load in uv-coverage

    if Train == True:
        path = folders[1]
    else:
        path = folders[2]

    uvw = np.load("data/uvw.npy")
    nrow, _ = uvw.shape
    uvw[:, 2] = 0.0
    # set frequency (this will determine image size - low freq -> smaller image)
    freq = np.array([1e9])

    # determine required pixel size
    uvmax = np.maximum(np.abs(uvw[:, 0]).max(), np.abs(uvw[:, 1]).max())
    cell_N = 1.0 / (2 * uvmax * freq.max() / lightspeed)  # Nyquist
    super_resolution_factor = 1.5
    cell_rad = cell_N / super_resolution_factor

    fov = np.deg2rad(1.0)  # 1 degree field of view
    npix = int(fov / cell_rad)
    if npix % 2:  # gridder wants and even number of pixels
        npix += 1

    print("npix = ", npix)

    # create a PSF
    sigma = 1.0  # [0.1, 0.25, 0.5, 0.75, 1.0, 10.0]
    # Sigma = sigma**2 * np.eye(nrow)
    weights = np.ones((nrow, 1), dtype=np.float32) / sigma ** 2
    psf = ms2dirty(uvw=uvw, freq=freq, ms=weights.astype(np.complex64),
                   npix_x=2 * npix, pixsize_x=cell_rad, npix_y=2 * npix, pixsize_y=cell_rad,
                   # psf assumed to be twice the size of the image
                   epsilon=1e-6, do_wstacking=False, nthreads=nthreads, double_precision_accumulation=True)
    wsum = psf.max()
    psf /= wsum  # needs to be normalised for cleaning
    psfo = PSF(psf, nthreads=nthreads, imsize=(npix, npix))

    # make a model
    model = np.zeros((npix, npix))
    # nsource = 100

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
    # Save Ix and Iy locations

    # create_output_dirs(basedir

    np.save(path + '/Ix.npy', Ix)
    np.save(path + '/Iy.npy', Iy)
    np.save(path + '/CombinedIxIy.npy', combinedLocs)


    if PowerMode == 0:
        flux = -np.sort(-(np.random.power(0.1, nsource)))
    else:
        flux = -np.sort(-(0.1 + np.abs(np.random.randn(nsource))))

    model[Ix[0:nsource], Iy[0:nsource]] = flux  # positive flux


    # extended sources (Gaussian blobs)
    if Sources == 'H':
        nfat_source = 10
    elif Sources == "T":
        nfat_source = 100

    # nfat_source = 10

    # source locations: Create 200. 100 will be +'ve cases, 100 will be -'ve cases.
    Ix_fat = np.random.randint(int(0.05*npix), int(0.95*npix), int(nfat_source)) #CHECK: that we are not placing
    Iy_fat = np.random.randint(int(0.05*npix), int(0.95*npix), int(nfat_source))
    fat_excess = 10
    I0_fat = 0.1 + fat_excess * np.abs(np.random.randn(nfat_source))

    model_fat = place_Gauss2D(Ix_fat, Iy_fat, I0_fat, npix, npix, 50.0, 50.0, 180.0)

    model_fat[npix//2:3*npix//2, npix//2:3*npix//2] += model

    # model visibilities
    model_vis = dirty2ms(uvw=uvw, freq=freq, dirty=model_fat.astype(real_type),
                         wgt=weights.astype(real_type),
                         pixsize_x=cell_rad, pixsize_y=cell_rad, epsilon=1e-6,
                         do_wstacking=False, nthreads=nthreads)

    # model_vis = dirty2ms(uvw=uvw, freq=freq, dirty=model.astype(real_type),
    #                      wgt=weights.astype(real_type),
    #                      pixsize_x=cell_rad, pixsize_y=cell_rad, epsilon=1e-6,
    #                      do_wstacking=False, nthreads=nthreads)
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
    #
    np.save(path + '/dirtyArray.npy', dirty_pad)





    # # make a dirty image
    # dirty = psfo.convolve
    # # HIDE
    # # plt.imshow(dirty)
    # # plt.show()
    # # cv.imshow('Dirty', dirty)
    # # cv.waitKey(0)
    # dirty_pad = np.pad(dirty, (26,), 'constant', constant_values=0)
    # # HIDE
    # # plt.imshow(dirty_pad)
    # # plt.show()
    # # Save
    # # import pdb; pdb.set_trace()
    # np.save(basedir + '/dirtyArray.npy', dirty_pad)
    # cv.imshow('Dirty pad', dirty_pad)
    # cv.waitKey(0)
    # clean image
    # model_rec, residual, locations = hogbom(dirty, psf)

    # Iteratively saves each item pf has an affect on
    # for i in range(1, 10):
    #     pf = i / 10
    pf = 0.01
    model_rec, residual, locations, esLocs = hogbom(dirty.copy(), combinedLocs, psf, gamma=0.1, pf=pf, maxit=10000)
    # if i == 1:
    #     print(locations[0])
    #     # print(len(locations[0][0])


    source_mod = path + '/model_rec.npy'
    source_res = path + '/residual.npy'
    source_ps_loc = path + '/PSLocations.npy'
    source_es_loc = path + '/NonPSLocations.npy'




    # sources_Modv.append(source_mod)
    # sources_Resv.append(source_res)
    # sources_Locv.append(source_loc)

    # Save
    np.save(source_mod, model_rec)
    np.save(source_res, residual)
    np.save(source_ps_loc, locations)
    np.save(source_es_loc, esLocs)
    # print(sources_Modv)
    # print(sources_Resv)
    # print(sources_Locv)

    # with open('/home/keagan/Documents/Honours/Project/Output/Test/paths.txt', 'w') as f:
    #     #Change
    #     for i in range(0,1):
    #         modv = sources_Modv[i]+"\n"
    #         resv = sources_Resv[i]+"\n"
    #         locv = sources_Locv[i]+"\n"

    #         f.write(modv)
    #         f.write(resv)
    #         f.write(locv)
    # Change
    # # print(xloc, yloc)
    # # HIDE
    # plt.figure('psf')
    # plt.imshow(psf, vmax=0.1)
    # #Save
    # plt.savefig('/home/keagan/Documents/Honours/Project/Output/Test/PSF.jpg', bbox_inches='tight',pad_inches=0)
    # # plt.colorbar()
    # # HIDE
    # plt.figure('ground truth')
    # plt.imshow(model.T, vmax=0.001)
    # # plt.colorbar()
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # #Save
    # plt.savefig("/home/keagan/Documents/Honours/Project/Output/Test/GT.jpg", bbox_inches='tight', pad_inches=0)
    #
    # plt.figure('dirty')
    # # HIDE
    # plt.imshow(dirty)
    # # plt.colorbar()
    # plt.gca().set_axis_off()
    #
    #
    #
    #
    #
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # #Save
    # plt.savefig("/home/keagan/Documents/Honours/Project/Output/Test/DIRT.jpg", bbox_inches='tight',pad_inches=0)
    #
    # plt.figure('truth recovered')
    # # HIDE
    # plt.imshow(model_rec, vmax=0.001)
    # #Save
    # plt.savefig("/home/keagan/Documents/Honours/Project/Output/Test/TruthRecovered.jpg", bbox_inches='tight',pad_inches=0)
    #
    # #plt.colorbar()
    # # HIDE
    # plt.figure('residual')
    # plt.imshow(residual)
    # #Save
    # plt.savefig("/home/keagan/Documents/Honours/Project/Output/Test/Residual.jpg", bbox_inches='tight',pad_inches=0)
    # #plt.colorbar()
    # # HIDE
    # # plt.show()
    # plt.close('all')

    # plt.imshow(dirty_pad)
    # plt.show()
    # plt.clf()
    # plt.cla()
    # plt.close()
# **************************************************************************************************************************
#     Displays Images

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

    if Train == True:
        print('Data Acquisition Complete')
    else:
        print('Unseen Data Acquisition Complete')
    # import Snippy
    #
    # Snippy.MainMethod(path)


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     sys.exit('Usage: %s output-folder' % sys.argv[0])
    #
    # basedir = sys.argv[1]

    PowerMode = 1  # 0 Powerlaw, 1 Normal Flux
    Sources = "T"  # H hundreds or T thousands
    basedir_Test = "/media/keagan/Digitide 1TB/Compare Models/"
    np.random.seed(123)
    main(basedir_Test, PowerMode, Sources)
    # MultiModel_Test.MainMethod(basedir_Test, basedir_Train)

    # np.random.seed(1)
    # main("/media/keagan/Digitide 1TB/Keagan/Beta_OneThousand-ES-Test")
    # CHECK: Power/Normal, Sources Nr

    # Set parameters
    # PowerMode = 1  # 0 Powerlaw, 1 Normal Flux
    # Sources = "T"  # H hundreds or T thousands
    # basedir = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y__%s_%s"%(Sources, PowerMode)
    # main(basedir, PowerMode, Sources)
    #
    # PowerMode = 0  # 0 Powerlaw, 1 Normal Flux
    # Sources = "T"  # H hundreds or T thousands
    # basedir = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_PFVary_%s_%s" % (Sources, PowerMode)
    # main(basedir, PowerMode, Sources)
    #
    # PowerMode = 1  # 0 Powerlaw, 1 Normal Flux
    # Sources = "H"  # H hundreds or T thousands
    # basedir = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_PFVary_%s_%s" % (Sources, PowerMode)
    # main(basedir, PowerMode, Sources)
    #
    # PowerMode = 0  # 0 Powerlaw, 1 Normal Flux
    # Sources = "H"  # H hundreds or T thousands
    # basedir = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_ES_%s_%s" % (Sources, PowerMode)
    # main(basedir, PowerMode, Sources)
    #
    # x_es_h_1_final


    # np.random.seed(123)
    # # X_ES_H_1_final
    # PowerMode = 1  # 0 Powerlaw, 1 Normal Flux
    # Sources = "T"  # H hundreds or T thousands
    # basedir = "/media/keagan/Digitide 1TB/Keagan/FINAL/X_ES_%s_%s_final" % (Sources, PowerMode)
    # main(basedir, PowerMode, Sources)

    # np.random.seed(1234)
    # Y_ES_H_1_final
    #
    # PowerMode = 1  # 0 Powerlaw, 1 Normal Flux
    # Sources = "H"  # H hundreds or T thousands
    # basedir = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_ES_%s_%s_final" % (Sources, PowerMode)
    # main(basedir, PowerMode, Sources)
    #
    # # Y_ES_T_1_final
    #
    # np.random.seed(1234)
    # PowerMode = 1  # 0 Powerlaw, 1 Normal Flux
    # Sources = "T"  # H hundreds or T thousands
    # basedir = "/media/keagan/Digitide 1TB/Keagan/ZETA/Y_ES_%s_%s_final" % (Sources, PowerMode)
    # main(basedir, PowerMode, Sources)


    # main("/media/keagan/Digitide 1TB/Keagan/Beta_OneThousandPower-ES-Test")
    # main("/media/keagan/Digitide 1TB/Keagan/Beta_OneHundred-ES-Test")
    # main("/media/keagan/Digitide 1TB/Keagan/Beta_OneHundredPower-ES-Test")

    # X


    # Y
'''
    # Set parameters
    PowerMode = 1  # 0 Powerlaw, 1 Normal Flux
    Sources = "T"  # H hundreds or T thousands
    basedir_Train = "/media/keagan/Digitide 1TB/FINAL/ES_%s_%s/X_ES_%s_%s" % (Sources, PowerMode, Sources, PowerMode)
    np.random.seed(123)
    main(basedir_Train, PowerMode, Sources)
    # MultiModel.MainMethod(basedir_Train)

    # Set parameters
    PowerMode = 1  # 0 Powerlaw, 1 Normal Flux
    Sources = "H"  # H hundreds or T thousands
    basedir_Train = "/media/keagan/Digitide 1TB/FINAL/ES_%s_%s/X_ES_%s_%s" % (Sources, PowerMode, Sources, PowerMode)
    np.random.seed(123)
    main(basedir_Train, PowerMode, Sources)
    # MultiModel.MainMethod(basedir_Train)

    PowerMode = 1  # 0 Powerlaw, 1 Normal Flux
    Sources = "H"  # H hundreds or T thousands
    basedir_Test = "/media/keagan/Digitide 1TB/FINAL/ES_%s_%s/Y_ES_%s_%s" % (Sources, PowerMode, Sources, PowerMode)
    np.random.seed(1234)
    main(basedir_Test, PowerMode, Sources)
    # MultiModel_Test.MainMethod(basedir_Test, basedir_Train)

    PowerMode = 1  # 0 Powerlaw, 1 Normal Flux
    Sources = "T"  # H hundreds or T thousands
    basedir_Test = "/media/keagan/Digitide 1TB/FINAL/ES_%s_%s/Y_ES_%s_%s" % (Sources, PowerMode, Sources, PowerMode)
    np.random.seed(1234)
    main(basedir_Test, PowerMode, Sources)
    # MultiModel_Test.MainMethod(basedir_Test, basedir_Train)

'''


import os
import nibabel as nib
import itertools
import numpy as np

voxelmorph_labels = [0,     # background label
                     1, 2,  # XX label (left/right)
                     3, 4,  # XX label (left/right)
                     5      # XX label (left/right)
                     ]

def util_bbox(img):
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N-1):
        nonezero = np.any(img, axis = ax)
        out.append(np.where(nonezero)[0][[0, -1]])
    return out

def random_points_from_mask(num, mask):
    coors = np.nonzero(mask)
    idx = np.random.randint(len(coors[0]), size = num)
    coors = np.transpose(np.array([coors[0][idx], coors[1][idx], coors[2][idx]], dtype = np.int16))
    return coors

def get_patch(img, patchshape, coor, cval = 0):
    patch = np.full(patchshape, cval)

    imgshape = np.array(img.shape, dtype = np.int16)
    cropshape = np.array(patchshape, dtype = np.int16)
    coor = np.array(coor, dtype = np.int16)

    pos_s = coor - np.floor(cropshape / 2)
    pos_e = pos_s + cropshape

    si = (np.maximum([0,0,0], pos_s)).astype(np.int16)
    ei = (np.minimum(imgshape, pos_e)).astype(np.int16)

    sp = (np.abs(np.minimum([0,0,0], pos_s))).astype(np.int16)
    ep = (sp + (ei - si)).astype(np.int16)

    mask = imgshape < cropshape
    cropshape[mask] = imgshape[mask]

    patch[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]] = img[si[0]:ei[0], si[1]:ei[1], si[2]:ei[2]]
    return patch

def extract_patches_nifti(niftiimg, patchshape, patchnum, savepath, niftiseg = None, ROI = [], ROIratio = 0.8, cval = 0):
    img = niftiimg.get_fdata()
    mask = np.full(img.shape, True)
    count = 0

    if len(ROI) == img.ndim:
        # extract patches from non-ROI part of the img
        patchnum_nroi = patchnum - int(ROIratio * patchnum)
        mask[ROI[0][0]:ROI[0][1], ROI[1][0]:ROI[1][1], ROI[2][0]:ROI[2][1]] = False
        coors = random_points_from_mask(patchnum_nroi, mask)
        for coor in coors:
            patch = get_patch(img, patchshape, coor, cval=cval)
            nib.save(nib.Nifti1Image(patch.astype(np.int16), niftiimg.affine), os.path.join(savepath, str(count)+'.nii.gz'))
            if niftiseg is not None:
                patch = get_patch(niftiseg.get_fdata(), patchshape, coor, cval=0)
                nib.save(nib.Nifti1Image(patch.astype(np.int16), niftiimg.affine), os.path.join(savepath, str(count)+'_label.nii.gz'))
            count = count + 1
        patchnum = int(ROIratio * patchnum)
        mask = ~mask

    coors = random_points_from_mask(patchnum, mask)
    for coor in coors:
        patch = get_patch(img, patchshape, coor, cval=cval)
        nib.save(nib.Nifti1Image(patch.astype(np.int16), niftiimg.affine), os.path.join(savepath, str(count)+'.nii.gz'))
        if niftiseg is not None:
            patch = get_patch(niftiseg.get_fdata(), patchshape, coor, cval=0)
            nib.save(nib.Nifti1Image(patch.astype(np.int16), niftiimg.affine), os.path.join(savepath, str(count)+'_label.nii.gz'))
        count = count + 1

def ct_preprosess(filename, savepath, patch_size, phase,
                  load_segs = True,
                  voxel_size = None, CT_clip = None, verbose = True):
    with open(filename) as f:
        for line in f:
            if load_segs:
                name, imgpath, labelpath = line.split()[0:3]
                niftiimg = nib.load(imgpath)
                niftiseg = nib.load(labelpath)
            else:
                name, imgpath = line.split()[0:2]
                niftiimg = nib.load(imgpath)

            if verbose:
                print('processing:', name)

            img = niftiimg.get_fdata()
            seg = niftiseg.get_fdata()

            cval = img.min()
            if CT_clip is not None:
                if verbose:
                    print('clipping CT value to', CT_clip)
                img[img > CT_clip[1]] = CT_clip[1]
                img[img < CT_clip[0]] = CT_clip[0]
                niftiimg = nib.Nifti1Image(img.astype(np.int16), niftiimg.affine)
                cval = CT_clip[0]

            """
            # Slow and has some unexpected behavior, instead, apply freesurfer mri_convert before this code
            if voxel_size is not None:
                if verbose:
                    print('resampling to', voxel_size)
                img = resample_to_output(img, voxel_sizes=voxel_size, order=2, cval = cval)
                print(name, img.get_shape())
                if load_segs:
                    seg = resample_to_output(seg, voxel_sizes=voxel_size, order=0)
            """

            ROI = util_bbox(seg)
            if verbose:
                print('saving to patches of', patch_size)
            savename = os.path.join(savepath, name)
            if not os.path.exists(savename):
                os.makedirs(savename)
            extract_patches_nifti(niftiimg, patchshape=patch_size, patchnum=100, savepath=savename, niftiseg=niftiseg, ROI=ROI, ROIratio=0.8, cval=cval)

            if verbose:
                print('finished processing', name)
    if verbose:
        print('preprocessing done!')

def write_list_to_txt(datalist, fname, path_dir = None):
    f = open(fname, 'w')
    for data in datalist:
        if path_dir is not None:
            data = os.path.join(path_dir, data)
        f.write(data+'\n')
    f.close()

def get_data_split(path, val_size=0.15, test_size=0.15):
    subjs = os.listdir(path)
    np.random.shuffle(subjs)

    N = len(subjs)
    slice1 = int(val_size*N)
    slice2 = slice1 + int(test_size*N)

    subjs_val = subjs[0:slice1]
    subjs_test = subjs[slice1:slice2]
    subjs_train = subjs[slice2:]

    write_list_to_txt(subjs_val, os.path.join(path, 'dataset_val.txt'), path_dir=path)
    write_list_to_txt(subjs_test, os.path.join(path, 'dataset_test.txt'), path_dir=path)
    write_list_to_txt(subjs_train, os.path.join(path, 'dataset_train.txt'), path_dir=path)
    write_list_to_txt(subjs, os.path.join(path, 'dataset_all.txt'), path_dir=path)

    print('train:', len(subjs_train))
    print('val:', len(subjs_val))
    print('test:', len(subjs_test))
'''
ct_preprosess(filename = '/home/SENSETIME/shenrui/data/pelvis_resampled(0.8,0.8,0.8)/dataset_all.txt',
              savepath = '/home/SENSETIME/shenrui/data/pelvis_processed',
              load_segs=True, CT_clip=[-1000, 2000], patch_size=(128,128,128), verbose=True)
'''
get_data_split('/home/SENSETIME/shenrui/data/pelvis_processed', val_size=0.15, test_size=0.15)

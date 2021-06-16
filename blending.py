import numpy as np
import cv2
import scipy.sparse.linalg
import scipy.sparse
import matplotlib.pyplot as plt

def create_mask(img_mask, img_target, img_src, offset=(0, 0)):
    '''
    Takes the np.array from the grayscale image
    '''

    # crop img_mask and img_src to fit to the img_target
    hm, wm = img_mask.shape
    ht, wt, _ = img_target.shape

    hd0 = max(0, -offset[0])
    wd0 = max(0, -offset[1])

    hd1 = hm - max(hm + offset[0] - ht, 0)
    wd1 = wm - max(wm + offset[1] - wt, 0)

    mask = np.zeros((hm, wm))
    mask[img_mask > 0] = 1
    mask[img_mask == 0] = 0

    mask = mask[hd0:hd1, wd0:wd1]
    src = img_src[hd0:hd1, wd0:wd1]

    # fix offset
    offset_adj = (max(offset[0], 0), max(offset[1], 0))

    # remove edge from the mask so that we don't have to check the
    # edge condition

    mask[:, -1] = 0
    mask[:, 0] = 0
    mask[-1, :] = 0
    mask[0, :] = 0

    return mask, src, offset_adj

def get_gradient_sum(img, i, j, h, w):
    """
    Return the sum of the gradient of the source imgae.
    * 3D array for RGB
    """

    v_sum = np.array([0.0, 0.0, 0.0])
    v_sum = img[i, j] * 4 - img[i + 1, j] - img[i - 1, j] - img[i, j + 1] - img[i, j - 1]

    return v_sum

def get_mixed_gradient_sum(img_src, img_target, i, j, h, w, ofs, c=1.0):
    """
    Return the sum of the gradient of the source imgae.
    * 3D array for RGB
    c(>=0): larger, the more important the target image gradient is
    """

    v_sum = np.array([0.0, 0.0, 0.0])
    nb = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    for kk in range(4):

        fp = img_src[i, j] - img_src[i + nb[kk, 0], j + nb[kk, 1]]
        gp = img_target[i + ofs[0], j + ofs[1]] - img_target[i + nb[kk, 0] + ofs[0], j + nb[kk, 1] + ofs[1]]

        # if np.linalg.norm(fp) > np.linalg.norm(gp):
        #     v_sum += fp
        # else:
        #     v_sum += gp

        v_sum += np.array([fp[0] if abs(fp[0] * c) > abs(gp[0]) else gp[0],
                           fp[1] if abs(fp[1] * c) > abs(gp[1]) else gp[1],
                           fp[2] if abs(fp[2] * c) > abs(gp[2]) else gp[2]])

    return v_sum


def poisson_blend(img_mask, img_src, img_target, method='mix', c=1.0, offset_adj=(0,0)):
    hm, wm = img_mask.shape
    region_size = hm * wm

    F = np.zeros((region_size, 3))
    A = scipy.sparse.identity(region_size, format='lil')

    get_k = lambda i, j: i + j * hm

    # plane insertion
    if method in ['target', 'src']:
        for i in range(hm):
            for j in range(wm):
                k = get_k(i, j)

                # ignore the edge case (# of neighboor is always 4)
                if img_mask[i, j] == 1:
                    if method == 'target':
                        F[k] = img_target[i + offset_adj[0], j + offset_adj[1]]
                    elif method == 'src':
                        F[k] = img_src[i, j]
                else:
                    F[k] = img_target[i + offset_adj[0], j + offset_adj[1]]

    # poisson blending
    else:
        if method == 'mix':
            grad_func = lambda ii, jj: get_mixed_gradient_sum(img_src, img_target, ii, jj, hm, wm, offset_adj, c=c)
        else:
            grad_func = lambda ii, jj: get_gradient_sum(img_src, ii, jj, hm, wm)

        for i in range(hm):
            for j in range(wm):
                k = get_k(i, j)

                # ignore the edge case (# of neighboor is always 4)
                if img_mask[i, j] == 1:
                    f_star = np.array([0.0, 0.0, 0.0])

                    if img_mask[i - 1, j] == 1:
                        A[k, k - 1] = -1
                    else:
                        f_star += img_target[i - 1 + offset_adj[0], j + offset_adj[1]]

                    if img_mask[i + 1, j] == 1:
                        A[k, k + 1] = -1
                    else:
                        f_star += img_target[i + 1 + offset_adj[0], j + offset_adj[1]]

                    if img_mask[i, j - 1] == 1:
                        A[k, k - hm] = -1
                    else:
                        f_star += img_target[i + offset_adj[0], j - 1 + offset_adj[1]]

                    if img_mask[i, j + 1] == 1:
                        A[k, k + hm] = -1
                    else:
                        f_star += img_target[i + offset_adj[0], j + 1 + offset_adj[1]]

                    A[k, k] = 4
                    F[k] = grad_func(i, j) + f_star

                else:
                    F[k] = img_target[i + offset_adj[0], j + offset_adj[1]]

    A = A.tocsr()

    img_pro = np.empty_like(img_target.astype(np.uint8))
    img_pro[:] = img_target.astype(np.uint8)

    for l in range(3):
        x = scipy.sparse.linalg.spsolve(A, F[:, l])
        x[x > 255] = 255
        x[x < 0] = 0
        x = np.array(x, img_pro.dtype)

        img_pro[offset_adj[0]:offset_adj[0] + hm, offset_adj[1]:offset_adj[1] + wm, l] = x.reshape(hm, wm, order='F')

    return img_pro

def generate_mask(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.getGaussianKernel(9,9)
    blur= cv2.GaussianBlur(gray_image,(5,5),0)

    kernel=np.ones((5,5),np.float32)/25
    averaged= cv2.filter2D(blur,-1,kernel)

    _, thresh = cv2.threshold(averaged,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    mask = np.ones(opening.shape)
    mask = cv2.bitwise_not(opening, mask)

    return mask

def blend(source, target, offset, mask='Total', method = 'normal'):
    if mask == 'Total':
        w, h, _ = source.shape
        mask = np.ones([w, h], dtype=np.float64)
    elif mask == 'Segmented':
        mask = generate_mask(source)

    #Convert all files to np.float64 format
    mask = mask.astype(np.float64)
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    
    _mask, _source, _offset = create_mask(mask, target, source, offset = offset)
    
    return poisson_blend(_mask, _source, target, method = method, offset_adj = _offset)

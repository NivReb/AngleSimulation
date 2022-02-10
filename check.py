import numpy as np
import pydicom as dcm
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Button, RadioButtons, Slider
import os
from PIL import Image, ImageDraw
import pylab as pl
from natsort import natsorted
from skimage import feature, measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2 as cv
import scipy
from scipy import ndimage
from scipy import special
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm
#import pylab as plt
import time
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection

class BodyMask:
    def __init__(self, slice, level):
        self.slicetomask = slice
        self.level = level

    def mask_body(self):
        """
                    Takes user clicked slice.
                    Find body edges.
                    Find body contour.
                    Create mask from body contour.
                    Find the first row at clicked column intersecting mask.
        :return: row click index
        """
        #sliceforcontour = self.rotvol[self.sliceind.astype(int), :, :]
        edges = measure.find_contours(self.slicetomask, level=self.level, fully_connected='low',
                                           positive_orientation='high') # marching square algorithm
        bodycontour = self.find_body(edges)
        body = self.create_mask_from_polygon(self.slicetomask, bodycontour)
        #rowind = np.argwhere(body[:, self.columnind.astype(int)] == 1)[0]
        return body

    def find_body(self, contours):
        """
                    Chooses the contours that correspond to the body
                    First, we exclude non-closed sets-contours
                    Then we assume some min area and volume to exclude small contours

                    Args:
                        contours: all the detected contours

                    Returns: contours that correspond to the body area
                    """
        body_contours = []
        vol_contours = []

        for contour in contours:
            hull = ConvexHull(contour)

            # set some constraints for the volume
            if hull.volume > 20000:
                body_contours.append(contour)
                vol_contours.append(hull.volume)

        # Discard non body contour
        if len(body_contours) == 2:
            return body_contours
        elif len(body_contours) > 2:
            vol_contours, body_contours = (list(t) for t in
                    zip(*sorted(zip(vol_contours, body_contours))))
            body_contours.pop(-1)
        return body_contours  # only body left !!!

    def create_mask_from_polygon(self, image, contours):
        """
                    Creates a binary mask with the dimensions of the image and
                    converts the list of polygon-contours to binary masks and merges them together
                    Args:
                        image: the image that the contours refer to
                        contours: list of contours

                    Returns:

                    """
        body_mask = np.array(Image.new('L', image.shape, 0))
        for contour in contours:
            x = contour[:, 0]
            y = contour[:, 1]
            polygon_tuple = list(zip(x, y))
            img = Image.new('L', image.shape, 0)
            ImageDraw.Draw(img).polygon(polygon_tuple, outline=0, fill=1)
            mask = img
            body_mask += mask

        body_mask[body_mask > 1] = 1  # sanity check to make 100% sure that the mask is binary
        return body_mask.T


lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk('CT scans from Eric'):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName, filename))

# Get ref file

lstFilesDCM = natsorted(lstFilesDCM)  # sorting the files in order
RefDsFirst = dcm.read_file(lstFilesDCM[0]) # refrence file
RefDsLast = dcm.read_file(lstFilesDCM[len(lstFilesDCM)-1])
# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (len(lstFilesDCM), int(RefDsFirst.Rows), int(RefDsFirst.Columns))
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDsFirst.pixel_array.dtype)
ArrayDicomHu = np.zeros(ConstPixelDims, dtype=RefDsFirst.pixel_array.dtype)
ArrayDicomMu = np.zeros(ConstPixelDims, dtype=RefDsFirst.pixel_array.dtype)
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dcm.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[len(lstFilesDCM)-1-lstFilesDCM.index(filenameDCM), :, :] = ds.pixel_array

PatientPosition = RefDsFirst.PatientPosition
if PatientPosition == 'HFS': # HFS - Head First Supine
    ArrayDicomRot = np.rot90(ArrayDicom, k=2, axes=(1, 2))
elif PatientPosition == 'HFP': # HFP - Head First Prone
    ArrayDicomRot = np.rot90(ArrayDicom, k=2, axes=(0, 2))
elif PatientPosition == 'FFP': # FFP - Feet First Prone
    ArrayDicomRot = ArrayDicom
elif PatientPosition == 'FFS': # FFS - Feet First Supine
    ArrayDicomRot = np.rot90(ArrayDicom, k=2, axes=(1, 2))
else:
    ArrayDicomRot = ArrayDicom
    print('Patient Position could not detect')

ArrayDicomHu = np.add(np.multiply(ArrayDicomRot, int(RefDsFirst.RescaleSlope)),
                          int(RefDsFirst.RescaleIntercept))  # Array based on hounsfield Unit
ArrayDicomHu[ArrayDicomHu <= -1000] = -1000


def rot(x, y, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    centerX = RefDsFirst.Columns/2
    centerY = RefDsFirst.Rows/2
    Xcorr = c * x + (-s * y) + x - c * centerX + s * centerY
    Ycorr = s * x + c * y + y - s * centerX - c * centerY
    return Xcorr, Ycorr


def a_multi_matrix():
    a_mat_s = time.time()
    """
    Affin matrix to convert pixal coordinate system to
    patient coordinate system

    """

    orient_pat = np.zeros((3, 2))
    orient_pat[:, 0] = RefDsFirst.ImageOrientationPatient[3:6]  # Image orientation Patient column
    orient_pat[:, 1] = RefDsFirst.ImageOrientationPatient[0:3]  # Image orientation Patient row
    orient_cross = np.cross(orient_pat[:, 0], orient_pat[:, 1])  # cross product of the two vectors above
    missing_r_col = np.zeros((3, 1))
    pos_pat_0 = RefDsFirst.ImagePositionPatient
    pos_pat_N = RefDsLast.ImagePositionPatient
    pixel_spacing = np.zeros((3, 2))
    pixel_spacing[:, :] = RefDsFirst.PixelSpacing
    NZ = len(lstFilesDCM)
    R3 = orient_pat * np.diag(pixel_spacing)
    R = np.zeros((4, 2))
    R[:3, :] = R3
    multi_aff = np.eye(4)
    multi_aff[:3, :2] = R3
    #trans_z_N = Matrix((0, 0, NZ - 1, 1))
    #multi_aff[:3, 2] = missing_r_col
    multi_aff[:3, 3] = pos_pat_0
    multi_aff[:3, 2] = np.subtract(pos_pat_N, pos_pat_0) / (NZ - 1)
    a_mat_e = time.time()
    #print('Affine Matrix calculation time: %s' % (a_mat_e-a_mat_s))
    return multi_aff


def voxal_to_patient(A, voxVec):
    voxalLocation = np.matmul(A, voxVec)
    return voxalLocation


def volume_rotation(vol, angle):
    rot_s = time.time()
    rotatedVol = ndimage.rotate(vol, float(angle), axes=(1, 2), reshape=False, mode='constant', cval=-1000)
    rot_e = time.time()
    #print('volume rotation time: %s' % (rot_e-rot_s))
    return rotatedVol



def find_body(contours):
    conto_s = time.time()
    """
    Chooses the contours that correspond to the lungs and the body
    First, we exclude non-closed sets-contours
    Then we assume some min area and volume to exclude small contours
    Then the body is excluded as the highest volume closed set
    The remaining areas correspond to the lungs
    Args:
        contours: all the detected contours

    Returns: contours that correspond to the lung area
    """
    body_contours = []
    vol_contours = []

    for contour in contours:
        hull = ConvexHull(contour)

       # set some constraints for the volume
        if hull.volume > 20000:
            body_contours.append(contour)
            vol_contours.append(hull.volume)

    # Discard body contour
    if len(body_contours) == 2:
        return body_contours
    elif len(body_contours) > 2:
        vol_contours, body_contours = (list(t) for t in
                zip(*sorted(zip(vol_contours, body_contours))))
        body_contours.pop(-1) # body is out!
    conto_e = time.time()
    #print('Find body countor time: %s' % (conto_e-conto_s))
    return body_contours # only lungs left !!!


def create_mask_from_polygon(image, contours):
    cont_mask_s = time.time()
    """
    Creates a binary mask with the dimensions of the image and
    converts the list of polygon-contours to binary masks and merges them together
    Args:
        image: the image that the contours refer to
        contours: list of contours

    Returns:

    """
    body_mask = np.array(Image.new('L', image.shape, 0))
    for contour in contours:
        x = contour[:, 0]
        y = contour[:, 1]
        polygon_tuple = list(zip(x, y))
        img = Image.new('L', image.shape, 0)
        ImageDraw.Draw(img).polygon(polygon_tuple, outline=0, fill=1)
        mask = np.array(img)
        body_mask += mask

    body_mask[body_mask > 1] = 1  # sanity check to make 100% sure that the mask is binary
    cont_mask_e = time.time()
    #print('countor mask cal time: %s' % (cont_mask_e-cont_mask_s))
    return body_mask.T


#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1, projection='3d')


def cone(p0, p1, R0, R1):
    cone_s = time.time()
    """
    Based on https://stackoverflow.com/a/39823124/190597 (astrokeat)
    """
    # vector in direction of axis
    v = p1 - p0
    # find magnitude of vector
    mag = norm(v)
    # unit vector in direction of axis
    v = v / mag
    # make some vector not in the same direction as v
    not_v = np.array([1, 1, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    # make vector perpendicular to v
    n1 = np.cross(v, not_v)
    # print n1,'\t',norm(n1)
    # normalize n1
    n1 /= norm(n1)
    # make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    n = 300
    t = np.linspace(0, mag, n)
    theta = np.linspace(0, 2 * np.pi, n)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    R = np.linspace(R0, R1, n)
    # generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + R *
               np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    cone_e = time.time()
    #ax.plot_surface(X, Y, Z, color=color, linewidth=0, antialiased=False)
    invAffinMatrix = np.linalg.inv(AffinMatrix)
    X_f = X.flatten
    ones = np.full(n*n, 1)
    coorArray = np.zeros((len(ones), 4))
    coorArray[:, 0] = X.flatten()
    coorArray[:, 1] = Y.flatten()
    coorArray[:, 2] = Z.flatten()
    coorArray[:, 3] = ones
    pixcoorArray = np.zeros(coorArray.shape)
    #for k in range(0, len(ones)):
    #    pixcoorArray[k, :] = np.round(np.matmul(invAffinMatrix, coorArray[k, :]))
    pixcoorArray = np.round(coorArray.dot(invAffinMatrix.T)[:,:-1])
    pixcoorArray[:, 2][pixcoorArray[:, 2] >= len(lstFilesDCM)] = len(lstFilesDCM) - 1
    '''
    slice_cone = np.where(pixcoorArray[:, 2] == clickedsliceind)
    x_cone = pixcoorArray[slice_cone, 0]
    y_cone = pixcoorArray[slice_cone, 1]
    plt.plot(x_cone, y_cone, color='green', marker=".")
    plt.show()
    '''
    cone_d = time.time()
    #print('cone corr place cal time: %s' % (cone_e-cone_s))
    #print('cone corr to voxal', cone_d-cone_e)
    return pixcoorArray#, coorArray


def voxal_ray_distance(ArrayShape, angle, RefDs):
    #calculation the distance each ray passing through each voxal
    PixelSpacingRow = int(RefDs.PixelSpacing[0])/10
    DistArray = np.full(ArrayShape, PixelSpacingRow/np.cos(angle*np.pi/180))
    return DistArray


def ref_point_for_cone_rotation(slice, columnind, sliceind, distolow, distocenter,distoupper, distbasetoskin, affine, rowspacing):
    s = time.time()
    edges = measure.find_contours(slice, level=-250, fully_connected='low', positive_orientation='high')
    bodycontour = find_body(edges)
    body = create_mask_from_polygon(slice, bodycontour)
    rowind = np.argwhere(body[:, columnind] == 1)[0]

    # pixal world to patient coordinate system
    centerbasepoint = voxal_to_patient(affine, np.array(
        (int(np.round(rowind - distbasetoskin / rowspacing)), columnind, sliceind, 1)))[0:3]
    upperapexpoint = voxal_to_patient(affine, np.array(
        (int(np.round(rowind + distoupper / rowspacing)), columnind, sliceind, 1)))[0:3]
    centeralapexpoint = voxal_to_patient(affine, np.array(
        (int(np.round(rowind + distocenter / rowspacing)), columnind, sliceind, 1)))[0:3]
    lowerapexpoint = voxal_to_patient(affine, np.array(
        (int(np.round(rowind + distolow / rowspacing)), columnind, sliceind, 1)))[0:3]

    f = time.time()
    #print('cone based image time', f-s)
    return centerbasepoint, centeralapexpoint


def volume_for_cone(vol, sliceind, R, angle):
    if sliceind + R > len(lstFilesDCM):
        return volume_rotation(vol[sliceind - R : len(lstFilesDCM), :, :], angle), sliceind-R
    elif sliceind - R < 0:
        return volume_rotation(vol[0:sliceind+R, :, :], angle), 0
    else:
        return volume_rotation(vol[sliceind-R:sliceind+R, :, :], angle), sliceind-R

# find the skin
def find_high_Patient_volume_slice(vol):
    s = time.time()
    slice_vol = []
    for i in range(0, np.shape(vol)[0]-1):
        edges = measure.find_contours(vol[i, :, :], level=-250, fully_connected='low',
                                      positive_orientation='high')
        bodycontour = find_body(edges)
        body = create_mask_from_polygon(vol[i, :, :], bodycontour)
        slice_vol.append(np.count_nonzero(body == 1))
    maxvol = max(slice_vol)
    f = time.time()
    #print('time to find largest slice in volume', f-s)
    return slice_vol.index(maxvol)


def row_and_col_indices_for_volume_cutting(vol, largest_slice_ind, row_addition, column_addition):
    s = time.time()
    # Create a boolian mask from the largest body volume slice
    body = BodyMask(vol[largest_slice_ind, :, :], -250)
    bodymask = body.mask_body()
    # Within that slice find the rows and columns that delimit the patient body
    first_row_volume_encounter = np.min([i for i in first_nonzero(bodymask, 0) if i > 0])
    first_column_volume_encounter = np.min([i for i in first_nonzero(bodymask, 1) if i > 0])
    last_row_volume_encounter = np.max(last_nonzero(bodymask, 0))
    last_column_volume_encounter = np.max(last_nonzero(bodymask, 1))
    # Slice the volume to the exact size that delimit the patient body
    chopedvol = vol[:, first_row_volume_encounter:last_row_volume_encounter, first_column_volume_encounter:last_column_volume_encounter]
    # Create larger space to accommodate sliced volume and integrate the patient body inside it.
    s, r, c = chopedvol.shape # indexes 0:r-1, 0:c-1
    data_vol = np.full((s, r+row_addition, c+column_addition), -1000)
    data_vol[:, int(row_addition*3/4-1):int(row_addition*3/4-1)+r, int(column_addition/2-1):int(column_addition/2-1)+c] = chopedvol
    f = time.time()
   # print('volume chopping time', f-s)
    return data_vol, first_row_volume_encounter, first_column_volume_encounter


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val,  invalid_val)


def update_slider_axial_rot(val):
    """
                Takes slider val and update CT image based on the val
    :param val: CT slice
    :return:
    """
    rotvolimage.set_data(data_vol_rot[val, :, :].squeeze())
    fig.canvas.draw_idle()


def update_slider_axial(val):
    """
                Takes slider val and update CT image based on the val
    :param val: CT slice
    :return:
    """
    notrotvolimage.set_data(chopedvol[val, :, :].squeeze())
    print(val)
    fig.canvas.draw_idle()


def cone_integrate(data_vol_rot, chopedvol, rowind, clickpointcolumn, clickedsliceind, R):
    centerbasepoint = voxal_to_patient(AffinMatrix, np.array(
        (int(np.round(rowind - distfromskin / RefDsFirst.PixelSpacing[0])), clickpointcolumn, clickedsliceind, 1)))[0:3]
    upperapexpoint = voxal_to_patient(AffinMatrix, np.array(
        (
        int(np.round(rowind + skintouppertarget / RefDsFirst.PixelSpacing[0])), clickpointcolumn, clickedsliceind, 1)))[
                     0:3]
    centeralapexpoint = voxal_to_patient(AffinMatrix, np.array(
        (int(np.round(rowind + skintotargetcenter / RefDsFirst.PixelSpacing[0])), clickpointcolumn, clickedsliceind,
         1)))[0:3]
    lowerapexpoint = voxal_to_patient(AffinMatrix, np.array(
        (
        int(np.round(rowind + skintolowertarget / RefDsFirst.PixelSpacing[0])), clickpointcolumn, clickedsliceind, 1)))[
                     0:3]

    uppercone = cone(upperapexpoint, centerbasepoint, 0, R)
    centercone = cone(centeralapexpoint, centerbasepoint, 0, R)
    lowercone = cone(lowerapexpoint, centerbasepoint, 0, R)
    # uppercone[:, 2] = uppercone[:, 2] - slicecorrectionfactor
    # centercone[:, 2] = centercone[:, 2] - slicecorrectionfactor
    # lowercone[:, 2] = lowercone[:, 2] - slicecorrectionfactor
    boolmat = np.zeros(chopedvol.shape)
    boolmat[uppercone[:, 2].astype(int), uppercone[:, 0].astype(int), uppercone[:, 1].astype(int)] = 1
    boolmat[centercone[:, 2].astype(int), centercone[:, 0].astype(int), centercone[:, 1].astype(int)] = 1
    boolmat[lowercone[:, 2].astype(int), lowercone[:, 0].astype(int), lowercone[:, 1].astype(int)] = 1
    boolmat.astype(bool)
    boolmatbool = boolmat.astype(bool)
    j = time.time()
    rotboolmat = ndimage.rotate(boolmatbool, float(-anglewillclick), axes=(1, 2), reshape=False)
    d = time.time()
    rotconemat = rotboolmat.astype(int)
    rotconemat = rotconemat * 6000
    chopedvol = chopedvol + rotconemat

    print("bool matrix rotation time", d-j)
    data_vol_rot[uppercone[:, 2].astype(int), uppercone[:, 0].astype(int), uppercone[:, 1].astype(int)] = 6000
    data_vol_rot[centercone[:, 2].astype(int), centercone[:, 0].astype(int), centercone[:, 1].astype(int)] = 6000
    data_vol_rot[lowercone[:, 2].astype(int), lowercone[:, 0].astype(int), lowercone[:, 1].astype(int)] = 6000

    return data_vol_rot, chopedvol


def clicked_points_index(clickedimagesize, angle, clickpointcolumn, rowind, offset):
    '''
    :param clickedimagesize: image size need for image centering befor rotation
    :param angle: angle at which  point need to rerotate
    :param clickpointcolumn: column pixel index at which user clicked at rotated image
    :param rowind: row pixel index at which user clicked at rotated image
    :return: pixel coordinate at not rotated image
    '''
    rotmat = ([np.cos(-angle*np.pi/180), -np.sin(-angle*np.pi/180), 0],
              [np.sin(-angle*np.pi/180), np.cos(-angle*np.pi/180), 0],
              [0, 0, 1])
    movetocentermat = ([1, 0, 0],
                       [0, 1, 0],
                       [-((clickedimagesize[1]) / 2), -((clickedimagesize[0]) / 2), 1])
    offsetmattocenter = ([1, 0, 0],
                         [0, 1, 0],
                         [-offset[0], -offset[1], 1])
    offsetmatfromcenter = ([1, 0, 0],
                         [0, 1, 0],
                         [offset[0], offset[1], 1])
    backtocornermat = ([1, 0, 0],
                       [0, 1, 0],
                       [(clickedimagesize[1]) / 2, (clickedimagesize[0]) / 2, 1])
    transmat = np.linalg.multi_dot([backtocornermat, rotmat, movetocentermat])
    invtransmat = np.linalg.inv(transmat)
    rotindvector = np.array([int(clickpointcolumn), int(rowind), 1])
    trial = np.matmul(rotmat, rotindvector)
    point = trial[0:1] + offset
    #point = np.matmul(transmat, rotindvector)
    return np.matmul(invtransmat, rotindvector)

def find_cone_crit_indices(imageshape, rowind, columnind):
    r = time.time()
    simarray = np.zeros(imageshape)
    conebasecol = int(np.ndarray.item(rowind) - distfromskin)
    coneappupcol = int(np.ndarray.item(rowind) + skintouppertarget)
    coneappcentercol = int(np.ndarray.item(rowind) + skintotargetcenter)
    coneapplowcol = int(np.ndarray.item(rowind) + skintolowertarget)
    simarray[conebasecol, columnind] = 1
    simarray[coneappupcol, columnind] = 1
    simarray[coneappcentercol, columnind] = 1
    simarray[coneapplowcol, columnind] = 1
    rotmat = ndimage.rotate(simarray, float(-anglewillclick), reshape=False, mode='constant', cval=0)

    # Create a binary image after rotation with threshold of 0.1.
    binary_image = cv.threshold(rotmat, 0.1, 1, cv.THRESH_BINARY)[1]
    # Finding clusters and assigning each cluster a number.
    num_labels, labels_im = cv.connectedComponents(binary_image.astype(np.uint8) * 255)

    idxs = []
    # find the max value in each cluster and returning its index
    for label in range(1, num_labels):
        mask = np.zeros(rotmat.shape)
        mask[labels_im == label] = 1
        idxs.append(np.unravel_index((rotmat * mask).argmax(), rotmat.shape))

    y = time.time()
    #print("cone index search time", y-r)
    return idxs
# not used in the current settings


def show_slice_window(slice, level, window):
   """
   Function to display an image slice
   Input is a numpy 2D array
   """
   slice_try = slice[slice > 300]
   max = level + window/2
   min = level - window/2
   slice = slice.clip(min, max)
   #plt.figure()
   #plt.imshow(slice, cmap="gray", origin="lower")
   #plt.savefig('L'+str(level)+'W'+str(window))
   return slice


def intensity_seg(slice, level, window):
   clipped = show_slice_window(slice, level, window)
   return measure.find_contours(clipped, level=300), clipped

def display(img, contours):
    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(img,'gray',vmin=0,vmax=255)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    plt.show()


def bone_binary_mask(slice, thresh):
    """
    finds the bone material in a 2d image. plot the image to the screen.
    :param slice: CT scan slice for bone detection.
    :return: binary mask with areas where bone is in the image.
    """
    #bone masking
    gray_scale_slice = gray_scale(slice)
    ret2, thresh2 = cv.threshold(gray_scale_slice, thresh, 255, cv.THRESH_BINARY_INV)
    if thresh == 127:
        thresh2_bool = thresh2 == 0
        thresh2_binary = thresh2_bool.astype(int)
        mask = ndimage.binary_fill_holes(thresh2_binary).astype(int)
        return mask
    return thresh2


def gray_scale(image):
    """
    Convert any image to gray scale image.
    For this function to work as wanted we need to use the raw ct data before converting to HU.
    :param image: image
    :return: gray scale image
    """
    # Step 1. Convert to float to avoid overflow or underflow losses.
    img_2d = image.astype(float)

    # Step 2. Rescaling grey scale between 0-255
    img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0

    # Step 3. Convert to uint
    img_2d_scaled = np.uint8(img_2d_scaled)

    return img_2d_scaled


def threshold_option_show(gray_scale_img):
    """
    plots all optin for image thresholding methods with cv library.
    :param gray_scale_img: gray scale image
    :return: plot of threshold images
    """
    ret1,thresh1 = cv.threshold(gray_scale_img,50,255,cv.THRESH_BINARY)
    ret2,thresh2 = cv.threshold(gray_scale_img,50,255,cv.THRESH_BINARY_INV)
    ret3,thresh3 = cv.threshold(gray_scale_img,50,255,cv.THRESH_TRUNC)
    ret4,thresh4 = cv.threshold(gray_scale_img,50,255,cv.THRESH_TOZERO)
    ret5,thresh5 = cv.threshold(gray_scale_img,50,255,cv.THRESH_TOZERO_INV)
    thresh2_bool = thresh2 == 0
    thresh2_binary = thresh2_bool.astype(int)
    filled_hols = ndimage.binary_fill_holes(thresh2_binary).astype(int)
    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [filled_hols, thresh1, thresh2, thresh3, thresh4, thresh5]
    plt.imshow(filled_hols)

    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()

    return thresh1, thresh2, thresh3, thresh4, thresh5


def boby_segments_cone_interaction(mask, slice, cone_vol):
    cone_hitting_spots = mask * cone_vol[slice, :, :]
    return cone_hitting_spots


# clicked point parameters for cone calculaiton
R = 44 #mm
distfromskin = 13 #mm
skintouppertarget = 47 #mm
skintotargetcenter = 70.5 #mm
skintolowertarget = 94 #mm
clickedsliceind = 60
clickpointcolumn = 239
anglewillclick = 25
# cv2 thresh value for cv2.threshold method
bone_thresh = 127
skin_thresh = 50
bone_thresh_type = cv.THRESH_BINARY_INV
skin_thresh_type = cv.THRESH_BINARY



# Affine Matrix
AffinMatrix = a_multi_matrix()



row_vol_increase = 100
column_vol_increase = 50

largest_slice = find_high_Patient_volume_slice(ArrayDicomHu)

chopedvol, row_movment, column_movment = row_and_col_indices_for_volume_cutting(ArrayDicomHu, largest_slice, row_vol_increase, column_vol_increase) # returnes cut volume


# calculation for rotated volume with cone
data_vol_rot = volume_rotation(chopedvol, anglewillclick)

edges = measure.find_contours(data_vol_rot[clickedsliceind, :, :], level=-250, fully_connected='low', positive_orientation='high')
bodycontour = find_body(edges)
body = create_mask_from_polygon(data_vol_rot[clickedsliceind, :, :], bodycontour)

clickedslice = data_vol_rot[clickedsliceind, :, :]
rowind = np.argwhere(body[:, clickpointcolumn] == 1)[0]
rotconeind = find_cone_crit_indices(clickedslice.shape, rowind, clickpointcolumn)

centerbasepoint = voxal_to_patient(AffinMatrix, np.array((rotconeind[0][0], rotconeind[0][1], clickedsliceind, 1)))[0:3]
upperapexpoint = voxal_to_patient(AffinMatrix, np.array((rotconeind[1][0], rotconeind[1][1], clickedsliceind, 1)))[0:3]
centeralapexpoint = voxal_to_patient(AffinMatrix, np.array((rotconeind[2][0], rotconeind[2][1], clickedsliceind, 1)))[0:3]
lowerapexpoint = voxal_to_patient(AffinMatrix, np.array((rotconeind[3][0], rotconeind[3][1], clickedsliceind, 1)))[0:3]

# cone integration in images without rotation
uppercone = cone(upperapexpoint, centerbasepoint, 0, R)
centercone = cone(centeralapexpoint, centerbasepoint, 0, R)
lowercone = cone(lowerapexpoint, centerbasepoint, 0, R)

cone_vol = np.zeros(chopedvol.shape)

cone_vol[uppercone[:, 2].astype(int), uppercone[:, 0].astype(int), uppercone[:, 1].astype(int)] = 1
cone_vol[centercone[:, 2].astype(int), centercone[:, 0].astype(int), centercone[:, 1].astype(int)] = 1
cone_vol[lowercone[:, 2].astype(int), lowercone[:, 0].astype(int), lowercone[:, 1].astype(int)] = 1

# Find cone top spikes at each slice
left_cone_column_peak = first_nonzero(cone_vol[clickedsliceind, :, :], 1, invalid_val=-1)
left_side_cone_column = np.min(left_cone_column_peak[left_cone_column_peak > 0])
right_side_cone_column = np.max(last_nonzero(cone_vol[clickedsliceind, :, :], 1, invalid_val=-1))

left_cone_row = np.where(cone_vol[clickedsliceind, :, left_side_cone_column] == 1)[0][0]
right_cone_row = np.where(cone_vol[clickedsliceind, :, right_side_cone_column] == 1)[0][0]




chopedvol = np.divide(chopedvol - int(RefDsFirst.RescaleIntercept), int(RefDsFirst.RescaleSlope))
#img = gray_scale(chopedvol[clickedsliceind, :, :])
#threshold_option_show(img)

#img = gray_scale(chopedvol[clickedsliceind, :, :])
#threshold_option_show(img)
s = time.time()
bone_mask = bone_binary_mask(chopedvol[clickedsliceind, :, :], bone_thresh)
skin_mask = bone_binary_mask(chopedvol[clickedsliceind, :, :], skin_thresh)

bone_cone = boby_segments_cone_interaction(bone_mask, clickedsliceind, cone_vol)
skin_cone = boby_segments_cone_interaction(skin_mask, clickedsliceind, cone_vol)

if skin_cone[left_side_cone_column, left_cone_row] == 0:
    cone_edge_left = plt.Circle((left_side_cone_column, left_cone_row), 10, color='r', fill=False)
if skin_cone[right_side_cone_column, right_cone_row] == 0:
    cone_edge_right = plt.Circle((right_side_cone_column, right_cone_row), 10, color='r', fill=False)

e = time.time()
chopedvol[uppercone[:, 2].astype(int), uppercone[:, 0].astype(int), uppercone[:, 1].astype(int)] = 6000
chopedvol[centercone[:, 2].astype(int), centercone[:, 0].astype(int), centercone[:, 1].astype(int)] = 6000
chopedvol[lowercone[:, 2].astype(int), lowercone[:, 0].astype(int), lowercone[:, 1].astype(int)] = 6000
print('masking time', e-s)


'''
num_labels, labels_im = cv.connectedComponents(skin_cone.astype(np.uint8) * 255)
print(num_labels)
#if num_labels < 3:
cone_left_edges = np.where(labels_im == 1)
cone_edge_left = plt.Circle((cone_left_edges[1][0], cone_left_edges[0][0]), 10, color='r', fill=False)
cone_right_edges = np.where(labels_im == 2)
cone_edge_right = plt.Circle((cone_right_edges[1][0], cone_right_edges[0][0]), 10, color='r', fill=False)
'''
'''
titles = ['Bone-Cone interaction','Bone Mask','CT Image with cone','Skin-Cone interaction','Skin Mask','Cone']
images = [bone_cone, bone_mask, chopedvol[clickedsliceind, :, :], skin_cone, skin_mask, cone_vol[clickedsliceind, :, :]]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
'''
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
bone_cone_plt = ax1.imshow(bone_cone, cmap='gray')
ax1.set_title('Bone-Cone interaction')
bone = ax2.imshow(bone_mask, cmap='gray')
ax2.set_title('Bone Mask')
ct_cone_plt = ax3.imshow(chopedvol[clickedsliceind, :, :], cmap='gray')
ax3.set_title('CT Image with cone')
ax3.add_patch(cone_edge_left)
ax3.add_patch(cone_edge_right)
skin_cone_plt = ax4.imshow(skin_cone, cmap='gray')
ax4.set_title('Skin-Cone interaction')
skin = ax5.imshow(skin_mask, cmap='gray')
ax5.set_title('Skin Mask')
cone_slice = ax6.imshow(cone_vol[clickedsliceind, :, :], cmap='gray')
ax6.set_title('Cone')
plt.show()


'''
# cone integration for rotated images
rotated_data_vol_with_cone, data_vol_with_cone = cone_integrate(data_vol_rot, chopedvol, rowind, clickpointcolumn, clickedsliceind, R)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

#Axial Slider
posAxialrot = ax1.get_position()
axAxial_slide_rot = plt.axes([posAxialrot.x0, posAxialrot.y0-0.1, 0.2, 0.01])  # Slider position
AxialSlider_rot = Slider(axAxial_slide_rot, '', valmin=0, valmax=rotated_data_vol_with_cone.shape[0]-1, valstep=1, valinit=rotated_data_vol_with_cone.shape[0]/2,
                                    orientation='horizontal')  # slider range
        #ax_slide.set_xticks(np.arange(self.sliceind))
AxialSlider_rot.valtext.set_visible(True)
AxialSlider_rot.label.set_color('white')
AxialSlider_rot.valtext.set_color('white')
AxialSlider_rot.on_changed(update_slider_axial_rot)

posAxial = ax3.get_position()
axAxial_slide = plt.axes([posAxial.x0, posAxial.y0-0.1, 0.2, 0.01])  # Slider position
AxialSlider = Slider(axAxial_slide, '', valmin=0, valmax=data_vol_with_cone.shape[0]-1, valstep=1, valinit=data_vol_with_cone.shape[0]/2,
                                    orientation='horizontal')  # slider range
        #ax_slide.set_xticks(np.arange(self.sliceind))
AxialSlider.valtext.set_visible(True)
AxialSlider.label.set_color('white')
AxialSlider.valtext.set_color('white')
AxialSlider.on_changed(update_slider_axial)


rotvolimage = ax1.imshow(rotated_data_vol_with_cone[clickedsliceind, :, :], cmap='gray')
correctnot_rot_vol = ax2.imshow(ndimage.rotate(rotated_data_vol_with_cone[clickedsliceind, :, :], float(-anglewillclick), reshape=False, mode='constant', cval=-1000), cmap='gray')
'''
fig, (ax1, ax2) = plt.subplots(1, 2)

notrotvolimage = ax1.imshow(chopedvol[clickedsliceind, :, :], cmap='gray')
posAxial = ax1.get_position()
axAxial_slide = plt.axes([posAxial.x0, posAxial.y0-0.1, 0.2, 0.01])  # Slider position
AxialSlider = Slider(axAxial_slide, '', valmin=0, valmax=chopedvol.shape[0]-1, valstep=1, valinit=chopedvol.shape[0]/2,
                                    orientation='horizontal')  # slider range
AxialSlider.valtext.set_visible(True)
#AxialSlider.label.set_color('black')
#AxialSlider.valtext.set_color('black')
AxialSlider.on_changed(update_slider_axial)
body_contour = BodyMask(chopedvol[clickedsliceind, :, :], 300)
body = body_contour.mask_body()
contour = ax2.imshow(body, cmap='gray')
plt.show()





#clickedslice = data_vol_rot[clickedsliceind, :, :]
#clickedcolumn = clickedslice[:, clickpointcolumn]
#rowind = np.where(clickedcolumn > -250)[0][0]

#column_for_cone_not_rot = clickpointcolumn - column_vol_increase/2 + column_movment
#row_for_cone_not_rot = rowind - row_vol_increase/2  #row_movment

# calculation of cone points in base image without rotation. input are the points needed for cone calculation at rotated image.
'''
#off set calculation
input_arr = np.asarray(data_vol_rot)
ndim = input_arr.ndim
print(ndim)
axes = (1, 0)

if ndim < 2:
    raise ValueError('input array should be at least 2D')

axes = list(axes)

if len(axes) != 2:
    raise ValueError('axes should contain exactly two values')

if not all([float(ax).is_integer() for ax in axes]):
    raise ValueError('axes should contain only integer values')

if axes[0] < 0:
    axes[0] += ndim
if axes[1] < 0:
    axes[1] += ndim
if axes[0] < 0 or axes[1] < 0 or axes[0] >= ndim or axes[1] >= ndim:
    raise ValueError('invalid rotation plane specified')

c, s = special.cosdg(anglewillclick), special.sindg(anglewillclick)

rot_matrix = np.array([[c, s],
                        [-s, c]])

img_shape = np.asarray(input_arr.shape)

axes.sort()
in_plane_shape = img_shape[axes]
out_plane_shape = img_shape[axes]
out_center = rot_matrix @ ((out_plane_shape - 1) / 2)
in_center = (in_plane_shape - 1) / 2
offset = in_center - out_center
print(offset)
'''

slice_for_cone_rotation = data_vol_rot[clickedsliceind, :, :]

'''
centerbase_notrotpoint = clicked_points_index(slice_for_cone_rotation.shape, anglewillclick, clickpointcolumn, rowind - distfromskin / RefDsFirst.PixelSpacing[0], offset)
upperapex_notrotpoint = clicked_points_index(slice_for_cone_rotation.shape, anglewillclick, clickpointcolumn, rowind + skintouppertarget / RefDsFirst.PixelSpacing[0], offset)
centerapex_notrotpoint = clicked_points_index(slice_for_cone_rotation.shape, anglewillclick, clickpointcolumn, rowind + skintotargetcenter / RefDsFirst.PixelSpacing[0], offset)
lowerapex_notrotpoint = clicked_points_index(slice_for_cone_rotation.shape, anglewillclick, clickpointcolumn, rowind + skintolowertarget / RefDsFirst.PixelSpacing[0], offset)

centerbase_notrotpoint = clicked_points_index(slice_for_cone_rotation.shape, anglewillclick, clickpointcolumn + offset[1], rowind - distfromskin / RefDsFirst.PixelSpacing[0] + offset[0], offset)
upperapex_notrotpoint = clicked_points_index(slice_for_cone_rotation.shape, anglewillclick, clickpointcolumn + offset[1], rowind + skintouppertarget / RefDsFirst.PixelSpacing[0] + offset[0], offset)
centerapex_notrotpoint = clicked_points_index(slice_for_cone_rotation.shape, anglewillclick, clickpointcolumn + offset[1], rowind + skintotargetcenter / RefDsFirst.PixelSpacing[0] + offset[0], offset)
lowerapex_notrotpoint = clicked_points_index(slice_for_cone_rotation.shape, anglewillclick, clickpointcolumn + offset[1], rowind + skintolowertarget / RefDsFirst.PixelSpacing[0] + offset[0], offset)


# Find cone base points in patient coordinate system
centerbasepoint = voxal_to_patient(AffinMatrix, np.array(
        (int(round(centerbase_notrotpoint[1])), round(centerbase_notrotpoint[0]), clickedsliceind, 1)))[0:3]
upperapexpoint = voxal_to_patient(AffinMatrix, np.array(
        (int(round(upperapex_notrotpoint[1])), round(upperapex_notrotpoint[0]), clickedsliceind, 1)))[0:3]
centeralapexpoint = voxal_to_patient(AffinMatrix, np.array(
        (int(round(centerapex_notrotpoint[1])), round(centerapex_notrotpoint[0]), clickedsliceind, 1)))[0:3]
lowerapexpoint = voxal_to_patient(AffinMatrix, np.array(
        (int(round(lowerapex_notrotpoint[1])), round(lowerapex_notrotpoint[0]), clickedsliceind, 1)))[0:3]
'''


'''
boolmat = np.zeros(chopedvol.shape)
boolmat[uppercone[:, 2].astype(int), uppercone[:, 0].astype(int), uppercone[:, 1].astype(int)] = 1
boolmat[centercone[:, 2].astype(int), centercone[:, 0].astype(int), centercone[:, 1].astype(int)] = 1
boolmat[lowercone[:, 2].astype(int), lowercone[:, 0].astype(int), lowercone[:, 1].astype(int)] = 1
boolmat.astype(bool)
boolmatbool = boolmat.astype(bool)
rotboolmat = ndimage.rotate(boolmatbool, anglewillclick, axes=(1, 2), reshape=False)
rotconemat = rotboolmat.astype(int)
rotconemat = rotconemat * 6000
chopedvol = chopedvol + rotconemat
'''




'''
rotslice = ndimage.rotate(ArrayDicomHu[clickedsliceind, :, :], float(anglewillclick), reshape=False, mode='constant', cval=-1000) #rotate volume to the angle user chose
index_new_vol_clicked_slice = int((2*R+10)/2)
rowpixelspacing = RefDsFirst.PixelSpacing[0]
#refrence points for cone calculation
#coneimagebase_b, coneimagebase_c = ref_point_for_cone_rotation(ArrayDicomHu[clickedsliceind, :, :], clickpointcolumn, clickedsliceind, skintolowertarget, skintotargetcenter, skintouppertarget, distfromskin, AffinMatrix, rowpixelspacing)
conerotimagebase_b, conerotimagebase_c = ref_point_for_cone_rotation(rotslice, clickpointcolumn, clickedsliceind, skintolowertarget, skintotargetcenter, skintouppertarget, distfromskin, AffinMatrix, rowpixelspacing)
rotslice[conerotimagebase_b, clickpointcolumn] = 6000
rotslice[conerotimagebase_c, clickpointcolumn] = 6000
rotslicenorm = ndimage.rotate(rotslice, float(-anglewillclick), reshape=False, mode='constant', cval=-1000)
conecoorunrotimage = np.where(rotslicenorm > 3000)
conecoor[:,0] = conecoorunrotimage[0]
baserow = min(conecoorunrotimage[0])
centapex = max(conecoorunrotimage[0])

#[row, column, slice, 1]
voxal_to_patient(AffinMatrix, )

#calculate points for cone position after rotation
#centerbasepoint[1], centerbasepoint[0] = rot(centerbasepoint[1], centerbasepoint[0], angle=-10)
#centeralapexpoint[1], centeralapexpoint[0] = rot(centeralapexpoint[1], centeralapexpoint[0], angle=-10)

#centercone[:, 3] = 1
#uppercone = cone(upperapexpoint, centerbasepoint, 0, R, 'red')
#lowercne = cone(lowerapexpoint, centerbasepoint, 0, R, 'green')

#cone coordinates
centercone, conecoordinates = cone(coneimagebase_c, coneimagebase_b, 0, R)
centerrotcone, rotconecoordinates = cone(conerotimagebase_c, conerotimagebase_b, 0, R)
#uppercone = cone(upperapexpoint, centerbasepoint, 0, R)
#lowercone = cone(lowerapexpoint,centerbasepoint, 0, R)

conevol = np.full((ArrayDicomHu.shape), 0)
cone_imp_s = time.time()
ArrayDicomHu[centercone[:, 2].astype(int), centercone[:, 0].astype(int), centercone[:, 1].astype(int)] = 6000
conevol[centerrotcone[:, 2].astype(int), centerrotcone[:, 0].astype(int), centerrotcone[:, 1].astype(int)] = 6000
rotcone = ndimage.rotate(conevol[clickedsliceind, :, :], float(anglewillclick), reshape=False, mode='constant', cval=0)
ArrayDicomHu[clickedsliceind, :, :] = ArrayDicomHu[clickedsliceind, :, :] + rotcone


#rotcone = volume_rotation(conevol, -anglewillclick)
#CTwithcone = ArrayDicomHu + rotcone
'''


#cone_imp_e = time.time()
#print('cone impementation time: %s' % (cone_imp_e-cone_imp_s))
#fig = plt.figure()
#fig.patch.set_facecolor('black')
#gs = GridSpec(1, 3, figure=fig)
#ax1 = fig.add_subplot(gs[0, 0])
#ax2 = fig.add_subplot(gs[0, 1])
#ax3 = fig.add_subplot(gs[0, 2])
#ax1.add_wedge(wedge1)
#ax1.add_wedge(wedge2)
#ax1.imshow(rotvol[index_new_vol_clicked_slice, :, :].squeeze(), cmap='gray')
#ax2.imshow(np.rot90(rotvol[:, :, index_new_vol_clicked_slice].squeeze(), k=3), cmap='gray') # 150,512
#ax3.imshow(rotvol[:, int(np.round(rowind+skintotargetcenter/rowpixelspacing)), :].squeeze(), cmap='gray') #150,512







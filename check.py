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
import cv2
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
    def __init__(self, slice):
        self.slicetomask = slice

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
        edges = measure.find_contours(self.slicetomask, level=-250, fully_connected='low',
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
            mask = np.array(img)
            body_mask += mask

        body_mask[body_mask > 1] = 1  # sanity check to make 100% sure that the mask is binary
        return body_mask.T


lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk('/Users/nivravhon/CT Scans for AngleSimulation/CT scans from Eric'):
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
    print('Affine Matrix calculation time: %s' % (a_mat_e-a_mat_s))
    return multi_aff


def voxal_to_patient(A, voxVec):
    voxalLocation = np.matmul(A, voxVec)
    return voxalLocation


def volume_rotation(vol, angle):
    rot_s = time.time()
    rotatedVol = ndimage.rotate(vol, float(angle), axes=(1, 2), reshape=False, mode='constant', cval=-1000)
    rot_e = time.time()
    print('volume rotation time: %s' % (rot_e-rot_s))
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
    cone_d = time.time()
    print('cone corr place cal time: %s' % (cone_e-cone_s))
    print('cone corr to voxal', cone_d-cone_e)
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
    print('cone based image time', f-s)
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
    print('time to find largest slice in volume', f-s)
    return slice_vol.index(maxvol)


def row_and_col_indices_for_volume_cutting(vol, largest_slice_ind, row_addition, column_addition):
    s = time.time()
    #edges = measure.find_contours(vol[largest_slice_ind, :, :], level=-250, fully_connected='low', positive_orientation='high')
    #bodycontour = find_body(edges)
    #body = create_mask_from_polygon(vol[largest_slice_ind, :, :], bodycontour)
    body = BodyMask(vol[largest_slice_ind, :, :])
    bodymask = body.mask_body()
    first_row_volume_encounter = np.min([i for i in first_nonzero(bodymask, 0) if i > 0])
    first_column_volume_encounter = np.min([i for i in first_nonzero(bodymask, 1) if i > 0])
    last_row_volume_encounter = np.max(last_nonzero(bodymask, 0))
    last_column_volume_encounter = np.max(last_nonzero(bodymask, 1))
    chopedvol = vol[:, first_row_volume_encounter:last_row_volume_encounter, first_column_volume_encounter:last_column_volume_encounter]
    s, r, c = chopedvol.shape # indexes 0:r-1, 0:c-1
    data_vol = np.full((s, r+row_addition, c+column_addition), -1000)
    data_vol[:, int(row_addition/2-1):int(row_addition/2-1)+r, int(column_addition/2-1):int(column_addition/2-1)+c] = chopedvol
    #if last_column_volume_encounter >= np.shape(ArrayDicomHu)[2]-20:
    #    last_column_volume_encounter = np.shape(ArrayDicomHu)[2]-20
    #if last_row_volume_encounter >= np.shape(ArrayDicomHu)[1]-10:
    #    last_row_volume_encounter = np.shape(ArrayDicomHu)[1]-10
    #if first_row_volume_encounter <= 50:
    #    first_row_volume_encounter = 50
    #if first_column_volume_encounter <= 20:
    #    first_column_volume_encounter = 20
    f = time.time()
    print('volume chopping time', f-s)
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
    fig.canvas.draw_idle()


def cone_integrate(data_vol, rowind, clickpointcolumn, clickedsliceind, R):
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

    data_vol[uppercone[:, 2].astype(int), uppercone[:, 0].astype(int), uppercone[:, 1].astype(int)] = 6000
    data_vol[centercone[:, 2].astype(int), centercone[:, 0].astype(int), centercone[:, 1].astype(int)] = 6000
    data_vol[lowercone[:, 2].astype(int), lowercone[:, 0].astype(int), lowercone[:, 1].astype(int)] = 6000

    return data_vol


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
    backtocornermat = ([1, 0, 0],
                       [0, 1, 0],
                       [(clickedimagesize[1]) / 2, (clickedimagesize[0]) / 2, 1])
    transmat = np.linalg.multi_dot([backtocornermat, rotmat, movetocentermat])
    invtransmat = np.linalg.inv(transmat)
    rotindvector = np.array([int(clickpointcolumn), int(rowind), 1])
    point = np.matmul(transmat, rotindvector)
    return np.matmul(invtransmat, rotindvector)

# not used in the current settings


# clicked point parameters for cone calculaiton
R = 44 #mm
distfromskin = 13 #mm
skintouppertarget = 47 #mm
skintotargetcenter = 70.5 #mm
skintolowertarget = 94 #mm
clickedsliceind = 60
clickpointcolumn = 239
anglewillclick = 5

AffinMatrix = a_multi_matrix()


row_vol_increase = 100
column_vol_increase = 50
largest_slice = find_high_Patient_volume_slice(ArrayDicomHu)
chopedvol, row_movment, column_movment = row_and_col_indices_for_volume_cutting(ArrayDicomHu, largest_slice, row_vol_increase, column_vol_increase)

conevol, slicecorrectionfactor = volume_for_cone(ArrayDicomHu, clickedsliceind, R, anglewillclick)
# calculation for rotated volume with cone
data_vol_rot = volume_rotation(ArrayDicomHu, anglewillclick)
edges = measure.find_contours(data_vol_rot[clickedsliceind, :, :], level=-250, fully_connected='low', positive_orientation='high')
bodycontour = find_body(edges)
body = create_mask_from_polygon(data_vol_rot[clickedsliceind, :, :], bodycontour)
clickedslice = data_vol_rot[clickedsliceind, :, :]
rowind = np.argwhere(body[:, clickpointcolumn] == 1)[0]

#clickedslice = data_vol_rot[clickedsliceind, :, :]
#clickedcolumn = clickedslice[:, clickpointcolumn]
#rowind = np.where(clickedcolumn > -250)[0][0]

#column_for_cone_not_rot = clickpointcolumn - column_vol_increase/2 + column_movment
#row_for_cone_not_rot = rowind - row_vol_increase/2  #row_movment

# calculation of cone points in base image without rotation. input are the points needed for cone calculation at rotated image.
input_arr = np.asarray(ArrayDicomHu)
ndim = input_arr.ndim
axes = (1, 2)

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

slice_for_cone_rotation = ArrayDicomHu[clickedsliceind, :, :]
#off set calculation
img_shape = np.asarray(input_arr.shape)

axes.sort()
in_plane_shape = img_shape[axes]
out_plane_shape = img_shape[axes]
out_center = rot_matrix @ ((out_plane_shape - 1) / 2)
in_center = (in_plane_shape - 1) / 2
offset = in_center - out_center
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

# cone integration in images without rotation
uppercone = cone(upperapexpoint, centerbasepoint, 0, R)
centercone = cone(centeralapexpoint, centerbasepoint, 0, R)
lowercone = cone(lowerapexpoint, centerbasepoint, 0, R)
ArrayDicomHu[uppercone[:, 2].astype(int), uppercone[:, 0].astype(int), uppercone[:, 1].astype(int)] = 6000
ArrayDicomHu[centercone[:, 2].astype(int), centercone[:, 0].astype(int), centercone[:, 1].astype(int)] = 6000
ArrayDicomHu[lowercone[:, 2].astype(int), lowercone[:, 0].astype(int), lowercone[:, 1].astype(int)] = 6000

# cone integration for rotated images
rotated_data_vol_with_cone = cone_integrate(data_vol_rot, rowind, clickpointcolumn, clickedsliceind, R)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

#Axial Slider
posAxialrot = ax1.get_position()
axAxial_slide_rot = plt.axes([posAxialrot.x0, posAxialrot.y0-0.1, 0.2, 0.01])  # Slider position
AxialSlider_rot = Slider(axAxial_slide_rot, '', valmin=0, valmax=data_vol_rot.shape[0]-1, valstep=1, valinit=data_vol_rot.shape[0]/2,
                                    orientation='horizontal')  # slider range
        #ax_slide.set_xticks(np.arange(self.sliceind))
AxialSlider_rot.valtext.set_visible(True)
AxialSlider_rot.label.set_color('white')
AxialSlider_rot.valtext.set_color('white')
AxialSlider_rot.on_changed(update_slider_axial_rot)

posAxial = ax3.get_position()
axAxial_slide = plt.axes([posAxial.x0, posAxial.y0-0.1, 0.2, 0.01])  # Slider position
AxialSlider = Slider(axAxial_slide, '', valmin=0, valmax=ArrayDicomHu.shape[0]-1, valstep=1, valinit=ArrayDicomHu.shape[0]/2,
                                    orientation='horizontal')  # slider range
        #ax_slide.set_xticks(np.arange(self.sliceind))
AxialSlider.valtext.set_visible(True)
AxialSlider.label.set_color('white')
AxialSlider.valtext.set_color('white')
AxialSlider.on_changed(update_slider_axial)


rotvolimage = ax1.imshow(data_vol_rot[clickedsliceind, :, :], cmap='gray')
correctnot_rot_vol = ax2.imshow(ndimage.rotate(data_vol_rot[clickedsliceind, :, :], float(-anglewillclick), reshape=False, mode='constant', cval=-1000), cmap='gray')
notrotvolimage = ax3.imshow(ArrayDicomHu[clickedsliceind, :, :], cmap='gray')
plt.show()


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







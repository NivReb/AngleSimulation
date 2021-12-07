import numpy as np
import pydicom as dcm
import matplotlib.pyplot as plt
import skimage.draw
from matplotlib.gridspec import GridSpec
import os
import pathlib
import multiprocessing
from multiprocessing import shared_memory
import concurrent.futures
from natsort import natsorted
from scipy import ndimage, interpolate
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import RangeSlider, Button, RadioButtons, Slider
import time
import json
from tkinter import *
from tkinter import filedialog
import glob
import pandas as pd
from pathlib import Path
from functools import partial
from PIL import Image, ImageDraw
from skimage import feature, measure
from skimage.draw import circle_perimeter, line
from scipy.spatial import ConvexHull
from scipy.linalg import norm
startIn = time.time()


class IndexTracker:
    def __init__(self, ax, drr_volume, title):
        self.ax = ax # axes for ploting
        self.volume = drr_volume # 3D numpy array contains 2D processed images after generating DRR in specified angles.
        #self.volumetarget = drr_volume
        self.slices, rows, cols = drr_volume.shape
        self.ind = self.slices//2
        self.im = ax.imshow(self.volume[self.ind, :, :], cmap='gray')# , extent=[0, RefDsFirst.Columns * RefDsFirst.PixelSpacing[1], 0, rows * RefDsFirst.PixelSpacing[0]])
        self.title = title
        self.targetarray = np.zeros(np.shape(self.volume[self.ind, :, :]))
        self.imageindexkeeper = self.ind # saves image index after click on target
        #self.imagetargetarray = drr_volume #volume to add target markers.
        ax.set_title(self.title + '' + 'Angle %s [Degrees]' % Angle[self.ind])
        ax.title.set_color('white')
        ax.set_ylabel('mm')
        ax.set_xlabel('mm')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Window level
        """
                    Slider settings for DRR window leveling
        """
        if title == 'DRR':
            self.pos = ax.get_position()
            self.ax_slide = plt.axes([self.pos.x0+0.78, 0.5337, 0.02, 0.35]) # Slider position
            self.ApSlider = RangeSlider(self.ax_slide, "Threshold", np.min(self.volume), np.max(self.volume), orientation='vertical') #slider range
            self.ApSlider.valtext.set_visible(False)
            self.ApSlider.label.set_color('white')
            self.ApSlider.on_changed(self.update_slider)

        self.update()

    def on_scroll(self, event):
        """
                    Change image index based on scrolling event
        :param event:
        :return:
        """
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        """
                    Update Image according to user scrolling between images
        :return:
        """

        self.im.set_data(self.volume[self.ind, :, :])
        self.ax.set_title(self.title + ' ' + 'Angle %s [degrees]' % Angle[self.ind])
        self.im.axes.figure.canvas.draw()

    def update_slider(self, val):
        """
                    Takes val parameter from RangeSlider (user decision).
                    Set the max min values of the image to those values(change all image values accordingly).
                    Update image.
        :param val:
        :return:
        """
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max)

        # Update the image's colormap
        self.im.norm.vmin = val[0]
        self.im.norm.vmax = val[1]

        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()

    def click_target_marking(self, row, column):
        """
                    Upon user click on DRR image draw
                    white cross at click coordinates
                    and update image.
        """
        if np.any(self.targetarray):
            self.volume[self.imageindexkeeper, :, :] = self.volume[self.imageindexkeeper, :, :] - self.targetarray
        else:
            pass
        self.targetarray = np.zeros(np.shape(self.volume[self.ind, :, :]))
        self.imageindexkeeper = self.ind
        upperringradius = (30.5/2) # in units mm
        lowerringradius = 25.5/2 # in units mm
        rowlowerring, columnlowerring = skimage.draw.circle_perimeter(row, column, int(lowerringradius/RefDsFirst.PixelSpacing[0]))
        rowupperring, columnupperring = skimage.draw.circle_perimeter(row, column, int(upperringradius/RefDsFirst.PixelSpacing[0]))
        rowlinetotarget, columnlinetotarget = skimage.draw.line(rowlowerring[0], columnlowerring[0], row, column)
        self.targetarray[rowlowerring, columnlowerring] = 6000
        self.targetarray[rowupperring, columnupperring] = 6000
        self.targetarray[rowlinetotarget, columnlinetotarget] = 6000
        self.volume[self.imageindexkeeper, :, :] = self.volume[self.imageindexkeeper, :, :] + self.targetarray
        self.im.set_data(self.volume[self.imageindexkeeper, :, :])
        self.im.axes.figure.canvas.draw()



class cone:
    """
                Generating acoustic beam based on user click at DRR image.
    """
    # Parameters for acoustic beam cone
    R = 44  # mm
    distfromskin = 13  # mm
    skintouppertarget = 47  # mm
    skintotargetcenter = 70.5  # mm
    skintolowertarget = 94  # mm
    gelpadthin = 30 # mm
    gelpadThick = 45 # mm

    def __init__(self, ct_volume, axAxial, axSagittal, axCoronal, indTracker):
        self.indexTracker = indTracker
        # expanding volume for cone accommodation
        s, c, r = ct_volume.shape
        self.data_vol = np.full((s,c+cone.gelpadThick,r+20), -1000)
        self.data_vol[:, 44:44+c, 9:9+r] = ct_volume

        self.conevol = np.zeros(self.data_vol.shape) #volum in the size of expanded vol that will accomodate cone.
        self.rotvol = ct_volume
        self.sliceind = self.data_vol.shape[0]//2
        self.columnind = self.data_vol.shape[2]//2
        self.rowind = self.data_vol.shape[1]//2
        self.AffinMatrix = self.a_multi_matrix()
        self.invAffinMatrix = np.linalg.inv(self.AffinMatrix)
        self.rowpixelspacing = RefDsFirst.PixelSpacing[0]
        self.imAxial = axAxial.imshow(self.data_vol[self.sliceind, :, :], cmap='gray')
        self.imSagittal = axSagittal.imshow(np.rot90(self.data_vol[:, :, self.columnind], k=3), cmap='gray')
        self.imCoronal = axCoronal.imshow(self.data_vol[:, self.rowind, :], cmap='gray')


        # Parameters and settings for CT images plots and sliders
        axCoronal.set_title('Coronal Plane')
        axAxial.set_title('Axial Plane')
        axSagittal.set_title('Sagittal Plane')
        axCoronal.title.set_color('white')
        axAxial.title.set_color('white')
        axSagittal.title.set_color('white')
        axCoronal.set_ylabel('mm')
        axCoronal.set_xlabel('mm')
        axCoronal.yaxis.label.set_color('white')
        axCoronal.xaxis.label.set_color('white')
        axCoronal.tick_params(axis='x', colors='white')
        axCoronal.tick_params(axis='y', colors='white')
        axAxial.set_ylabel('mm')
        axAxial.set_xlabel('mm')
        axAxial.yaxis.label.set_color('white')
        axAxial.xaxis.label.set_color('white')
        axAxial.tick_params(axis='x', colors='white')
        axAxial.tick_params(axis='y', colors='white')
        axSagittal.set_ylabel('mm')
        axSagittal.set_xlabel('mm')
        axSagittal.yaxis.label.set_color('white')
        axSagittal.xaxis.label.set_color('white')
        axSagittal.tick_params(axis='x', colors='white')
        axSagittal.tick_params(axis='y', colors='white')

        #Axial Slider
        self.posAxial = axAxial.get_position()
        self.axAxial_slide = plt.axes([self.posAxial.x0, self.posAxial.y0-0.1, 0.2, 0.01])  # Slider position
        self.AxialSlider = Slider(self.axAxial_slide, '', valmin=0, valmax=self.data_vol.shape[0]-1, valinit=self.sliceind, valstep=1,
                                    orientation='horizontal')  # slider range
        #ax_slide.set_xticks(np.arange(self.sliceind))
        self.AxialSlider.valtext.set_visible(True)
        self.AxialSlider.label.set_color('white')
        self.AxialSlider.valtext.set_color('white')
        self.AxialSlider.on_changed(self.update_slider_Axial)
        #Sagittal Slider
        self.posSagittal = axSagittal.get_position()
        self.axSagittal_slide = plt.axes([self.posSagittal.x0, self.posAxial.y0 - 0.1, 0.2, 0.01])  # Slider position
        self.SagittalSlider = Slider(self.axSagittal_slide, '', valmin=0, valmax=self.data_vol.shape[2] - 1, valinit=self.columnind,
                               valstep=1,
                               orientation='horizontal')  # slider range
        # ax_slide.set_xticks(np.arange(self.sliceind))
        self.SagittalSlider.valtext.set_visible(True)
        self.SagittalSlider.label.set_color('white')
        self.SagittalSlider.valtext.set_color('white')
        self.SagittalSlider.on_changed(self.update_slider_Sagittal)
        # Coronal Slider
        self.posCoronal = axCoronal.get_position()
        self.axCoronal_slide = plt.axes(
            [self.posCoronal.x0, self.posCoronal.y0 - 0.1, 0.2, 0.01])  # Slider position
        self.CoronalSlider = Slider(self.axCoronal_slide, '', valmin=0, valmax=self.data_vol.shape[1] - 1,
                                     valinit=self.rowind,
                                     valstep=1,
                                     orientation='horizontal')  # slider range
        # ax_slide.set_xticks(np.arange(self.sliceind))
        self.CoronalSlider.valtext.set_visible(True)
        self.CoronalSlider.label.set_color('white')
        self.CoronalSlider.valtext.set_color('white')
        self.CoronalSlider.on_changed(self.update_slider_Coronal)

    def update_slider_Axial(self, val):
        """
                    Takes slider val and update CT image based on the val
        :param val: CT slice
        :return:
        """
        self.imAxial.set_data(self.data_vol[val, :, :].squeeze())
        fig.canvas.draw_idle()

    def update_slider_Sagittal(self, val):
        """
                    Takes slider val and update CT image based on the val
        :param val: CT slice
        :return:
        """
        self.imSagittal.set_data(np.rot90(self.data_vol[:, :, val].squeeze(), k=3))
        fig.canvas.draw_idle()

    def update_slider_Coronal(self, val):
        """
                    Takes slider val and update CT image based on the val
        :param val: CT slice
        :return:
        """
        self.imCoronal.set_data(self.data_vol[:, val, :].squeeze())
        fig.canvas.draw_idle()

    def on_click(self, event):
        """
                    Once user clicked image accept click indices.
                    Initiate cone calculation and implementation at CT volume.
        :param event: click event
        :return:
        """
        # get the x and y pixel coords

        if event.button is MouseButton.LEFT and event.inaxes == self.indexTracker.ax:
            s = time.time()
            self.angle = Angle[self.indexTracker.ind]

            x, y = event.x, event.y
            #ax2 = event.inaxes  # the axes instance
            print('data coords %f %f' % (np.round(event.xdata), np.round(event.ydata)))
            self.sliceind = np.round(event.ydata)
            self.columnind = np.round(event.xdata)
            self.indexTracker.click_target_marking(self.sliceind.astype(int), self.columnind.astype(int))
            # remove old cone after new click
            if np.any(self.conevol):
                self.data_vol = self.data_vol - self.conevol
                self.conevol = np.zeros(self.data_vol.shape)
            else:
                pass

            self.data_vol = self.volume_rotation(self.data_vol, self.angle)
            #find skin row index after rotation with cutted volume
            self.sliceforcontour = self.data_vol[self.sliceind.astype(int), :, :]
            bodymaskclass = BodyMask(self.sliceforcontour)
            bodymask = bodymaskclass.mask_body()
            self.rowind = np.argwhere(bodymask[:, self.columnind.astype(int)] == 1)[0]
            base, up, center, low = self.cone_coordinates(self.rowind, self.columnind, self.sliceind) # cone points coordinates in patient coordinate system
            centercone = self.cone(center, base, 0, cone.R) # center cone indexes
            uppercone = self.cone(up, base, 0, cone.R)# upper cone indexes
            lowercone = self.cone(low, base, 0, cone.R)# lower cone indexes

            self.conevol = self.cone_volume_integrate(self.conevol, centercone, uppercone, lowercone)
            self.data_vol = self.data_vol + self.conevol
            f = time.time()
            print('click target cal time', f-s)
            self.update()

    def update(self):
        """
                    With each user click update CT images with cone at clicked place.
        :return:
        """
        self.imAxial.set_data(self.data_vol[self.sliceind.astype(int), :, :].squeeze())
        self.imSagittal.set_data(np.rot90(self.data_vol[:, :, self.columnind.astype(int)].squeeze(), k=3))
        self.imCoronal.set_data(self.data_vol[:, int(np.round(self.rowind + cone.skintotargetcenter / self.rowpixelspacing)), :].squeeze())
        self.imAxial.axes.figure.canvas.draw()
        self.imSagittal.axes.figure.canvas.draw()
        self.imCoronal.axes.figure.canvas.draw()

    def cone_volume_integrate(self, volume, center_cone, upper_cone, lower_cone):
        """

        :param lower_cone: indices for the lower cone
        :param center_cone: indices for the center cone
        :param upper_cone: indices for the upper cone
        :param volume: CT volume after rotation in the angle user chose
        :return: CT volume after cones implementation
        """
        center_cone[:, 2][center_cone[:, 2] >= len(lstFilesDCM)] = len(lstFilesDCM) - 1
        upper_cone[:, 2][upper_cone[:, 2] >= len(lstFilesDCM)] = len(lstFilesDCM) - 1
        lower_cone[:, 2][lower_cone[:, 2] >= len(lstFilesDCM)] = len(lstFilesDCM) - 1
        volume[center_cone[:, 2].astype(int), center_cone[:, 0].astype(int), center_cone[:, 1].astype(
            int)] = 6000
        volume[upper_cone[:, 2].astype(int), upper_cone[:, 0].astype(int), upper_cone[:, 1].astype(
            int)] = 6000
        volume[lower_cone[:, 2].astype(int), lower_cone[:, 0].astype(int), lower_cone[:, 1].astype(
            int)] = 6000
        return volume

    def cone_coordinates(self, row_ind, column_ind, slice_ind):
        """
                        Transformation from voxel to patient coordinate system for cone calculation.
        :param row_ind: clicked row index
        :param column_ind: clicked column index
        :param slice_ind: clicked slice index
        :return: points coordinates in patient world which define cones
        """
        # pixal world to patient coordinate system
        centerbasepoint = self.voxel_to_patient(self.AffinMatrix, np.array(
            (int(np.round(row_ind - cone.distfromskin / self.rowpixelspacing)), column_ind,
             slice_ind, 1)))[0:3]
        upperapexpoint = self.voxel_to_patient(self.AffinMatrix, np.array(
            (int(np.round(row_ind + cone.skintouppertarget / self.rowpixelspacing)), column_ind,
             slice_ind, 1)))[0:3]
        centeralapexpoint = self.voxel_to_patient(self.AffinMatrix, np.array(
            (int(np.round(row_ind + cone.skintotargetcenter / self.rowpixelspacing)), column_ind,
             slice_ind, 1)))[0:3]
        lowerapexpoint = self.voxel_to_patient(self.AffinMatrix, np.array(
            (int(np.round(row_ind + cone.skintolowertarget / self.rowpixelspacing)), column_ind,
             slice_ind, 1)))[0:3]
        return centerbasepoint, upperapexpoint, centeralapexpoint, lowerapexpoint

    def a_multi_matrix(self):
        """
        Calculate the affine matrix for voxel to patient coordinate system transformation
        :return: 4x4 affine matrix based on DICOM data
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
        return multi_aff

    def voxel_to_patient(self, affine_matrix, voxel_coordinates):
        """

        :param affine_matrix: The calculated affine matrix
        :param voxel_coordinates: Coordinates to transform
        :return: Voxel location in patient coordinate system
        """
        voxel_location = np.matmul(affine_matrix, voxel_coordinates)
        return voxel_location

    def volume_rotation(self, vol, angle):
        """

        :param vol: Volume to rotate
        :param angle: angle to rotate to
        :return: rotated volume
        """
        rotatedVol = ndimage.rotate(vol, float(angle), axes=(1, 2), reshape=False, mode='constant', cval=-1000)
        return rotatedVol

    def cone(self, p0, p1, R0, R1):
        """
        Calculate cone location based on clicked points in patient coordinates system.
        Transfer cone location from patient coordinate system back to voxel coordinate system.

        Based on https://stackoverflow.com/a/39823124/190597 (astrokeat)
        :param p0: - cone Apex point
        :param p1: - cone base center point
        :param R0: - radius at the cone apex
        :param R1: - radius at cone base
        :return: Array with cone indices
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
        # ax.plot_surface(X, Y, Z, color=color, linewidth=0, antialiased=False)
        ones = np.full(n * n, 1)
        coorArray = np.zeros((len(ones), 4))
        coorArray[:, 0] = X.flatten()
        coorArray[:, 1] = Y.flatten()
        coorArray[:, 2] = Z.flatten()
        coorArray[:, 3] = ones
        pixcoorArray = np.round(coorArray.dot(self.invAffinMatrix.T)[:, :-1]) # calculate the transformation between patient coordinate system back to voxel coordinate system
        '''
        ones = np.full(X.shape, 1)
        coorArray = np.zeros((4, len(ones), len(ones)))
        coorArray[0, :, :] = X
        coorArray[1, :, :] = Y
        coorArray[2, :, :] = Z
        coorArray[3, :, :] = ones
        pixcoorArray = np.zeros(coorArray.shape)
        for k in range(0, len(ones)):
            for j in range(0, len(ones)):
                pixcoorArray[:, k, j] = np.round(np.matmul(self.invAffinMatrix, coorArray[:, k, j]))
        '''
        return pixcoorArray


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


def voxal_ray_distance(ArrayShape, angle, RefDs):
    """
                Calculating the distance each X-ray ray passing through each voxel.
    :param ArrayShape: CT volume array shape
    :param angle: volume rotation angle
    :param RefDs: reference DICOM file for attributes
    :return: Array containing the distances per voxel.
    """
    #calculation the distance each ray passing through each voxal
    PixelSpacingRow = int(RefDs.PixelSpacing[0])/10
    DistArray = np.full(ArrayShape, PixelSpacingRow*np.cos(angle*np.pi/180))
    return DistArray


def f_shared(angle, dtype, shape, RefDs, airattcoeff):
    """
            Create shared memory space in which to save the volume data.
            Rotating volume.
            Calculate DRR projection.

    :param angle: angle to rotate volume
    :param dtype: volume data type
    :param shape: volume shape
    :param RefDs: reference DICOM file attributes
    :return:DRR image rotate at specified angle
    """
    print(angle)
    sh = shared_memory.SharedMemory(name='ArrayDicomMuShared')
    ArrayDicomMu = np.ndarray(shape, dtype, buffer=sh.buf)
    angle_base = ndimage.rotate(ArrayDicomMu, float(angle), axes=(1, 2), reshape=False, mode='constant', cval=airattcoeff) # positive angles rotating counter clock wise
    VoxalDist = voxal_ray_distance(ArrayDicomMu.shape, angle, RefDs) # calculating the distance each ray passing through voxal
    res = np.sum(np.exp(-1 * angle_base * VoxalDist), axis=1)
    sh.close()
    return res


def ap_drr(ArrayDicomMu, Angle, airattcoeff):
    """
                calculating for each defined Angle the DRR ( each angle calculated in different core).
                stacking all images to 3D numpy array.

    :param ArrayDicomMu: Volume with each voxel value set to the linear attenuation coefficient of that specific voxel
    :param Angle: list of angles to perform the calculation on
    :return: 3D numpy array containing DRR images at each angle
    """
    ArrayDicomMu = ArrayDicomMu.astype(np.float32)
    APBaseAngleDRR = np.zeros((len(Angle), np.shape(ArrayDicomMu)[0], np.shape(ArrayDicomMu)[2]))
    t = time.time()
    tasks = []
    for i in range(0, len(Angle)):
        tasks.append(Angle[i])

    print("Time1: {:.2f}".format(time.time()-t))
    shm = shared_memory.SharedMemory(create=True, size=ArrayDicomMu.nbytes, name='ArrayDicomMuShared')
    shared_ArrayDicomMu = np.ndarray(ArrayDicomMu.shape, dtype=ArrayDicomMu.dtype, buffer=shm.buf)
    shared_ArrayDicomMu[:] = ArrayDicomMu[:]

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        angle_bases = p.map(partial(f_shared, dtype=ArrayDicomMu.dtype, shape=ArrayDicomMu.shape, RefDs=RefDsFirst, airattcoeff=airattcoeff), tasks)
    shm.close()
    shm.unlink()
        #need to ref SOPInstanceUID for quick load of already processed DICOMS
    print("Time2: {:.2f}".format(time.time() - t))
    for i in range(0, len(Angle)):
        # APBaseAngleDRR[i, :, :] = np.sum(np.rot90(np.exp(MinusOneArrayAP * angle_bases[i] * ArrayDicomDistancesPlane), k=2, axes=(1, 2)), axis=1)
        APBaseAngleDRR[i, :, :] = angle_bases[i]
        # new cal for HUtoMU table
    print("Time3: {:.2f}".format(time.time() - t))

    s = time.time()
    print('DRR Run Time is', s - t)
    return APBaseAngleDRR

'''
def ap_drr2(ArrayDicomMu, Angle):

    APBaseAngleDRR = np.zeros((len(Angle), np.shape(ArrayDicomMu)[0], np.shape(ArrayDicomMu)[2]))
    ArrayDicomDistancesPlane = np.full(np.shape(ArrayDicomMu), int(RefDs.PixelSpacing[1]) / 10)
    MinusOneArrayAP = np.full(np.shape(ArrayDicomMu), -1)
    t = time.time()
    tasks = []
    for i in range(0, len(Angle)):
        tasks.append(Angle[i])
    angle_bases = []
    print("Time1: {:.2f}".format(time.time()-t))
    #with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    #    angle_base = p.map(partial(f, ArrayDicomMu=ArrayDicomMu), tasks)
    #    angle_bases.append(angle_base)
    for task in tasks:
        print(task)
        angle_base = partial(f, ArrayDicomMu=ArrayDicomMu)(task)
        angle_bases.append(angle_base)
        #need to ref SOPInstanceUID for quick load of already processed DICOMS
    print("Time2: {:.2f}".format(time.time() - t))
    for i in range(0, len(Angle)):
        APBaseAngleDRR[i, :, :] = np.sum(
            np.rot90(np.exp(MinusOneArrayAP * angle_bases[i] * ArrayDicomDistancesPlane), k=2, axes=(1, 2)), axis=1)
        # new cal for HUtoMU table
    print("Time3: {:.2f}".format(time.time() - t))

    s = time.time()
    print('DRR Run Time is', s - t)
    return APBaseAngleDRR
'''


def data_to_dict(drr_vol, kVp, data_dict):
    """
            Based on DICOM SOP Instance UID.
            Saving running time we save each DRR calculation at different kVp in JSON files.
            Next time user want to use same parameters he will not have to wait for DRR calculation again
    :param drr_vol: DRR volume after calculation
    :param kVp: kVp value in which the calculation was done
    :param data_dict: dictionary that will populate all the information
    :return:
    """
    if keys_exists(data_dict, str(RefDsFirst.SOPInstanceUID)): # if SOP Instance UID allready proccessed and at requested kVp data was'nt calculated insert that new kVp calculated data to dict ans json
        data_dict[str(RefDsFirst.SOPInstanceUID)][str(kVp)] = {'DRR': drr_vol}
    else:
        data_dict[str(RefDsFirst.SOPInstanceUID)] = {}
        data_dict[str(RefDsFirst.SOPInstanceUID)][str(kVp)] = {'DRR': drr_vol}
        data_dict[str(RefDsFirst.SOPInstanceUID)]['HU'] = {'HounsfieldUnit': ArrayDicomHu}
    with open('%s.json' % RefDsFirst.SOPInstanceUID, 'w') as f:
        f.write(json.dumps(data_dict, default=convert))


def convert(data_array):
    """

    :param data_array: array contains data to store in JSON file
    :return: data array as a list
    """
    if hasattr(data_array, "tolist"):  # numpy arrays have this
        return {"$array": data_array.tolist()}  # Make a tagged object
    raise TypeError(data_array)


def deconvert(json_data):
    """

    :param json_data: data comes from json file
    :return: data from json as numpy array
    """
    if len(json_data) == 1:  # Might be a tagged object...
        key, value = next(iter(json_data.items()))  # Grab the tag and value
        if key == "$array":  # If the tag is correct,
            return np.array(value)  # cast back to array
    return json_data


def linear_attenuation_coefficient_calculation(kVp):
    """
                Using linear attenuation coefficient tables for specified materials(based on kVp), density tables for
                those materials and average Hounsfield units.
                Perform linear interpolation to create linear attenuation coefficient vs hounsfield unit graph to
                transform DICOM data from hounsfield tolinear attenuation coefficient.

    :param kVp: kVp for linear attenuation coefficient calculation
    :return: hounsfield unit values vs linear attenuation coefficient values array which correspond to relevant materials
    """
    l = time.time()
    MeV = (kVp / 3) / 1000
    csv_files = natsorted(glob.glob(os.path.join(pathlib.Path(__file__).parent / 'Materials attenuation coefficient raw', "*.csv")))
    DensityTable = pd.read_csv(pathlib.Path(__file__).parent /'Density.csv')
    MaterialDensity = DensityTable[["Material", "Density"]]
    AvgHUfull = pd.read_csv(pathlib.Path(__file__).parent /'Average HU.csv')
    AvgHU = np.asarray(AvgHUfull["HU"]).astype(float)
    Mu_Data = np.zeros((len(AvgHU), 2))
    Mu_count = 0
    for f in csv_files:
        # read the csv file into NumPy Array
        df = pd.read_csv(f)
        Val = df.to_numpy()
        Mu_From_Table = np.interp(MeV, Val[:, 0], Val[:, 1])  # linear interpolation to find mu for relevant MeV
        DensityInd = np.asarray(np.where(
            MaterialDensity["Material"] == Path(f).stem))  # find the density for the material in question
        DensityInd = DensityInd[DensityInd != 0]
        Mu_For_Cal = np.multiply(Mu_From_Table, float(np.asarray(MaterialDensity["Density"][
                                                                     DensityInd])))  # getting the mu for the specified MeV after multypling by the density
        Subcount = str(AvgHUfull['Substance']).count(Path(f).stem)
        if Subcount > 1:
            Mu_Data[Mu_count, 0] = Mu_For_Cal
            Mu_Data[Mu_count + 1, 0] = Mu_For_Cal
            Mu_Data[Mu_count, 1] = AvgHU[Mu_count]
            Mu_Data[Mu_count + 1, 1] = AvgHU[Mu_count + 1]
            Mu_count = Mu_count + 2
        else:
            Mu_Data[Mu_count, 0] = Mu_For_Cal
            Mu_Data[Mu_count, 1] = AvgHU[Mu_count]
            Mu_count = Mu_count + 1

        HUtoMU = Mu_Data[np.argsort(Mu_Data[:, 1])]
    n = time.time()
    print('Linear Atenuation Coefficient Cal Run Time Is', n - l)

    return HUtoMU


def keys_exists(element, *keys):
    '''
    Check if *keys (nested) exists in `element` (dict).
    '''
    if not isinstance(element, dict):
        raise AttributeError('keys_exists() expects dict as first argument.')
    if len(keys) == 0:
        raise AttributeError('keys_exists() expects at least two arguments, one given.')

    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True


def dir_path():
    """
            Use Tkinter to get file directory path
    :return: directory path
    """
    root = Tk()
    root.withdraw()
    root.directoryname = filedialog.askdirectory(title="Select CT Images Directory")
    root.destroy()
    return root.directoryname


def get_data(kVp):
    """
            Accept kVp for calculation and check if calculation was all ready done with selected kVp and SOP Instance UID.
            If yes it extract the data from the relevant JSON file, if not - calculate the DRR with new parameters.

    :param kVp: kVp for calculation
    :return: volume after DRR calculation
    """
    with open('%s.json' % RefDsFirst.SOPInstanceUID) as file:
        data = file.read()
        data_dict = json.loads(data, object_hook=deconvert)
        b = time.time()
        if keys_exists(data_dict, str(RefDsFirst.SOPInstanceUID), str(kVp)):
            APData = data_dict[str(RefDsFirst.SOPInstanceUID)][str(kVp)]['DRR']
            HUdata = data_dict[str(RefDsFirst.SOPInstanceUID)]['HU']['HounsfieldUnit']
            v = time.time()
            print('Simulation Run time is', v - b)
            return APData, HUdata
        else:
            APData = cal_data(kVp, data_dict)
            return APData


def cal_data(kVp, data_dict):
    """
                Accept kVp for calculation.
                Calculate voxel linear attenuation coefficient using linear interpolation
    :param kVp: kVp for calculation
    :param data_dict: place for data storing
    :return: volume with DRR image.
    """
    b = time.time()
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)

    HUtoMUkVp = linear_attenuation_coefficient_calculation(kVp)
    ArrayDicomMu = np.interp(ArrayDicomHu, HUtoMUkVp[:, 1],
                             HUtoMUkVp[:, 0])  # Array based on Linear attenuation coefficient
    air_att_coeff = HUtoMUkVp[0, 0]
    o = time.time()
    APDRR = ap_drr(ArrayDicomMu, Angle, air_att_coeff)
    v = time.time()
    print('Calculation Time is', v - o)
    print('DRR calculation Run time is', v - b)
    data_to_dict(APDRR, kVp, data_dict)
    return APDRR


def ct_data_analysis(FilePath):
    """
                Initiating  DRR calculation, setting parameters

    :param FilePath: directory with CT images file path
    :return: calculated DRR volume
    """
    global RefDsFirst, RefDsLast, Angle, ArrayDicomHu, lstFilesDCM

    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(FilePath):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName, filename))

    # Get ref file
    lstFilesDCM = natsorted(lstFilesDCM)  # sorting the files in order
    RefDsLast = dcm.read_file(lstFilesDCM[len(lstFilesDCM) - 1])  # refrence file
    RefDsFirst = dcm.read_file(lstFilesDCM[0])

    Angle = range(-30, 30)
    kVp = 60
    if os.path.isfile('/Users/nivravhon/PycharmProjects/pythoProject/%s.json' % RefDsFirst.SOPInstanceUID):
        AP, ArrayDicomHu = get_data(kVp)
        return AP, ArrayDicomHu
    else:
        PatientPosition = RefDsFirst.PatientPosition
        ConstPixelDims = (len(lstFilesDCM), int(RefDsFirst.Rows), int(RefDsFirst.Columns))
        ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDsFirst.pixel_array.dtype)
        ArrayDicomHu = np.zeros(ConstPixelDims, dtype=RefDsFirst.pixel_array.dtype)
        for filenameDCM in lstFilesDCM:
            # read the file
            ds = dcm.read_file(filenameDCM)
            # store the raw image data
            ArrayDicom[len(lstFilesDCM) - 1 - lstFilesDCM.index(filenameDCM), :,
            :] = ds.pixel_array  # [149,:,:] slice where the head is
        # rotate volume based on Patient Position DICOM attribute
        if PatientPosition == 'HFS':  # HFS - Head First Supine
            ArrayDicomRot = np.rot90(ArrayDicom, k=2, axes=(1, 2))
        elif PatientPosition == 'HFP':  # HFP - Head First Prone
            ArrayDicomRot = np.rot90(ArrayDicom, k=2, axes=(0, 2))
        elif PatientPosition == 'FFP':  # FFP - Feet First Prone
            ArrayDicomRot = ArrayDicom
        elif PatientPosition == 'FFS':  # FFS - Feet First Supine
            ArrayDicomRot = np.rot90(ArrayDicom, k=2, axes=(1, 2))
        else:
            ArrayDicomRot = ArrayDicom
            print('Patient Position could not detect')

        # converting to Hounsfield Unit
        ArrayDicomHu = np.add(np.multiply(ArrayDicomRot, int(RefDsFirst.RescaleSlope)),
                              int(RefDsFirst.RescaleIntercept))  # Array based on hounsfield Unit
        ArrayDicomHu[ArrayDicomHu <= -1000] = -1000
        ArrayDicomHu = find_high_Patient_volume_slice(ArrayDicomHu)
        data_dict = {}
        AP = cal_data(kVp, data_dict)
        return AP, ArrayDicomHu

def find_high_Patient_volume_slice(vol):
    s = time.time()
    slice_vol = []
    for i in range(0, np.shape(vol)[0]-1):
        #edges = measure.find_contours(vol[i, :, :], level=-250, fully_connected='low',
                                      #positive_orientation='high')
       # bodycontour = find_body(edges)
        body = BodyMask(vol[i, :, :])
        bodymask = body.mask_body()#create_mask_from_polygon(vol[i, :, :], bodycontour)
        slice_vol.append(np.count_nonzero(bodymask == 1))
    maxvol = max(slice_vol)
    f = time.time()
    highvolslice = slice_vol.index(maxvol)
    print('time to find largest slice in volume', f-s)
    return row_and_col_indices_for_volume_cupping(vol, highvolslice)


def row_and_col_indices_for_volume_cupping(vol, largest_slice_ind):
    s = time.time()
    #edges = measure.find_contours(vol[largest_slice_ind, :, :], level=-250, fully_connected='low', positive_orientation='high')
    #bodycontour = find_body(edges)
    #body = create_mask_from_polygon(vol[largest_slice_ind, :, :], bodycontour)
    body = BodyMask(vol[largest_slice_ind, :, :])
    bodymask = body.mask_body()
    first_column_volume_encounter = np.min([i for i in first_nonzero(bodymask, 0) if i > 0])
    first_row_volume_encounter = np.min([i for i in first_nonzero(bodymask, 1) if i > 0])
    last_column_volume_encounter = np.max(last_nonzero(bodymask, 0))
    last_row_volume_encounter = np.max(last_nonzero(bodymask, 1))
    #if last_column_volume_encounter >= np.shape(ArrayDicomHu)[2]-25:
    #    last_column_volume_encounter = np.shape(ArrayDicomHu)[2]-25
    #if last_row_volume_encounter >= np.shape(ArrayDicomHu)[1]-25:
    #    last_row_volume_encounter = np.shape(ArrayDicomHu)[1]-25
    #if first_row_volume_encounter <= 25:
    #    first_row_volume_encounter = 25
    #if first_column_volume_encounter <= 25:
    #    first_column_volume_encounter = 25
    f = time.time()
    print('volume chopping time', f-s)
    return vol[:, first_column_volume_encounter:last_column_volume_encounter, first_row_volume_encounter:last_row_volume_encounter]#vol[:, first_column_volume_encounter-25:last_column_volume_encounter+25, first_row_volume_encounter-25:last_row_volume_encounter+25]


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val,  invalid_val)


if __name__ == '__main__':
    AP, ArrayDicomHu = ct_data_analysis(dir_path())
    fig = plt.figure()
    fig.patch.set_facecolor('black')
    gs = GridSpec(2, 3, figure=fig)
    #ax1 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])
    #trackerMIP = IndexTracker(ax1, MIP, 'MIP')
    trackerAP = IndexTracker(ax1, AP, 'DRR')
     #getting the angle at which the user is corrently viewing
    Cone = cone(ArrayDicomHu, ax2, ax3, ax4, trackerAP)
    fig.canvas.mpl_connect('scroll_event', trackerAP.on_scroll)
    fig.canvas.mpl_connect('button_press_event', Cone.on_click)


    #plt.connect('button_press_event', on_click)

    plt.show()


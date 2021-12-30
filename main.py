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
import matplotlib.lines as lines
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
from scipy import special
startIn = time.time()


class IndexTracker:

    upperringradius = 30.5 / 2  # in units mm
    lowerringradius = 25.5 / 2  # in units mm

    def __init__(self, ax, drr_volume, title):
        self.ax = ax # axes for ploting
        self.volume = drr_volume # 3D numpy array contains 2D processed images after generating DRR in specified angles.
        #self.volumetarget = drr_volume
        self.slices, rows, cols = drr_volume.shape
        self.ind = self.slices//2
        self.im = ax.imshow(self.volume[self.ind, :, :], cmap='gray')# , extent=[0, RefDsFirst.Columns * RefDsFirst.PixelSpacing[1], 0, rows * RefDsFirst.PixelSpacing[0]])
        self.title = title
        self.targetarray = np.zeros(np.shape(self.volume[self.ind, :, :]))
        self.imageindexkeeper = 0 # saves image index after click on target
        self.counter = 0
        self.clickedrow = 0
        self.clickedcolumn = 0
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

        # Remove rings if Angle was changed & draw the ring back when user toggle to the angle he had clicked on before
        if self.counter > 0 and self.ax.patches:
            self.ax.patches[0].remove()
            self.ax.patches[0].remove()
            self.ax.patches[0].remove()

        if self.imageindexkeeper == self.ind:
            cirup = plt.Circle((self.clickedcolumn, self.clickedrow), IndexTracker.upperringradius, color='w', fill=False)
            cirlow = plt.Circle((self.clickedcolumn, self.clickedrow), IndexTracker.lowerringradius, color='w', fill=False)
            targetline = plt.Rectangle((self.clickedcolumn-0.5, self.clickedrow - IndexTracker.lowerringradius), 1,
                                       IndexTracker.lowerringradius-1, color='w')
            self.ax.add_patch(cirup)
            self.ax.add_patch(cirlow)
            self.ax.add_patch(targetline)
        self.update()

    def update(self):
        """
                    Update Image according to user scrolling between images
        :return:
        """

        self.im.set_data(self.volume[self.ind, :, :])
        self.ax.set_title(self.title + ' ' + 'Angle %s [degrees]' % Angle[self.ind])
        self.im.axes.figure.canvas.draw_idle()

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
        cirup = plt.Circle((column, row), IndexTracker.upperringradius, color='w', fill=False)
        cirlow = plt.Circle((column, row), IndexTracker.lowerringradius, color='w', fill=False)
        targetline = plt.Rectangle((column-0.5, row-IndexTracker.lowerringradius), 1, IndexTracker.lowerringradius-1, color='w', fill=True)

        self.clickedcolumn = column
        self.clickedrow = row

        if self.counter == 0:
            self.counter = self.counter + 1
            self.imageindexkeeper = self.ind
        elif self.counter > 0 and self.ax.patches:
            self.ax.patches[0].remove()
            self.ax.patches[0].remove()
            self.ax.patches[0].remove()
            self.imageindexkeeper = self.ind
        elif self.counter > 0:
            self.imageindexkeeper = self.ind

        self.ax.add_patch(cirup)
        self.ax.add_patch(cirlow)
        self.ax.add_patch(targetline)
        print(self.ax.patches)


class Cone:
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
        #s, c, r = ct_volume.shape
        #self.data_vol = np.full((s, c+Cone.gelpadThick, r+20), -1000)
        #self.data_vol[:, 44:44+c, 9:9+r] = ct_volume
        self.conevol = np.zeros(ct_volume.shape) #volum in the size of expanded vol that will accomodate cone.
        self.basevol = ct_volume
        self.clickcounter = 0
        self.sliceind = self.basevol.shape[0]//2
        self.columnind = self.basevol.shape[2]//2
        self.rowind = self.basevol.shape[1]//2
        self.AffinMatrix = self.a_multi_matrix()
        self.invAffinMatrix = np.linalg.inv(self.AffinMatrix)
        self.rowpixelspacing = RefDsFirst.PixelSpacing[0]
        self.imAxial = axAxial.imshow(self.basevol[self.sliceind, :, :], cmap='gray')
        self.imSagittal = axSagittal.imshow(np.rot90(self.basevol[:, :, self.columnind], k=3), cmap='gray')
        self.imCoronal = axCoronal.imshow(self.basevol[:, self.rowind, :], cmap='gray')


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
        self.AxialSlider = Slider(self.axAxial_slide, '', valmin=0, valmax=self.basevol.shape[0]-1, valstep=1, valinit=self.basevol.shape[0]/2,
                                    orientation='horizontal')  # slider range
        #ax_slide.set_xticks(np.arange(self.sliceind))
        self.AxialSlider.valtext.set_visible(True)
        self.AxialSlider.label.set_color('white')
        self.AxialSlider.valtext.set_color('white')
        self.AxialSlider.on_changed(self.update_slider_Axial)
        #Sagittal Slider
        self.posSagittal = axSagittal.get_position()
        self.axSagittal_slide = plt.axes([self.posSagittal.x0, self.posAxial.y0 - 0.1, 0.2, 0.01])  # Slider position
        self.SagittalSlider = Slider(self.axSagittal_slide, '', valmin=0, valmax=self.basevol.shape[2] - 1, valinit=self.basevol.shape[2]/2,
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
        self.CoronalSlider = Slider(self.axCoronal_slide, '', valmin=0, valmax=self.basevol.shape[1] - 1, valinit=self.basevol.shape[1]/2,
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
        self.imAxial.set_data(self.basevol[val, :, :].squeeze())
        fig.canvas.draw_idle()

    def update_slider_Sagittal(self, val):
        """
                    Takes slider val and update CT image based on the val
        :param val: CT slice
        :return:
        """
        self.imSagittal.set_data(np.rot90(self.basevol[:, :, val].squeeze(), k=3))
        fig.canvas.draw_idle()

    def update_slider_Coronal(self, val):
        """
                    Takes slider val and update CT image based on the val
        :param val: CT slice
        :return:
        """
        self.imCoronal.set_data(self.basevol[:, val, :].squeeze())
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
            if self.clickcounter > 0:
                self.basevol = self.basevol - self.conevol
                self.clickcounter += 1
            t = time.time()

            self.angle = Angle[self.indexTracker.ind]
            x, y = event.x, event.y
            #ax2 = event.inaxes  # the axes instance
            print('data coords %f %f' % (np.round(event.xdata), np.round(event.ydata)))
            self.sliceind = np.round(event.ydata)
            self.columnind = np.round(event.xdata)
            self.indexTracker.click_target_marking(self.sliceind.astype(int), self.columnind.astype(int))
            '''
            # remove old cone after new click
            s, c, r = self.basevol.shape
            self.data_vol = np.full((s, c + Cone.gelpadThick, r + 20), -1000)
            self.data_vol[:, 44:44 + c, 9:9 + r] = self.basevol
            simulationR = 19
            '''
            self.slice = ndimage.rotate(self.basevol[self.sliceind.astype(int),:,:], float(self.angle), reshape=False, mode='constant', cval=-1000)
            o = time.time()
            #self.data_vol, self.correctionfactor = self.volume_for_cone(self.data_vol, self.sliceind.astype(int), int(Cone.R-simulationR), self.angle)# correction factor = factor to change sliceind to new place.
            #l = time.time()
            self.conevol = np.zeros(self.basevol.shape) #set volume dimensions for cone.
            #find skin row index after rotation with cutted volume
            #self.clickedslice = self.data_vol[self.sliceind.astype(int) - self.correctionfactor, :, :]
            clickedcolumn = self.slice[:, self.columnind.astype(int)]
            if any(x in clickedcolumn for x in range(-250, 0)):
                self.rowind = np.where(clickedcolumn > -250)[0][0]
            else:
                print('Selected Target is out of the Allowed Therapeutic Area, please select target within Therapeutic Area')

            g = time.time()
            base, up, center, low = self.cone_coordinates(self.rowind, self.columnind, self.sliceind) # cone points coordinates in patient coordinate system
            centercone = self.cone(center, base, 0, Cone.R) # center cone indexes
            uppercone = self.cone(up, base, 0, Cone.R)# upper cone indexes
            lowercone = self.cone(low, base, 0, Cone.R)# lower cone indexes

            d = time.time()
            self.conevol = self.cone_volume_integrate(self.conevol, centercone, uppercone, lowercone, Cone.R)
            self.basevol = self.basevol + self.conevol
            self.clickcounter += 1
            f = time.time()
            print('vol rotation time is', o-t)
            print('Finding row skin index time is', g-o)
            print('Cone calculation time is', d-g)
            print('Cone integration in vol time is', f-d)
            print('click target cal time', f-t)
            self.update_CT()

    def clicked_points_index(self, clickedimagesize, angle, clickpointcolumn, rowind):
        '''
        :param clickedimagesize: image size need for image centering befor rotation
        :param angle: angle at which  point need to rerotate
        :param clickpointcolumn: column pixel index at which user clicked at rotated image
        :param rowind: row pixel index at which user clicked at rotated image
        :return: pixel coordinate at not rotated image
        '''
        rotmat = ([np.cos(-angle * np.pi / 180), -np.sin(-angle * np.pi / 180), 0],
                  [np.sin(-angle * np.pi / 180), np.cos(-angle * np.pi / 180), 0],
                  [0, 0, 1])
        movetocentermat = ([1, 0, 0],
                           [0, 1, 0],
                           [(-clickedimagesize[1]) / 2, (-clickedimagesize[0]) / 2, 1])
        backtocornermat = ([1, 0, 0],
                           [0, 1, 0],
                           [(clickedimagesize[1]) / 2, (clickedimagesize[0]) / 2, 1])
        transmat = np.linalg.multi_dot([backtocornermat, rotmat, movetocentermat])
        invtransmat = np.linalg.inv(transmat)
        rotindvector = np.array([int(clickpointcolumn), int(rowind), 1])
        point = np.matmul(transmat, rotindvector)
        return np.matmul(invtransmat, rotindvector)

    def offset_cal(self, vol_shape, vol_dim, angle, axes):
        #input_arr = np.asarray(chopedvol)
        ndim = vol_dim
        axes = axes

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

        c, s = special.cosdg(angle), special.sindg(angle)

        rot_matrix = np.array([[c, s],
                               [-s, c]])
        img_shape = np.asarray(vol_shape)

        axes.sort()
        in_plane_shape = img_shape[axes]
        out_plane_shape = img_shape[axes]
        out_center = rot_matrix @ ((out_plane_shape - 1) / 2) # perform matrix multiplication, same as np.matmul(rot_matrix, ((out_plane_shape - 1)/2)
        in_center = (in_plane_shape - 1) / 2
        offset = in_center - out_center
        return offset

    def update_CT(self):
        """
                    With each user click update CT images with cone at clicked place.
        :return:
        """
        self.imAxial.set_data(self.basevol[self.sliceind.astype(int), :, :].squeeze())
        self.imSagittal.set_data(np.rot90(self.basevol[:, :, self.columnind.astype(int)].squeeze(), k=3))
        self.imCoronal.set_data(self.basevol[:, int(np.round(self.rowind + Cone.skintotargetcenter / self.rowpixelspacing)), :].squeeze())
        self.imAxial.axes.figure.canvas.draw_idle()
        self.imSagittal.axes.figure.canvas.draw_idle()
        self.imCoronal.axes.figure.canvas.draw_idle()

    def cone_volume_integrate(self, volume, center_cone, upper_cone, lower_cone, effectiveR):
        """

        :param lower_cone: indices for the lower cone
        :param center_cone: indices for the center cone
        :param upper_cone: indices for the upper cone
        :param volume: CT volume after rotation in the angle user chose
        :return: CT volume after cones implementation
        """
        '''
        # Removes indices that are out of volume because of volume size reduction
        center_cone = center_cone[center_cone[:, -1] > self.sliceind.astype(int) - effectiveR]
        center_cone = center_cone[center_cone[:, -1] < self.sliceind.astype(int) + effectiveR]
        upper_cone = upper_cone[upper_cone[:, -1] > self.sliceind.astype(int) - effectiveR]
        upper_cone = upper_cone[upper_cone[:, -1] < self.sliceind.astype(int) + effectiveR]
        lower_cone = lower_cone[lower_cone[:, -1] > self.sliceind.astype(int) - effectiveR]
        lower_cone = lower_cone[lower_cone[:, -1] < self.sliceind.astype(int) + effectiveR]
        # slice index correction after volume size reduction
        if self.sliceind - effectiveR < 0:
            pass
        else:
            upper_cone[:, 2] = upper_cone[:, 2] - min(upper_cone[:, 2])
            center_cone[:, 2] = center_cone[:, 2] - min(center_cone[:, 2])
            lower_cone[:, 2] = lower_cone[:, 2] - min(lower_cone[:, 2])
        '''
        # slice index correction after volume size reduction
        center_cone = center_cone[center_cone[:, -1] >= 0]
        center_cone = center_cone[center_cone[:, -1] < len(lstFilesDCM)]
        upper_cone = upper_cone[upper_cone[:, -1] >= 0]
        upper_cone = upper_cone[upper_cone[:, -1] < len(lstFilesDCM)]
        lower_cone = lower_cone[lower_cone[:, -1] >= 0]
        lower_cone = lower_cone[lower_cone[:, -1] < len(lstFilesDCM)]
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
        #find coordinates for cone
        offset = self.offset_cal(self.basevol.shape, self.basevol.ndim, self.angle, axes = (1, 2))
        centerbase_notrotpoint = self.clicked_points_index(self.slice.shape, self.angle,
                                                      column_ind + offset[1],
                                                      row_ind - Cone.distfromskin / self.rowpixelspacing + offset[0])
        upperapex_notrotpoint = self.clicked_points_index(self.slice.shape, self.angle,
                                                     column_ind + offset[1],
                                                     row_ind + Cone.skintouppertarget / self.rowpixelspacing + offset[
                                                         0])
        centerapex_notrotpoint = self.clicked_points_index(self.slice.shape, self.angle,
                                                      column_ind + offset[1],
                                                      row_ind + Cone.skintotargetcenter / self.rowpixelspacing + offset[
                                                          0])
        lowerapex_notrotpoint = self.clicked_points_index(self.slice.shape, self.angle,
                                                     column_ind + offset[1],
                                                     row_ind + Cone.skintolowertarget / self.rowpixelspacing + offset[
                                                         0])
        # pixal world to patient coordinate system
        centerbasepoint = self.voxel_to_patient(self.AffinMatrix, np.array([
            np.round(centerbase_notrotpoint[1]), round(centerbase_notrotpoint[0]),
             slice_ind, 1]))[0:3]
        upperapexpoint = self.voxel_to_patient(self.AffinMatrix, np.array([
            np.round(upperapex_notrotpoint[1]), round(upperapex_notrotpoint[0]),
             slice_ind, 1]))[0:3]
        centeralapexpoint = self.voxel_to_patient(self.AffinMatrix, np.array([
            np.round(centerapex_notrotpoint[1]), np.round(centerapex_notrotpoint[0]),
             slice_ind, 1]))[0:3]
        lowerapexpoint = self.voxel_to_patient(self.AffinMatrix, np.array([
            np.round(lowerapex_notrotpoint[1]), np.round(lowerapex_notrotpoint[0]),
             slice_ind, 1]))[0:3]
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

    def volume_for_cone(self, vol, sliceind, R, angle):
        if sliceind + R > len(lstFilesDCM):
            return self.volume_rotation(vol[sliceind - R: len(lstFilesDCM), :, :], angle), sliceind - R + 20
        elif sliceind - R < 0:
            return self.volume_rotation(vol[0:sliceind + R, :, :], angle), 0
        else:
            return self.volume_rotation(vol[sliceind - R:sliceind + R, :, :], angle), sliceind - R + 20

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


def data_to_dict(drr_vol, ctcutdata, kVp, data_dict):
    """
            Based on DICOM SOP Instance UID.
            Saving running time we save each DRR calculation at different kVp in JSON files.
            Next time user want to use same parameters he will not have to wait for DRR calculation again
    :param drr_vol: DRR volume after calculation
    :param kVp: kVp value in which the calculation was done
    :param data_dict: dictionary that will populate all the information
    :return:
    """
    if keys_exists(data_dict, str(RefDsFirst.SeriesInstanceUID)): # if SOP Instance UID allready proccessed and at requested kVp data was'nt calculated insert that new kVp calculated data to dict ans json
        data_dict[str(RefDsFirst.SeriesInstanceUID)][str(kVp)] = {'DRR': drr_vol}
    else:
        data_dict[str(RefDsFirst.SeriesInstanceUID)] = {}
        data_dict[str(RefDsFirst.SeriesInstanceUID)][str(kVp)] = {'DRR': drr_vol}
        data_dict[str(RefDsFirst.SeriesInstanceUID)]['HU'] = {'HounsfieldUnit': ctcutdata}
    with open('%s.json' % RefDsFirst.SeriesInstanceUID, 'w') as f:
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
    with open('%s.json' % RefDsFirst.SeriesInstanceUID) as file:
        data = file.read()
        data_dict = json.loads(data, object_hook=deconvert)
        b = time.time()
        if keys_exists(data_dict, str(RefDsFirst.SeriesInstanceUID), str(kVp)):
            APData = data_dict[str(RefDsFirst.SeriesInstanceUID)][str(kVp)]['DRR']
            HUdata = data_dict[str(RefDsFirst.SeriesInstanceUID)]['HU']['HounsfieldUnit']
            v = time.time()
            print('Simulation Run time is', v - b)
            return APData, HUdata
        else:
            print('Data for the requested settings is not available, calculation need to take place. '
                  'Please wait until calculation is done. Do not quit Simulation before calculation complete ')
            HUdata = data_dict[str(RefDsFirst.SeriesInstanceUID)]['HU']['HounsfieldUnit']
            APData = cal_data(HUdata, kVp, data_dict)
            return APData, HUdata


def cal_data(dataarray, kVp , data_dict):
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
    ArrayDicomMu = np.interp(dataarray, HUtoMUkVp[:, 1],
                             HUtoMUkVp[:, 0])  # Array based on Linear attenuation coefficient
    air_att_coeff = HUtoMUkVp[0, 0]
    o = time.time()
    APDRR = ap_drr(ArrayDicomMu, Angle, air_att_coeff)
    v = time.time()
    print('Calculation Time is', v - o)
    print('DRR calculation Run time is', v - b)
    data_to_dict(APDRR, dataarray, kVp, data_dict)
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
    ct_scan_volum_increase_for_cone_integration = np.asarray([100, 50])
    if os.path.isfile('/Users/nivravhon/PycharmProjects/pythoProject/%s.json' % RefDsFirst.SeriesInstanceUID):
        AP, ArrayDicomHuCut = get_data(kVp)
        return AP, ArrayDicomHuCut
    else:
        PatientPosition = RefDsFirst.PatientPosition
        ConstPixelDims = (len(lstFilesDCM), int(RefDsFirst.Rows), int(RefDsFirst.Columns))
        ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDsFirst.pixel_array.dtype)
        #ArrayDicomHu = np.zeros(ConstPixelDims, dtype=RefDsFirst.pixel_array.dtype)
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
        ArrayDicomHuCut = find_high_Patient_volume_slice_cut_volume_accordingly(ArrayDicomHu, ct_scan_volum_increase_for_cone_integration)
        data_dict = {}
        AP = cal_data(ArrayDicomHuCut, kVp, data_dict)
        return AP, ArrayDicomHuCut

def find_high_Patient_volume_slice_cut_volume_accordingly(vol, vol_increase):
    s = time.time()
    slice_vol = []
    for i in range(0, np.shape(vol)[0]-1):
        body = BodyMask(vol[i, :, :])
        bodymask = body.mask_body()#create_mask_from_polygon(vol[i, :, :], bodycontour)
        slice_vol.append(np.count_nonzero(bodymask == 1))
    maxvol = max(slice_vol)
    f = time.time()
    highvolslice = slice_vol.index(maxvol)

    print('time to find largest slice in volume', f-s)
    return row_and_col_indices_for_volume_cupping(vol, highvolslice, vol_increase)


def row_and_col_indices_for_volume_cupping(vol, largest_slice_ind, vol_increase):
    s = time.time()
    body = BodyMask(vol[largest_slice_ind, :, :])
    bodymask = body.mask_body()
    first_row_volume_encounter = np.min([i for i in first_nonzero(bodymask, 0) if i > 0])
    first_column_volume_encounter = np.min([i for i in first_nonzero(bodymask, 1) if i > 0])
    last_row_volume_encounter = np.max(last_nonzero(bodymask, 0))
    last_column_volume_encounter = np.max(last_nonzero(bodymask, 1))
    chopedvol = vol[:, first_row_volume_encounter:last_row_volume_encounter,
                first_column_volume_encounter:last_column_volume_encounter]
    s, r, c = chopedvol.shape  # indexes 0:r-1, 0:c-1
    data_vol = np.full((s, r + vol_increase[0], c + vol_increase[1]), -1000)
    data_vol[:, int(vol_increase[0] * 3 / 4 - 1):int(vol_increase[0] * 3 / 4 - 1) + r,
                int(vol_increase[1] / 2 - 1):int(vol_increase[1] / 2 - 1) + c] = chopedvol
    f = time.time()
    print('volume chopping time', f-s)
    return data_vol

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
    Cone = Cone(ArrayDicomHu, ax2, ax3, ax4, trackerAP)
    fig.canvas.mpl_connect('scroll_event', trackerAP.on_scroll)
    fig.canvas.mpl_connect('button_press_event', Cone.on_click)


    #plt.connect('button_press_event', on_click)

    plt.show()


import numpy as np
import sympy
from sympy import Matrix, Symbol, symbols, zeros, ones, eye
import pydicom as dcm
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import scipy


lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk('/Users/nivravhon/CT Scans for AngleSimulation/CT scans from Eric'):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName, filename))

# Get ref file
lstFilesDCM = natsorted(lstFilesDCM)# sorting the files in order
RefDsFirst = dcm.read_file(lstFilesDCM[0])
RefDsLast = dcm.read_file(lstFilesDCM[len(lstFilesDCM) - 1])

ConstPixelDims = (len(lstFilesDCM), int(RefDsFirst.Rows), int(RefDsFirst.Columns))
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDsFirst.pixel_array.dtype)
ArrayDicomHu = np.zeros(ConstPixelDims, dtype=RefDsFirst.pixel_array.dtype)
ArrayDicomMu = np.zeros(ConstPixelDims, dtype=RefDsFirst.pixel_array.dtype)
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dcm.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[lstFilesDCM.index(filenameDCM), :, :] = ds.pixel_array

# converting to Hounsfield Unit
ArrayDicomHu = np.add(np.multiply(np.rot90(ArrayDicom[:, :, :], k=2, axes=(1, 2)), int(RefDsFirst.RescaleSlope)),
                 int(RefDsFirst.RescaleIntercept))  # Array based on hounsfield Unit
ArrayDicomHu[ArrayDicomHu <= -1000] = -1000

fig, ax = plt.subplot()
ax.imshow(ArrayDicomHu[144, :, :], cmap='gray')
#ax1.imshow(ArrayDicomHu[145, :, :], cmap='gray')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
newxlim = xlim*0.1
newylim = ylim*0.1
ax.set_xlim(newxlim)
ax.set_ylim(newylim)
plt.show()
print(xlim)
print(ylim)
'''
# The code below is general (independent of SPMs code)


# premultiplication matrix to go from 0 based to 1 based indexing
one_based = eye(4)
one_based[:3, 3] = (1, 1, 1)
# premult for swapping row and column indices
row_col_swap = eye(4)
row_col_swap[:, 0] = eye(4)[:, 1]
row_col_swap[:, 1] = eye(4)[:, 0]

# various worming matrices
orient_pat = np.zeros((3, 2))
orient_pat[:, 0] = RefDsFirst.ImageOrientationPatient[3:6] # Image orientation Patient column
orient_pat[:, 1] = RefDsFirst.ImageOrientationPatient[0:3] # Image orientation Patient row
orient_cross = np.cross(orient_pat[:, 0], orient_pat[:, 1]) # cross product of the two vectors above
missing_r_col = np.zeros((3, 1))
pos_pat_0 = RefDsFirst.ImagePositionPatient
pos_pat_N = RefDsLast.ImagePositionPatient
pixel_spacing = np.zeros((3, 2))
pixel_spacing[:, :] = RefDsFirst.PixelSpacing
NZ = len(lstFilesDCM)
slice_spacing = Symbol(r'\Delta{s}')

R3 = orient_pat * np.diag(pixel_spacing)
R = zeros(4, 2)
R[:3, :] = R3

# The following is specific to the SPM algorithm.
x1 = ones(4, 1)
y1 = ones(4, 1)
y1[:3, :] = pos_pat_0

to_inv = zeros(4, 4)
to_inv[:, 0] = x1
to_inv[:, 1] = symbols('a b c d')
to_inv[0, 2] = 1
to_inv[1, 3] = 1
inv_lhs = zeros(4, 4)
inv_lhs[:, 0] = y1
inv_lhs[:, 1] = symbols('e f g h')
inv_lhs[:, 2:] = R


def spm_full_matrix(x2, y2):
    rhs = to_inv[:, :]
    rhs[:, 1] = x2
    lhs = inv_lhs[:, :]
    lhs[:, 1] = y2
    return lhs * rhs.inv()


# single slice case
orient = zeros(3, 3)
orient[:3, :2] = orient_pat
orient[:, 2] = orient_cross
x2_ss = Matrix((0, 0, 1, 0))
y2_ss = zeros(4, 1)
y2_ss[:3, :] = orient * Matrix((0, 0, slice_spacing))
A_ss = spm_full_matrix(x2_ss, y2_ss)

# many slice case
x2_ms = Matrix((1, 1, NZ, 1))
y2_ms = ones(4, 1)
y2_ms[:3, :] = pos_pat_N
A_ms = spm_full_matrix(x2_ms, y2_ms)

# End of SPM algorithm

# Rather simpler derivation from DICOM affine formulae - see
# dicom_orientation.rst

# single slice case
single_aff = eye(4)
rot = orient
rot_scale = rot * np.diag(pixel_spacing[:] + (slice_spacing,))
single_aff[:3, :3] = rot_scale
single_aff[:3, 3] = pos_pat_0

# For multi-slice case, we have the start and the end slice position
# patient.  This gives us the third column of the affine, because,
# ``pat_pos_N = aff * [[0,0,ZN-1,1]].T
multi_aff = eye(4)
multi_aff[:3, :2] = R3
trans_z_N = Matrix((0, 0, NZ - 1, 1))
multi_aff[:3, 2] = missing_r_col
multi_aff[:3, 3] = pos_pat_0
est_pos_pat_N = multi_aff * trans_z_N
multi_aff[:3, 2] = np.subtract(pos_pat_N, pos_pat_0)/(NZ-1)
#eqns = tuple(np.subtract(est_pos_pat_N[:3, 0], pos_pat_N))
#solved = sympy.solve(eqns, tuple(missing_r_col))
#multi_aff_solved = multi_aff[:, :]
#multi_aff_solved[:3, 2] = multi_aff_solved[:3, 2].subs(solved)

print(multi_aff)

# Check that SPM gave us the same result
A_ms_0based = A_ms * one_based
A_ms_0based.simplify()
A_ss_0based = A_ss * one_based
A_ss_0based.simplify()
assert single_aff == A_ss_0based
assert multi_aff_solved == A_ms_0based

# Now, trying to work out Z from slice affines
A_i = single_aff
nz_trans = eye(4)
NZT = Symbol('d')
nz_trans[2, 3] = NZT
A_j = A_i * nz_trans
IPP_i = A_i[:3, 3]
IPP_j = A_j[:3, 3]

# SPM does it with the inner product of the vectors
spm_z = IPP_j.T * orient_cross
spm_z.simplify()

# We can also do it with a sum and division, but then we'd get undefined
# behavior when orient_cross sums to zero.
ipp_sum_div = sum(IPP_j) / sum(orient_cross)
ipp_sum_div = sympy.simplify(ipp_sum_div)


# Dump out the formulae here to latex for the RST docs
def my_latex(expr):
    S = sympy.latex(expr)
    return S[1:-1]


print('Latex stuff')
print('   R = ' + my_latex(to_inv))
print('   ')
print('   L = ' + my_latex(inv_lhs))
print()
print('   0B = ' + my_latex(one_based))
print()
print('   ' + my_latex(solved))
print()
print('   A_{multi} = ' + my_latex(multi_aff_solved))
print('   ')
print('   A_{single} = ' + my_latex(single_aff))
print()
print(r'   \left(\begin{smallmatrix}T^N\\1\end{smallmatrix}\right) = A ' + my_latex(trans_z_N))
print()
print('   A_j = A_{single} ' + my_latex(nz_trans))
print()
print('   T^j = ' + my_latex(IPP_j))
print()
print(r'   T^j \cdot \mathbf{c} = ' + my_latex(spm_z))
'''
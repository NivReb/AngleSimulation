import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import json
import pandas as pd
from natsort import natsorted
from scipy import interpolate
from pathlib import Path
import re

def convert(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
        return {"$array": x.tolist()}  # Make a tagged object
    raise TypeError(x)


def deconvert(x):
    if len(x) == 1:  # Might be a tagged object...
        key, value = next(iter(x.items()))  # Grab the tag and value
        if key == "$array":  # If the tag is correct,
            return array(value)  # cast back to array
    return x


def linear_attenuation_coefficient(kVp):
    MeV = (kVp/3)/1000
    csv_files = glob.glob(
        os.path.join('/Users/nivravhon/PycharmProjects/pythoProject/Materials attenuation coefficient raw', "*.csv"))
    csv_files = natsorted(csv_files)
    DensityTable = pd.read_csv('/Users/nivravhon/PycharmProjects/pythoProject/materials density and HU tables/Density.csv')
    MaterialDensity = DensityTable[["Material", "Density"]]
    AvgHUfull = pd.read_csv('/Users/nivravhon/PycharmProjects/pythoProject/materials density and HU tables/Average HU.csv')
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
            Mu_Data[Mu_count +1, 0] = Mu_For_Cal
            Mu_Data[Mu_count, 1] = AvgHU[Mu_count]
            Mu_Data[Mu_count + 1, 1] = AvgHU[Mu_count + 1]
            Mu_count = Mu_count + 2
        else:
            Mu_Data[Mu_count, 0] = Mu_For_Cal
            Mu_Data[Mu_count, 1] = AvgHU[Mu_count]
            Mu_count = Mu_count + 1

        HUtoMU = Mu_Data[np.argsort(Mu_Data[:, 1])]

    return HUtoMU


mu = linear_attenuation_coefficient(60)
mu_sorted = mu[np.argsort(mu[:, 1])]

#f = interpolate.interp1d(mu_sorted[:, 0], mu_sorted[:, 1])
xnew = np.arange(-1000, 3000, 5)
yinterp = np.interp(xnew, mu_sorted[:, 1], mu_sorted[:, 0])

plt.plot(mu_sorted[:, 1], mu_sorted[:, 0], 'o')
plt.plot(xnew, yinterp, '-')
plt.show()

print('done')




'''
kVpRange = np.linspace(40, 220, num=181)
KeV = kVpRange * (1 / 3)
MeV = KeV / 1000
path = os.getcwd()
csv_files = glob.glob(os.path.join('/Users/nivravhon/PycharmProjects/pythoProject/Materials attenuation coefficient raw', "*.csv"))
csv_files = natsorted(csv_files)
DensityTable = pd.read_csv('/Users/nivravhon/PycharmProjects/pythoProject/materials density tables/Density.csv')
MaterialDensity = DensityTable[["Material", "Density"]]
AvgHUfull = pd.read_csv('/Users/nivravhon/PycharmProjects/pythoProject/Average HU.csv')
AvgHU = np.asarray(AvgHUfull["HU"]).astype(float)
Mu_Data = np.zeros((len(kVpRange), len(csv_files)))
Data_count = 0
for f in csv_files:
    # read the csv file into NumPy Array
    df = pd.read_csv(f)
    Val = df.to_numpy()
    Mu_From_Table = np.interp(MeV, Val[:, 0], Val[:, 1])# linear interpolation to find mu for relevant MeV
    DensityInd = np.asarray(np.where(MaterialDensity["Material"] == Path(f).stem)) # find the density for the material in question
    DensityInd = DensityInd[DensityInd != 0]
    Mu_For_Cal = np.multiply(Mu_From_Table, float(np.asarray(MaterialDensity["Density"][DensityInd]))) # getting the mu for the specified MeV after multypling by the density
    Mu_Data[:, Data_count] = Mu_For_Cal # stacking all mu for each material in one np array
    Data_count = Data_count + 1

# creating dictionary for HU vs MU
ind = 0
HUtoMU = {}
for kVp in kVpRange:

    HUtoMU[kVp] = {'HU': AvgHU, 'Mu': Mu_Data[ind, :]}

    ind = ind + 1

json_data = json.dumps(HUtoMU, default=convert)

with open('HU to MU.json', 'w') as f:
    f.write(json.dumps(HUtoMU, default=convert))


#data2 = json.loads(json_data, object_hook=deconvert)
'''





print('done')






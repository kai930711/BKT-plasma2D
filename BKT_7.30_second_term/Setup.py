import numpy as np


def minimum_image_displacement(dR):
    dR = np.asarray(dR, dtype=float)
    return dR - np.rint(dR)


def hardcore_overlap_pair(Ri, Rj, a):
    dR = minimum_image_displacement(Ri - Rj)
    dist2 = np.dot(dR, dR)
    return dist2 < a**2


def hardcore_check(R_new, R_all, a, exclude_index=None):
    R_new = np.asarray(R_new, dtype=float)
    R_all = np.asarray(R_all, dtype=float)

    for j, Rj in enumerate(R_all):
        if exclude_index is not None and j == exclude_index:
            continue

        if hardcore_overlap_pair(R_new, Rj, a):
            return True

    return False
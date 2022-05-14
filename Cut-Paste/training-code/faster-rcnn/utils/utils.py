import numpy as np

def load_accs(accs_file):
    """
    Returns idxs (ndarray), class_accs (ndarray), total_accuracy (float)
    """
    idxs = []
    accs = []
    with open(accs_file) as f:
        for line in f:
            if line.startswith("Class ID:"):
                s = line.split()
                idxs.append(int(s[2]))
                accs.append(float(s[4]))
            elif line.startswith("Total Accuracy:"):
                total = float(line.split()[2])

    assert len(accs) > 0, "No class accuracies found in the file."

    if total is None:
        total = np.mean(accs)

    return np.array(idxs), np.array(accs), total


def computeIOU(bboxA, bboxB):
    xA_1, yA_1, wA, hA = bboxA
    xB_1, yB_1, wB, hB = bboxB

    xA_2 = xA_1 + wA
    yA_2 = yA_1 + hA
    xB_2 = xB_1 + wB
    yB_2 = yB_1 + hB

    # coords of intersecting box
    xI_1 = max(xA_1, xB_1)
    yI_1 = max(yA_1, yB_1)
    xI_2 = min(xA_2, xB_2)
    yI_2 = min(yA_2, yB_2)

    wI = xI_2 - xI_1
    hI = yI_2 - yI_1

    intersection_area = wI * hI if wI > 0 and hI > 0 else 0
    union_area = wA * hA + wB * hB - intersection_area

    if union_area > 0:
        return intersection_area / union_area
    else: # prevent division by 0
        return 0
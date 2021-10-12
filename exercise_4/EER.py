import numpy as np
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


def EER(clients, impostors):
    threshold = np.arange(-1, 1, 0.1)
    print(threshold)
    FAR = []
    FRR = []
    for th in threshold:
        far = 0.0
        for score in impostors:
            if score.item() > float(th):
                far += 1
        frr = 0.0
        for score in clients:
            if score.item() <= float(th):
                frr += 1
        FAR.append(far / impostors.size)
        FRR.append(frr / clients.size)

    ERR = 0.0
    dist = 1.0
    for far, frr in zip(FAR, FRR):
        if abs(far - frr) < dist:
            ERR = (far + frr) / 2
            dist = abs(far - frr)
    return float("{0:.3f}".format(100 * ERR))


# Insert the output file here
filename = "scores_VoxCeleb-1"

lst = open(filename, "r")

scores = []
lines = lst.readlines()
for x in lines:
    scores.append(np.float(x.split()[0]))

c = np.array(scores[0::2])
i = np.array(scores[1::2])

print("EER is : %.3f" % EER(c, i))

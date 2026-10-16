"""MediaPipe FaceMesh (478) landmark subsets used for fitting, with weights."""
JAW = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361]
BROWS = [70, 63, 105, 66, 107, 300, 293, 334, 296, 336]
EYES = [33, 160, 158, 133, 153, 144, 263, 387, 385, 362, 380, 373]
NOSE = [168, 6, 4, 1, 2, 98, 327, 64, 294]
MOUTH = [61, 291, 0, 17, 13, 14, 37, 267, 84, 314]
GROUPS = {"jaw": JAW, "brows": BROWS, "eyes": EYES, "nose": NOSE, "mouth": MOUTH}
WEIGHTS = {"jaw": 1.0, "brows": 0.7, "eyes": 1.5, "nose": 1.0, "mouth": 1.0}
ALL = [i for g in GROUPS.values() for i in g]
def weight_of(i):
    for g, ids in GROUPS.items():
        if i in ids: return WEIGHTS[g]
    return 1.0

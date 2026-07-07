from . import np
import matplotlib.pyplot as plt

def clamp(n, mn, mx):
    return max(mn, min(mx, n))

def get_inset_seeds(points, width, height, margin=0.15):
    xmin = width * margin 
    xmax = width * (1 - margin)
    ymin = height * margin
    ymax = height * (1 - margin)

    valid = []
    invalid = []
    for i, (x , y) in enumerate(points):
        if xmin <= x <= xmax and ymin <= y <= ymax:
            valid.append(i)
        else:
            invalid.append(i)

    return valid, invalid

def invert_dict(d: dict):
    nd = {}
    for key, value in d.items():
        if value not in nd:
            nd[value] = np.array([], dtype=np.int32)
            nd[value] = np.append(nd[value], np.int32(key))
        else:
            nd[value] = np.append(nd[value], np.int32(key))
    
    return nd

def between(n: int | float, lower: int | float, upper: int | float) -> bool:
    return n >= lower and n < upper

def describe_array(x, percentiles=[10, 20, 60, 90, 99], plot=False, name=None):
    if name: print(f'----{name}----')
    print(f"percentiles: {percentiles}:\n", np.percentile(x, percentiles))
    print("mean", x.mean())
    print("max", x.max())
    print("min", x.min())

    if plot:
        fig, ax = plt.subplots()
        ax.hist(x, bins=50)

        plt.show()

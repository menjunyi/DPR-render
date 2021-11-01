import numpy as np
import math

# https://github.com/lmb-freiburg/demon/blob/master/python/depthmotionnet/evaluation/metrics.py
def scale_invariant(depth1, depth2):
    """
    Computes the scale invariant loss based on differences of logs of depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)
    depth1:  one depth map
    depth2:  another depth map
    Returns:
        scale_invariant_distance
    """
    # sqrt(Eq. 3)
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    log_diff = np.log(depth1) - np.log(depth2)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))


# https://github.com/janivanecky/Depth-Estimation/blob/master/eval_depth.py
def LogDepth(depth):
	depth = np.maximum(depth, 1.0 / 255.0)
	return 0.179581 * np.log(depth) + 1

def ScaleInvariantMeanSquaredError(output, gt):

	output = LogDepth(output / 10.0) * 10.0
	gt = LogDepth(gt / 10.0) * 10.0
	d = output - gt
	diff = np.mean(d * d)

	relDiff = (d.sum() * d.sum()) / float(d.size * d.size)
	return diff - relDiff

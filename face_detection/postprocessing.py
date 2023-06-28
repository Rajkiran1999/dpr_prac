import numpy as np

# https://github.com/jrosebr1/imutils/blob/master/imutils/object_detection.py
def non_max_suppression(boxes, confidences, overlap_thresh=0.3):
	"""Non maximum suppression"""
	if len(boxes) == 0:
		return [], []

	valid = []

	boxes = np.asarray(boxes)
	confidences = np.asarray(confidences)

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes
	area = (x2 - x1 + 1) * (y2 - y1 + 1)

	idxs = np.argsort(confidences)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		valid.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the ratio of overlap
		overlap = (xx2 - xx1) * (yy2 - yy1) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlap_thresh)[0])))

	# return only the bounding boxes that are valid
	return boxes[valid], confidences[valid]
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 


from os import listdir
from os.path import isfile, join

cv2.ocl.setUseOpenCL(False)

class ImageStream:

	def __init__(self, ROS=False):

		if not ROS:
			print "OpenCV version :  {0}".format(cv2.__version__)
			self.DIR = 'goodSecond/'			
			self.imageFiles = [f for f in listdir(self.DIR) if isfile(join(self.DIR, f))]
		else:
			pass

	def hasImage(self):
		return len(self.imageFiles) != 0

	def getImage(self):
		if self.hasImage():
			return cv2.imread(self.DIR + self.imageFiles.pop(0), 0)
		return None

def featureTracker(tracker, matcher, currImg, nextImg):
	kp1 = tracker.detect(currImg, None)
	kp2 = tracker.detect(nextImg, None)

	# find the keypoints and descriptors with tracker
	kp1, des1 = tracker.compute(currImg, kp1)
	kp2, des2 = tracker.compute(nextImg, kp2)

	# tCurr = cv2.drawKeypoints(currImg, kp1[:10], None, color=(0,255,0), flags=4)
	# plt.imshow(tCurr), plt.show()

	# tNext = cv2.drawKeypoints(nextImg, kp2[:10], None, color=(0,255,0), flags=4)
	# plt.imshow(tNext), plt.show()

	# Match descriptors.
	matches = matcher.match(des1, des2)
	# Sort them in the order of their distance.
	matches = sorted(matches, key=lambda x:x.distance)

	return kp1, kp2, des1, des2, matches

def flow(matches, kp1, kp2):
	
	dT = 0
	dX = 0
	dY = 0

	matchedkp1 = np.zeros((len(matches), 2))
	matchedkp2 = np.zeros((len(matches), 2))

	for x, match in enumerate(matches):
		i = match.queryIdx
		j = match.trainIdx

		feature1 = kp1[i]
		feature2 = kp2[j]
		
		matchedkp1[x] = np.array(feature1.pt)
		matchedkp2[x] = np.array(feature2.pt)
		
		(x1,y1) = feature1.pt
		(x2,y2) = feature2.pt

		theta1 = feature1.angle
		theta2 = feature2.angle

		dT += theta2 - theta1
		dX += x2 - x1
		dY += y2 - y1

	dT /= len(matches)
	dX /= len(matches)
	dY /= len(matches)

	return -dT, -(dX / 640.0) * 1.5, (dY/360.0) * 0.914, matchedkp1, matchedkp2

def updatePosition(dX, dY, x, y):

	x += dX
	y += dY
	p = np.array([[x], [y]])

	return p




def run():

	plt.ion()
	ax = plt.gca()

	stream = ImageStream(ROS=False)
	# Initiate STAR detector
	orb = cv2.ORB_create()
	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	currImg = stream.getImage()
	
	x = [0.0]
	y = [0.0]

	while stream.hasImage():

		if not stream.hasImage():
			continue
		
		nextImg = stream.getImage()
		kp1, kp2, des1, des2, matches = featureTracker(orb, bf, currImg, nextImg)
		
		dTheta, dX, dY, matchedkp1, matchedkp2 = flow(matches[:30], kp1, kp2)

		position = updatePosition(dX, dY, x[-1], y[-1])
		
		x.append(position[0][0])
		y.append(position[1][0])
		
		ax.plot(x, y)
		plt.pause(0.0001)
		plt.draw()
		img = cv2.drawMatches(currImg,kp1,nextImg,kp2,matches[:30], None, flags=2)
		cv2.imshow(None, img)
		cv2.waitKey(0)

		currImg = nextImg


run()
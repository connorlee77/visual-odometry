import numpy as np 
import cv2 
import matplotlib.pyplot as plt 


from os import listdir
from os.path import isfile, join


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

class ImageStream:

	def __init__(self, ROS=False):

		if not ROS:
			print "OpenCV version :  {0}".format(cv2.__version__)
			self.DIR = 'outdoors/'			
			self.imageFiles = [f for f in listdir(self.DIR) if isfile(join(self.DIR, f))]
		else:
			pass

	def hasImage(self):
		return len(self.imageFiles) != 0

	def getImage(self):
		if self.hasImage():
			return cv2.imread(self.DIR + self.imageFiles.pop(0), 0)
		return None

class DataStream:

	def __init__(self, rate=8.226, ROS=False):
		if not ROS:
			self.DIR = 'outdoors/data/'
			self.angular = np.loadtxt(open(self.DIR + "angular.csv", "rb"), delimiter=",", skiprows=0)[10:]
			self.time = np.loadtxt(open(self.DIR +"time.csv", "rb"), delimiter=",", skiprows=0)	
			self.length = len(self.time)
			self.i = 0
			self.rate = rate

	def hasNext(self):
		return np.round(self.i + self.rate) < self.length

	def getNext(self):
		if self.hasNext():
			self.i += self.rate
			return self.time[np.round(self.i)] - self.time[np.round(self.i) - np.floor(self.rate)], np.mean(self.angular[np.round(self.i) - np.floor(self.rate):np.round(self.i)], axis=0)

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

	return -dT, -(dX / 640.0) * 1.0, (dY/360.0) * 0.8, matchedkp1, matchedkp2

def updatePosition(dX, dY, x, y, theta):

	x += -dY*np.sin(theta) + dX * np.cos(theta)
	y += dX*np.sin(theta) + dY * np.cos(theta)
	p = np.array([[x], [y]])
	
	return p




def run():

	# plt.ion()
	# ax = plt.gca()

	stream = ImageStream(ROS=False)
	dStream = DataStream(rate=8.226, ROS=False)
	# Initiate STAR detector
	orb = cv2.ORB()
	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	currImg = stream.getImage()
	dt, V = dStream.getNext()
	x = [0.0]
	y = [0.0]

	theta = V[2] * dt 
	
	c = 0
	while stream.hasImage() and dStream.hasNext():
		
		nextImg = stream.getImage()
		dt, V = dStream.getNext()
		theta += V[2] * dt 

		kp1, kp2, des1, des2, matches = featureTracker(orb, bf, currImg, nextImg)
		
		dTheta, dX, dY, matchedkp1, matchedkp2 = flow(matches[:30], kp1, kp2)

		position = updatePosition(dX, dY, x[-1], y[-1], theta)
		
		x.append(position[0][0])
		y.append(position[1][0])
		
		# ax.plot(x, y)
		# plt.pause(0.0001)
		# plt.draw()
		print c
		c += 1
		#img = drawMatches(currImg,kp1,nextImg,kp2,matches[:30])

		currImg = nextImg


	plt.plot(x, y)
	plt.title('Path of drone outlining the patch of grass outside Gates Thomas')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()


run()
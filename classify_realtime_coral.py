
from edgetpu.classification.engine import ClassificationEngine
from imutils.video import VideoStream
from imutils.video import FPS
from PIL import Image
import argparse
import imutils
import pickle
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to TensorFlow Lite classification model")
ap.add_argument("-l", "--labels", required=True,
	help="path to labels file")
args = vars(ap.parse_args())

# initialize the labels list
print("[INFO] loading class labels...")
classNames = pickle.loads(open(args["labels"], "rb").read())

# load the Google Coral classification model
print("[INFO] loading Coral model...")
model = ClassificationEngine(args["model"])

vs = VideoStream(src=0).start()
time.sleep(2.0)
print("starting_videostream")
# loop over the frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	orig = frame.copy()
	# prepare the image for classification by converting from a NumPy
	# array to PIL image format
	frame = Image.fromarray(frame)

	# make predictions on the input frame
	results = model.classify_with_image(frame, top_k=1)

	# ensure at least one result was found
	if len(results) > 0:
		# draw the predicted class label and probability on the
		# output frame
		(classID, score) = results[0]
		text = "{}: {:.2f}% ({:.4f} sec)".format(classNames[classID],
			score * 100, model.get_inference_time() / 1000)
		cv2.putText(orig, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 0, 255), 2)

	# show the output frame and wait for a key press
	cv2.imshow("Frame", orig)
	key = cv2.waitKey(1) & 0xFF

	
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()

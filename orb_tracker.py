import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

NUM_OF_TRACKED_FEATURES = 50
NUM_OF_BEST_MATCHES = 10
FRAME_COLOR = (0, 255, 0)

def parse_arguments():
    parser = argparse.ArgumentParser(add_help=True,
                                     description='OpenCV ORB Tracker')
    parser.add_argument('-v', '--input_video',
                        default='sawmovie.mp4',     
                        help='Input video',
                        type=str)
  
    parser.add_argument('-p','--input_photo',
                        default='saw1.jpg', 
                        help='Photo with object that is going to be tracked',
                        type=str)
    
    return parser.parse_args()

def extract_features_from_picture(image):
    """ method that inputs an image and extracts its features
        using ORB Tracker
    Args:
        image (array of points describing an image): image representing an object
        which features are going to be to extracted

    Returns:
        image, keypoints, des: input image, keypoints from image, image descriptors
    """
    # creating orb tracker
    orb = cv2.ORB_create(nfeatures=NUM_OF_TRACKED_FEATURES)
    
    # creating keypoints
    keypoints = orb.detect(image, None)
    keypoints, des = orb.compute(image, keypoints)

    return image, keypoints, des

def track_object_in_video(video_path, image_path):
    """ method that plays video from path and tracks the object
        on video using input image as reference

    Args:
        video_path (str): path to input video
        image_path (str): path to input image
    """
    # creating window
    cv2.namedWindow('Video Playback',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Playback', 1024,768)
    
    # extracting features from input image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    detected_image, detected_keypoints, detected_descriptors = extract_features_from_picture(img)
    
    cap = cv2.VideoCapture(video_path)

    # using Brute-Force matcher from OpenCV library
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # video loop
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Stream ended. Exiting ...")
            break
        
        # extracting features from frame
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_image, frame_keypoints, frame_descriptors = extract_features_from_picture(grayscale_frame)
        
        # creating matches between frame and input image
        matches = bf.match(detected_descriptors, frame_descriptors)
        
        # sorting matches by distance
        matches = sorted(matches, key = lambda x:x.distance)
        matches = matches[:NUM_OF_BEST_MATCHES]

        # assigning values to prepare for searching extreme points from matches
        min_x, max_x = np.int32(frame_keypoints[0].pt).reshape(1,2)[0][0], 0
        min_y, max_y = np.int32(frame_keypoints[0].pt).reshape(1,2)[0][0], 0

        for match in matches:
            keypoint = np.int32(frame_keypoints[match.trainIdx].pt).reshape(1,2)
            if keypoint[0][0] <= min_x:
                min_x = keypoint[0][0]
            if keypoint[0][0] >= max_x:
                max_x = keypoint[0][0]
            if keypoint[0][1] <= min_y:
                min_y = keypoint[0][1]
            if keypoint[0][1] >= max_y:
                max_y = keypoint[0][1]

        # drawing rectangle around tracked object
        grayscale_frame = cv2.cvtColor(grayscale_frame, cv2.COLOR_GRAY2BGR)
        grayscale_frame = cv2.rectangle(grayscale_frame,(min_x, min_y), (max_x,max_y), FRAME_COLOR, 5)

        # displaying image
        cv2.imshow('Video Playback', grayscale_frame)

        # terminating program on pressing 'q' or 'close window'
        if cv2.waitKey(10) == ord('q') or not cv2.getWindowProperty('Video Playback', cv2.WND_PROP_VISIBLE):
            cv2.destroyAllWindows()
            break

    cap.release()

def main(args):
    track_object_in_video(args.input_video, args.input_photo)

if __name__ == "__main__":
    main(parse_arguments())
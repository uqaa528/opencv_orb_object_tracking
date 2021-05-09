import cv2
import numpy as np
from matplotlib import pyplot as plt
def extract_features_from_picture(image):
    orb = cv2.ORB_create(nfeatures=100)
    keypoints = orb.detect(image, None)
    keypoints, des = orb.compute(image, keypoints)
    image = cv2.drawKeypoints(image,keypoints,None)

    return image, keypoints, des

def play_video(video_path, image_path):
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1024,768)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    detected_image, detected_keypoints, detected_descriptors = extract_features_from_picture(img)
    
    cap = cv2.VideoCapture(video_path)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        #grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale_frame = frame
        frame_image, frame_keypoints, frame_descriptors = extract_features_from_picture(grayscale_frame)
        matches = bf.match(detected_descriptors, frame_descriptors)
        matches = sorted(matches, key = lambda x:x.distance)
        matches = matches[:10]

        for match in matches:
            index = match.trainIdx
            grayscale_frame = cv2.circle(grayscale_frame,
            (
                int(frame_keypoints[match.trainIdx].pt[0]), int(frame_keypoints[match.trainIdx].pt[1])
            ),
            radius=15,
            thickness=10, 
            color=(0, 0, 255)
            ) 

        #matching_result = cv2.drawMatches(frame_image, frame_keypoints, detected_image, detected_keypoints, matches[:10], None, flags=2)
        
        cv2.imshow('frame', grayscale_frame)
        if cv2.waitKey(33) == ord('q'):
            break


print(cv2.__version__)
play_video('sawmovie.mp4', 'saw1.jpg')
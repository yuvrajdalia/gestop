import numpy as np
import torch
import pandas as pd
import cv2
import keyboard
from blazebase import resize_pad, denormalize_detections
from blazeface import BlazeFace
from blazepalm import BlazePalm
from blazeface_landmark import BlazeFaceLandmark
from blazehand_landmark import BlazeHandLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

itr=83

#loading the models
palm_detector = BlazePalm().to(gpu)
palm_detector.load_weights("blazepalm.pth")
palm_detector.load_anchors("anchors_palm.npy")
palm_detector.min_score_thresh = .75

hand_regressor = BlazeHandLandmark().to(gpu)
hand_regressor.load_weights("blazehand_landmark.pth")


#opening opncv for capturing live video
WINDOW='test'
cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)
xco=[]
yco=[]
if capture.isOpened():
    hasFrame, frame = capture.read()
    frame_ct = 0
else:
    hasFrame = False

while hasFrame:
    frame_ct +=1
#getting the keypoints
    frame = np.ascontiguousarray(frame[:,::-1,::-1])

    img1, img2, scale, pad = resize_pad(frame)

    normalized_palm_detections = palm_detector.predict_on_image(img1)

    palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)


#visualization of the keypoints and the area of interest
    xc, yc, scale, theta = palm_detector.detection2roi(palm_detections)
    img, affine2, box2 = hand_regressor.extract_roi(frame, xc, yc, theta, scale)
    img=torch.tensor(img,device=gpu)
    flags2, handed2, normalized_landmarks2 = hand_regressor(img)
    landmarks2 = hand_regressor.denormalize_landmarks(normalized_landmarks2, affine2)
    

#selecting the index finger tip keypints among all key points recieved and saving it inside a list
    curr_point_x=-1
    curr_point_y=-1
    for i in range(len(flags2)):
        landmark, flag = landmarks2[i], flags2[i]
        if flag>.5:
            #draw_landmarks(frame, landmark[:,:2], HAND_CONNECTIONS, size=2)
            curr_index_point=landmark[8:9,:2]
            curr_point_x=curr_index_point.tolist()[0][0]
            curr_point_y=curr_index_point.tolist()[0][1]
            if cv2.waitKey(33) == ord('z'):
                xco.append(curr_point_x)
                yco.append(curr_point_y)
                print(curr_point_x,curr_point_y)
#drawing detections
#    draw_roi(frame, box)
    draw_roi(frame, box2)
#    draw_detections(frame, face_detections)
    blank_image2 = 255 * np.ones(shape=[512, 512, 3], dtype=np.uint8)
    draw_detections(frame, palm_detections)
    flag3=0
#plotting the live line which is curvy is nature
    if(len(xco)>=2):
        print(len(xco),len(yco))
        for i in range(1,len(xco)):
            cv2.line(frame,(int(xco[i]),int(yco[i])),(int(xco[i-1]),int(yco[i-1])),(255,0,0),3)
            cv2.line(blank_image2,(int(xco[i]),int(yco[i])),(int(xco[i-1]),int(yco[i-1])),(255,0,0),3)
    cv2.circle(frame,(int(curr_point_x),int(curr_point_y)),4,(0,255,255),thickness=2)
    cv2.imshow(WINDOW, frame[:,:,::-1])
    # cv2.imwrite('sample/%04d.jpg'%frame_ct, frame[:,:,::-1])

#clear out screen
    if cv2.waitKey(33) == ord('c'):
        xco=[]
        yco=[]
#smoothen the curvy line using moving average
    if cv2.waitKey(33) == ord('s'):
        xsr=pd.Series(xco)
        ysr=pd.Series(yco)
        xco=xsr.rolling(5, win_type ='triang').mean()
        yco=ysr.rolling(5, win_type ='triang').mean()
        xco=xco[9:].tolist()
        yco=yco[9:].tolist()
        for i in range(1,len(xco)):
            cv2.line(frame,(int(xco[i]),int(yco[i])),(int(xco[i-1]),int(yco[i-1])),(255,0,0),3)
            cv2.line(blank_image2,(int(xco[i]),int(yco[i])),(int(xco[i-1]),int(yco[i-1])),(255,0,0),3)
#save the drawn figure to an image
    if cv2.waitKey(33) == ord('x'):
        cv2.imwrite(str(itr)+'.jpg', blank_image2)
        itr=itr+1
    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()

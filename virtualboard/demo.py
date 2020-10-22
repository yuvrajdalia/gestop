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


#loading the models
def initialise_models():
    palm_detector = BlazePalm().to(gpu)
    palm_detector.load_weights("models/blazepalm.pth")
    palm_detector.load_anchors("models/anchors_palm.npy")
    palm_detector.min_score_thresh = .75

    hand_regressor = BlazeHandLandmark().to(gpu)
    hand_regressor.load_weights("models/blazehand_landmark.pth")

    return palm_detector,hand_regressor


#opening opncv for capturing live video
def initialise_opencvstream():
    WINDOW='test'
    cv2.namedWindow(WINDOW)
    capture = cv2.VideoCapture(0)
    if capture.isOpened():
        hasFrame, frame = capture.read()
        frame_ct = 0
    else:
        hasFrame = False
    return frame_ct,hasFrame,frame,WINDOW,capture


#generate detections of various points from the input stream of images
def generate_detections(frame,palm_detector):
    frame = np.ascontiguousarray(frame[:,::-1,::-1])

    img1, img2, scale, pad = resize_pad(frame)

    normalized_palm_detections = palm_detector.predict_on_image(img1)

    palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

    return palm_detections,frame


#get the co-ordinates of the keypoints of the palm and detected region of interest containing those keypoints
def get_keypoints_roi(palm_detector,palm_detections,hand_regressor,frame):
    xc, yc, scale, theta = palm_detector.detection2roi(palm_detections)
    
    img, affine2, box2 = hand_regressor.extract_roi(frame, xc, yc, theta, scale)
    
    img=torch.tensor(img,device=gpu)
    
    flags2, handed2, normalized_landmarks2 = hand_regressor(img)
    
    landmarks2 = hand_regressor.denormalize_landmarks(normalized_landmarks2, affine2)

    return flags2,landmarks2,box2

#selecting the index finger tip keypoint among all key points recieved and saving it inside a list
def get_and_store_curr_point(x_coordinates,y_coordinates,flags2,landmarks2):
    curr_point_x=0
    curr_point_y=0
    for i in range(len(flags2)):
        landmark, flag = landmarks2[i], flags2[i]
        if flag>.5:
            #draw_landmarks(frame, landmark[:,:2], HAND_CONNECTIONS, size=2)
            curr_index_point=landmark[8:9,:2]
            curr_point_x=curr_index_point.tolist()[0][0]
            curr_point_y=curr_index_point.tolist()[0][1]
            if cv2.waitKey(33) == ord('z'):
                x_coordinates.append(curr_point_x)
                y_coordinates.append(curr_point_y)
                print(curr_point_x,curr_point_y)
    return curr_point_x,curr_point_y

#plotting the live line from previous stored coordinates which is curvy is nature
def draw_live_line(x_coordinates,y_coordinates,curr_point_x,curr_point_y,frame,blank_image,WINDOW):
    if(len(x_coordinates)>=2):
        print(len(x_coordinates),len(y_coordinates))
        for i in range(1,len(x_coordinates)):
            cv2.line(frame,(int(x_coordinates[i]),int(y_coordinates[i])),(int(x_coordinates[i-1]),int(y_coordinates[i-1])),(255,0,0),3)
            cv2.line(blank_image,(int(x_coordinates[i]),int(y_coordinates[i])),(int(x_coordinates[i-1]),int(y_coordinates[i-1])),(255,0,0),3)
    cv2.circle(frame,(int(curr_point_x),int(curr_point_y)),4,(0,255,255),thickness=2)
    cv2.imshow(WINDOW, frame[:,:,::-1])

def smoothening(x_coordinates,y_coordinates,frame,blank_image):
    xsr=pd.Series(x_coordinates)
    ysr=pd.Series(y_coordinates)
    x_coordinates=xsr.rolling(5, win_type ='triang').mean()
    y_coordinates=ysr.rolling(5, win_type ='triang').mean()
    x_coordinates=x_coordinates[9:].tolist()
    y_coordinates=y_coordinates[9:].tolist()
    return x_coordinates,y_coordinates

def main():

    saved_pic_num=0

    palm_detector,hand_regressor = initialise_models()

    frame_ct,hasFrame,frame,WINDOW,capture = initialise_opencvstream()
    
    x_coordinates=[]
    y_coordinates=[]
    
    while hasFrame:
        frame_ct +=1
        
        palm_detections,frame=generate_detections(frame,palm_detector)


        flags2,landmarks2,box2 = get_keypoints_roi(palm_detector,palm_detections,hand_regressor,frame)
        
        curr_point_x,curr_point_y = get_and_store_curr_point(x_coordinates,y_coordinates,flags2,landmarks2)
        
        draw_roi(frame, box2)

        draw_detections(frame, palm_detections)

        #initialise blank image to be stored in memory upon clicking x
        blank_image = 255 * np.ones(shape=[512, 512, 3], dtype=np.uint8)
        
        draw_live_line(x_coordinates,y_coordinates,curr_point_x,curr_point_y,frame,blank_image,WINDOW)
        

        #Few functionalities which will be triggered upon pressing specific keys
        
        #1.clear out screen
        if cv2.waitKey(33) == ord('c'):
            x_coordinates=[]
            y_coordinates=[]

        #2.smoothen the curvy line using moving average
        if cv2.waitKey(33) == ord('s'):
            x_coordinates,y_coordinates=smoothening(x_coordinates,y_coordinates,frame,blank_image)

        #save the drawn figure to an image
        if cv2.waitKey(33) == ord('x'):
            cv2.imwrite(str(saved_pic_num)+'.jpg', blank_image)
            df=pd.DataFrame()
            df['x-coordinates']=x_coordinates
            df['y-coordinates']=y_coordinates
            df.to_csv('shape'+str(saved_pic_num)+'.csv')
            saved_pic_num=saved_pic_num+1
        
        
        hasFrame, frame = capture.read()
        key = cv2.waitKey(1)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
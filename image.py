import cv2
from predict import Predict

def ip() :
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    predictor = Predict()
    ret, frame = cam.read()
    predictor.inference(frame)
    #while True:
       # ret, frame = cam.read()
        #print(frame.shape)
       # predictor.inference(frame)
        #return a
        # print (a)

    cam.release()
    cv2.destroyAllWindows()

ip()

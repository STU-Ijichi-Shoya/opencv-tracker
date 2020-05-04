import cv2
# import numba
import dlib

# from sample import cv2_tracker_demo


def frame_resize(frame):
    size=1024
    ratio=size/frame.shape[1]
    h=frame.shape[0]*ratio
    h=int(h)
    w=size
    frame=cv2.resize(frame,(w,h))
    return frame


class face_detector_wrapper:
    def detect(self,Frame):
        pass

class cv2_detector(face_detector_wrapper):
    def __init__(self,cas_file:str):
        self.face_cascade = cv2.CascadeClassifier(cas_file)
    def detect(self,Frame,min=10):
        return self.face_cascade.detectMultiScale(Frame, minNeighbors=min)

class dlib_detector(face_detector_wrapper):
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()

    def detect(self,Frame):
        detected_faces = self.face_detector(Frame, 0)
        faces = []
        for face_rect in detected_faces:
            x, y = face_rect.left(), face_rect.top()
            w, h = face_rect.right() - x, face_rect.bottom() - y
            faces.append((x, y, w, h))
        return faces

class Point:
    def __init__(self,x=0.,y=0.):
        self.x=x
        self.y=y

    def __add__(self,other):
        if type(other) is WidthHeight:
            p=Point(self.x+other.w,self.y+other.h)
            return p
        elif type(other) is Point :
            p=Point(self.x+other.x,self.y+other.y)
            return p

class WidthHeight:
    def __init__(self,w=0.,h=0.):
        self.w=w
        self.h=h

class BBox:
    def __init__(self,left:Point=None,right:WidthHeight=None,tuple_boxes:tuple=None):
        if tuple_boxes is not None:
            self.left=Point(tuple_boxes[0],tuple_boxes[1])
            self.right=WidthHeight(tuple_boxes[2],tuple_boxes[3])
        else:
            self.left=left
            self.right=right

    def get_P1(self):
        return self.left

    def get_P2(self):
        return self.left+self.right
    def get_tuple(self)->(int,int,int,int):
        return (int(self.left.x),int(self.left.y),int(self.right.w),int(self.right.h))

def OverWrapArea(P1:Point,P2:Point,P3:Point,P4:Point=None)->float:
    tx=P2.x-P3.x
    ty=P2.y-P3.y
    if tx>P1.x and ty>P1.y:
        area=tx*ty        
    else:
        area=0.
    return area

import math
import random
random.seed(114514)
class Tracking_Person:
    Tracker_num=1
    HANARERU_VALUE=10
    def __init__(self,first_bbox:BBox,frame,name="",tracking_method="CSRT"):
        if name=="":
            self.name="Tracker.No:"+str(Tracking_Person.Tracker_num)
            Tracking_Person.Tracker_num+=1
        else: self.name=name

        self.color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
        self.Bbox=first_bbox

        if tracking_method=="CSRT":
            self.tracking_method=cv2.TrackerCSRT_create()
        elif tracking_method=="TLD":
            self.tracking_method=cv2.TrackerTLD_create()
        elif tracking_method=="MedianFlow":
            self.tracking_method=cv2.TrackerMedianFlow_create()
        else:
            self.tracking_method=cv2.TrackerKCF_create()
        self.frame=frame
        self.tracking_method.init(frame,first_bbox.get_tuple())

        ## 好きに入れられるリスト。
        self.memo=[]

    def update_tracker(self,Frame):
        succsess,trackBbox=self.tracking_method.update(Frame)
        if succsess:
            self.Bbox=BBox(Point(trackBbox[0],trackBbox[1]),WidthHeight(trackBbox[2],trackBbox[3]))

        return succsess

    def get_overWrap(self,detect_BBox:BBox)->float:
        P1=self.Bbox.get_P1()
        P2=self.Bbox.get_P2()

        P3=detect_BBox.get_P1()

        return OverWrapArea(P1,P2,P3)

    def get_point_tuple(self):
        return self.Bbox.get_tuple()
        
    def update_merge(self,bbox:BBox):
        P1=bbox.get_P1()
        sP1=self.Bbox.get_P1()

        if math.sqrt((P1.x-sP1.x)**2+(P1.y-sP1.y)**2)>self.HANARERU_VALUE:
            self.tracking_method.init(self.frame,bbox.get_tuple())



# キャプチャをリリースして、ウィンドウをすべて閉じる
# cap.release()
# cv2.destroyAllWindows()
class face_rec:
    def __init__(self, detector_file_path='haarcascade_frontalface_default.xml', nebor=10):
        self.cas = cv2.CascadeClassifier(detector_file_path)
        self.ne = nebor

    def detect(self,Frame) -> [()]:

        return self.cas.detectMultiScale(Frame, minNeighbors=self.ne)


class Tracker_Contoller:
    def __init__(self,frame,faces:[()],track_method="MedianFlow"):
        self.tracking_mans = [Tracking_Person(BBox(
            tuple_boxes=t), frame, tracking_method=track_method) for t in faces]


    def tracker_updater(self,src,faces: [()]):
        self.i=0
        tracking_mans=self.tracking_mans
        for tp in tracking_mans:
            tp.update_tracker(src)

        for f in faces:
            f = BBox(tuple_boxes=f)
            max_over = 0.
            max_index = 0
            for index, tp in enumerate(tracking_mans):
                area = tp.get_overWrap(f)
                if max_over < area:
                    max_over = area
                    max_index = index
            tracking_mans[max_index].update_merge(f)


    def get_tracker_list(self)->[Tracking_Person]:
        return self.tracking_mans



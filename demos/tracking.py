import cv2
from points_objects import Tracker_Contoller,dlib_detector,cv2_detector

face_cascade_path = 'haarcascade_frontalface_default.xml'
vp = r"anyvideo.mp4"

def demo():

    cap = cv2.VideoCapture(vp)
    face_len = 0
    faces = tuple()

    detector=dlib_detector()
    # detector=cv2_detector(cas_file=face_cascade_path)
    while face_len <= 1:
        r, src = cap.read()
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        faces=detector.detect(src_gray)

        face_len = len(faces)

    ## contoller に　初期座標の登録
    contoller= Tracker_Contoller(src_gray, faces)

    while True:
        r, src = cap.read()
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(src_gray)
        faces= detector.detect(src_gray)
        ## contollerの検出結果とframeをアップデート
        contoller.tracker_updater(src_gray,faces)

        ## 描画処理
        for tp in contoller.get_tracker_list():
            tuple_face = tp.get_point_tuple()
            (x, y, w, h) = tuple_face

            cv2.putText(src, tp.name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(src, (x, y), (x + w, y + h), tp.color, 2)
            # face = src[y: y + h, x: x + w]
            # face_gray = src_gray[y: y + h, x: x + w]


        cv2.imshow("tracking demo", src)
        cv2.waitKey(1)

if __name__ == '__main__':
    demo()
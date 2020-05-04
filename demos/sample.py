import cv2

from points_objects import frame_resize
vp = r"anyvideo.mp4"

def cv2_tracker_demo():
    """
    Tracking手法を選ぶ。適当にコメントアウトして実行する。
    """
    # Boosting
    # tracker = cv2.TrackerBoosting_create()
    # MIL
    # tracker = cv2.TrackerMIL_create()
    # KCF
    # tracker = cv2.TrackerKCF_create()
    # TLD #GPUコンパイラのエラーが出ているっぽい
    # tracker = cv2.TrackerTLD_create()
    # MedianFlow
    # tracker = cv2.TrackerMedianFlow_create()
    trackers = [cv2.TrackerCSRT_create(), cv2.TrackerKCF_create(), cv2.TrackerTLD_create(),
                cv2.TrackerMedianFlow_create()]
    # trackers=[cv2.TrackerKCF_create(),]
    # trackers=[cv2.TrackerMedianFlow_create()]
    trackers_name = ["CSRT", "KCF", "TLD", "MedianFlow"]
    # trackers_name=["CSRT"]
    print(len(trackers))
    colors = [(i, 0, 0) for i in range(0, 256, (255 // len(trackers)))]
    # tracker=cv2.TrackerCSRT_create()
    # GOTURN # モデルが無いよって怒られた
    # https://github.com/opencv/opencv_contrib/issues/941#issuecomment-343384500
    # https://github.com/Auron-X/GOTURN-Example
    # http://cs.stanford.edu/people/davheld/public/GOTURN/trained_model/tracker.caffemodel
    # tracker = cv2.TrackerGOTURN_create()

    cap = cv2.VideoCapture(vp)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = frame_resize(frame)

        bbox = cv2.selectROI(frame, False)
        print(bbox)
        for t in trackers:
            t.init(frame, bbox)
        cv2.destroyAllWindows()
        break
    while True:
        # VideoCaptureから1フレーム読み込む
        ret, frame = cap.read()
        frame = frame_resize(frame)
        if not ret:
            k = cv2.waitKey(1)
            if k == 27:
                break
            continue

        # Start timer
        timer = cv2.getTickCount()

        # トラッカーをアップデートする
        track_boxes = []
        succsess = []
        for t in trackers:
            track, bbox = t.update(frame)
            succsess.append(track)
            track_boxes.append(bbox)

        # FPSを計算する
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        i = 1
        for track, bbox, color, tname in zip(succsess, track_boxes, colors, trackers_name):
            # 検出した場所に四角を書く
            if track:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, color, 2, 1)
                cv2.putText(frame, tname, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            else:
                # トラッキングが外れたら警告を表示する
                cv2.putText(frame, "Failure" + tname, (10, 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                            cv2.LINE_AA);
            i += 1
        # FPSを表示する
        cv2.putText(frame, "FPS : " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA);

        # 加工済の画像を表示する
        cv2.imshow("Tracking", frame)

        # キー入力を1ms待って、k が27（ESC）だったらBreakする

        k = cv2.waitKey(17)
        if k == 27:
            break

if __name__ == '__main__':
    cv2_tracker_demo()
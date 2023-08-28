import cv2
import time
import numpy as np
import sys


def calibrate(save_h = False, period =  100):
    video_feed = cv2.VideoCapture(-1)
    #video_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #video_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    t1 = time.time()

    while True:
        _, img = video_feed.read()
        print(_, end='\r')
        ret, corners = cv2.findChessboardCorners(img, (6, 9), None)
        if ret:
            corners_int = corners.astype(int)
            corners_int_tuple = tuple(map(tuple, corners_int.reshape(-1, 2)))
            src_pts = np.array([corners_int_tuple])
            #src_pts = src_pts.reshape(-1, 2)[[5, 53, 0, 48], :]
            src_pts = src_pts.reshape(-1, 2)[[48, 0, 53, 5], :]
            for i in range(len(src_pts)):
                cv2.circle(img, src_pts[i], 1, (0,0,255), 1)
                cv2.putText(img, str(i), src_pts[i], cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1, cv2.LINE_AA)

        



            x = src_pts[0, 0]
            y = src_pts[0, 1]
            scale = 7
            d_x = 8*scale
            d_y = 5*scale
            y_offset = -50
            x_offset = -100
        
            des_pts = np.array([src_pts[0], [x+d_x,y], [x,y+d_y], [x+d_x,y+d_y]])
            des_pts[:,0] = des_pts[:,0] + x_offset
            des_pts[:,1] = des_pts[:,1] + y_offset
            h, status = cv2.findHomography(src_pts, des_pts)

            warp_img = cv2.warpPerspective(img, h, (img.shape[1], img.shape[0]))

            cv2.imshow('img', warp_img)

            if save_h:
               np.save('h_from_board_function_low', h)
               print('saved h')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        t2 = time.time() - t1
        if t2 > period:
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    video_feed.release()
print(eval(sys.argv[1]))
calibrate(eval(sys.argv[1]))

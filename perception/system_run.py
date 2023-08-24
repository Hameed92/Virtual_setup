import cv2
import numpy as np
import time
from itertools import combinations
import mediapipe as mp
import math
import sys
from calibration import calibrate

def single():
    print('single')
    vid = cv2.VideoCapture(-1)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(2)

    def midpoint(x1, y1, x2, y2):
        return ((x1 + x2)/2, (y1 + y2)/2)

    def point_transform(p, h):
        p = np.array(p, dtype='float32').reshape((1,1,2))
        p_out = cv2.perspectiveTransform(p, h)[0,0,:]
        return p_out


    mp_objectron = mp.solutions.objectron

    fx, fy = (1.0, 1.0)
    px, py = (0.0, 0.0)
    iterations = [1, 2, 5, 6]
    h = np.load('h_from_board.npy')


    axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    t_end = time.time() + 60 * 15
    while time.time() < t_end:
        _, img = vid.read()
        objectron = mp_objectron.Objectron(mp_objectron.Objectron(static_image_mode=True, max_num_objects=5, min_detection_confidence=0.5, model_name='Shoe'))
        results = objectron.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ann_img = img.copy()
        #warp_img = img.copy()
        x = []
        y = []
        z_axis = []
        if not results.detected_objects:
            print('no detected objects', end='\r')
            continue
        for obj in results.detected_objects:
            
            x.append(int(obj.landmarks_2d.landmark[0].x * img.shape[1]))
            y.append(int(obj.landmarks_2d.landmark[0].y * img.shape[0]))

            axis_cam = np.matmul(obj.rotation, 0.1 * axis_world.T).T + obj.translation
            x_ori = axis_cam[..., 0]
            y_ori = axis_cam[..., 1]
            z_ori = axis_cam[..., 2]
        # Project 3D points to NDC space.
            x_ndc = np.clip(-fx * x_ori / (z_ori + 1e-5) + px, -1., 1.)
            y_ndc = np.clip(-fy * y_ori / (z_ori + 1e-5) + py, -1., 1.)
        # Convert from NDC space to image space.
            x_im = np.int32((1 + x_ndc) * 0.5 * img.shape[1])
            y_im = np.int32((1 - y_ndc) * 0.5 * img.shape[0])
            z_axis.append((x_im[3], y_im[3]))

        try:
            x, y = midpoint(x[0], y[0], x[1], y[1])
            x, y = point_transform((x,y), h)
            z0, z1 = midpoint(z_axis[0][0], z_axis[0][1], z_axis[1][0], z_axis[1][1])
            z0, z1 = point_transform((z0, z1), h)

            warp_img = cv2.warpPerspective(ann_img, h, (img.shape[1],img.shape[0]))
            warp_img = cv2.circle(warp_img, (int(x), int(y)), radius=100, color=(0, 0, 255), thickness=5)
            warp_img = cv2.arrowedLine(warp_img, (int(x), int(y)), (int(z0), int(z1)), color=(0, 128, 0), thickness=3, tipLength=0.5)
        except:
            print('not enough points', end='\r')

        try:
            cv2.imshow('ann_img', warp_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            print('wait', end='\r')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    vid.release()





def dual_person():
    print('dual')
    vid = cv2.VideoCapture(-1)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(2)


    def point_transform(p, h):
        p = np.array(p, dtype='float32').reshape((1,1,2))
        p_out = cv2.perspectiveTransform(p, h)[0,0,:]
        return p_out

    def midpoint(center, pair):
        x1 = center[pair[0]][0]
        x2 = center[pair[1]][0]
        y1 = center[pair[0]][1]
        y2 = center[pair[0]][1]
        return ((x1 + x2)/2, (y1 + y2)/2)

    def fetch_value(obj_index):
        return {
            0 : (0,1),
            1 : (0,2),
            2 : (0,3),
            3 : (1,2),
            4 : (1,3),
            5 : (2,3)
        }[obj_index]

    mp_objectron = mp.solutions.objectron

    fx, fy = (1.0, 1.0)
    px, py = (0.0, 0.0)
    iterations = [1, 2, 5, 6]
    h = np.load('h_from_board.npy')


    axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    t_end = time.time() + 60 * 15
    while time.time() < t_end:
        _, img = vid.read()
        objectron = mp_objectron.Objectron(mp_objectron.Objectron(static_image_mode=True, max_num_objects=5, min_detection_confidence=0.5, model_name='Shoe'))
        results = objectron.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ann_img = img.copy()
        #warp_img = img.copy()
        center = []
        z_axis = []
        if not results.detected_objects:
            print('no detected objects', end='\r')
            continue
        for obj in results.detected_objects:
            
            x = (int(obj.landmarks_2d.landmark[0].x * img.shape[1]))
            y = (int(obj.landmarks_2d.landmark[0].y * img.shape[0]))
            center.append((x,y))

            axis_cam = np.matmul(obj.rotation, 0.1 * axis_world.T).T + obj.translation
            x_ori = axis_cam[..., 0]
            y_ori = axis_cam[..., 1]
            z_ori = axis_cam[..., 2]
        # Project 3D points to NDC space.
            x_ndc = np.clip(-fx * x_ori / (z_ori + 1e-5) + px, -1., 1.)
            y_ndc = np.clip(-fy * y_ori / (z_ori + 1e-5) + py, -1., 1.)
        # Convert from NDC space to image space.
            x_im = np.int32((1 + x_ndc) * 0.5 * img.shape[1])
            y_im = np.int32((1 - y_ndc) * 0.5 * img.shape[0])
            z_axis.append((x_im[3], y_im[3]))

        comb = combinations(center, 2)
        comb = list(comb)
        dist = []
        try:
            for i in range(len(comb)):

                p1, p2 = comb[i]
                dist.append(math.dist(p1, p2))

            sorted = dist.copy()
            sorted.sort()
            obj_index = dist.index(sorted[0]), dist.index(sorted[1])
            pairs = fetch_value(obj_index[0]), fetch_value(obj_index[1])
            #second_pair = fetch_value(obj_index[1])
            for pair in pairs:
                x, y = midpoint(center, pair)
                z0, z1 = midpoint(z_axis, pair)
                cv2.circle(ann_img, (int(x), int(y)), radius=60, color=(0, 0, 255), thickness=5)
                cv2.arrowedLine(ann_img, (int(x), int(y)), (int(z0), int(z1)), color=(0, 128, 0), thickness=3, tipLength=3)
            cv2.imshow('img', ann_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        
        except:
            print('not enough points', end='\r')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    vid.release()



def multi():
    print('multi')
    vid = cv2.VideoCapture(-1)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(2)


    def point_transform(p, h):
        p = np.array(p, dtype='float32').reshape((1,1,2))
        p_out = cv2.perspectiveTransform(p, h)[0,0,:]
        return p_out

    def midpoint(center, pair):
        x1 = center[pair[0]][0]
        x2 = center[pair[1]][0]
        y1 = center[pair[0]][1]
        y2 = center[pair[0]][1]
        return ((x1 + x2)/2, (y1 + y2)/2)

    def fetch_value(obj_index):
        return {
            0 : (0,1),
            1 : (0,2),
            2 : (0,3),
            3 : (0,4),
            4 : (0,5),
            5 : (1,2),
            6 : (1,3),
            7 : (1,4),
            8 : (1,5),
            9 : (2,3),
            10 : (2,4),
            11: (2,5),
            12 : (3,4),
            13 : (3,5),
            14 : (4,5),
        }[obj_index]

    mp_objectron = mp.solutions.objectron

    fx, fy = (1.0, 1.0)
    px, py = (0.0, 0.0)
    iterations = [1, 2, 5, 6]
    h = np.load('h_from_board.npy')


    axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    t_end = time.time() + 60 * 15
    while time.time() < t_end:
        _, img = vid.read()
        objectron = mp_objectron.Objectron(mp_objectron.Objectron(static_image_mode=True, max_num_objects=5, min_detection_confidence=0.5, model_name='Shoe'))
        results = objectron.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ann_img = img.copy()
        #warp_img = img.copy()
        center = []
        z_axis = []
        if not results.detected_objects:
            print('no detected objects', end='\r')
            continue
        for obj in results.detected_objects:
            
            x = (int(obj.landmarks_2d.landmark[0].x * img.shape[1]))
            y = (int(obj.landmarks_2d.landmark[0].y * img.shape[0]))
            center.append((x,y))

            axis_cam = np.matmul(obj.rotation, 0.1 * axis_world.T).T + obj.translation
            x_ori = axis_cam[..., 0]
            y_ori = axis_cam[..., 1]
            z_ori = axis_cam[..., 2]
        # Project 3D points to NDC space.
            x_ndc = np.clip(-fx * x_ori / (z_ori + 1e-5) + px, -1., 1.)
            y_ndc = np.clip(-fy * y_ori / (z_ori + 1e-5) + py, -1., 1.)
        # Convert from NDC space to image space.
            x_im = np.int32((1 + x_ndc) * 0.5 * img.shape[1])
            y_im = np.int32((1 - y_ndc) * 0.5 * img.shape[0])
            z_axis.append((x_im[3], y_im[3]))

        comb = combinations(center, 2)
        comb = list(comb)
        dist = []
        try:
            for i in range(len(comb)):

                p1, p2 = comb[i]
                dist.append(math.dist(p1, p2))

            sorted = dist.copy()
            sorted.sort()
            obj_index = dist.index(sorted[0]), dist.index(sorted[1])
            pairs = fetch_value(obj_index[0]), fetch_value(obj_index[1]), fetch_value(obj_index[2])
            #second_pair = fetch_value(obj_index[1])
            for pair in pairs:
                x, y = midpoint(center, pair)
                z0, z1 = midpoint(z_axis, pair)
                cv2.circle(ann_img, (int(x), int(y)), radius=60, color=(0, 0, 255), thickness=5)
                cv2.arrowedLine(ann_img, (int(x), int(y)), (int(z0), int(z1)), color=(0, 128, 0), thickness=3, tipLength=3)
            cv2.imshow('img', ann_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        
        except:
            print('not enough points', end='\r')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    vid.release()

if sys.argv[2] == '1':
    calibrate(save_h=True)

if sys.argv[1] == '1':
    single()
elif sys.argv[1] == '2':
    dual_person()
else:
    multi()
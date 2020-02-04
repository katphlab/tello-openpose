import cv2
import os, sys
sys.path.append('./pose_extractor')

from sys import platform
from collections import namedtuple
import numpy as np
from FPS import FPS
from modules.inference_engine_pytorch import InferenceEnginePyTorch
from modules.parse_poses import parse_poses
from modules.draw import draw_poses

body_kp_id_to_name = {
    0: "Neck",
    1: "Nose",
    2: "BodyCenter",
    3: "LShoulder",
    4: "LElbow",
    5: "LWrist",
    6: "LHip",
    7: "LKnee",
    8: "LAnkle",
    9: "RShoulder",
    10: "RElbow",
    11: "RWrist",
    12: "RHip",
    13: "RKnee",
    14: "RAnkle",
    15: "REye",
    16: "LEye",
    17: "REar",
    18: "LEar"}

body_kp_name_to_id = {v: k for k, v in body_kp_id_to_name.items()}

# We will save the edges as pair of detected keypoints
Pair = namedtuple('Pair', ['p1', 'p2', 'color'])
color_right_side = (0,255,0)
color_left_side = (0,0,255)
color_middle = (0,255,255)
color_face = (255,255,255)

pairs_head = [
    Pair("Nose", "REye", color_right_side),
    Pair("Nose", "LEye", color_left_side),
    Pair("REye", "REar", color_right_side),
    Pair("LEye", "LEar", color_left_side)
]

pairs_upper_limbs = [
    Pair("Neck", "RShoulder", color_right_side),
    Pair("RShoulder", "RElbow", color_right_side),
    Pair("RElbow", "RWrist", color_right_side),
    Pair("Neck", "LShoulder", color_left_side),
    Pair("LShoulder", "LElbow", color_left_side),
    Pair("LElbow", "LWrist", color_left_side)
]

pairs_lower_limbs = [
    Pair("BodyCenter", "RHip", color_right_side),
    Pair("RHip", "RKnee", color_right_side),
    Pair("RKnee", "RAnkle", color_right_side),
    Pair("BodyCenter", "LHip", color_left_side),
    Pair("LHip", "LKnee", color_left_side),
    Pair("LKnee", "LAnkle", color_left_side),
]

pairs_spine = [
    Pair("Nose", "Neck", color_middle),
    Pair("Neck", "BodyCenter", color_middle)
]

pairs_body = pairs_head + pairs_upper_limbs + pairs_lower_limbs + pairs_spine

class PoseWrapper:
    @staticmethod
    def distance_kps(kp1,kp2):
        # kp1 and kp2: numpy array of shape (3,): [x,y,conf]
        x1,y1,c1 = kp1
        x2,y2,c2 = kp2
        if c1 > 0 and c2 > 0:
            return np.linalg.norm(kp1[:2]-kp2[:2])
        else: 
            return 0

    def __init__(self, draw_render=False):
        self.draw_render = draw_render
        
        self.net = InferenceEnginePyTorch('human-pose-estimation-3d.pth', 'GPU')        
        
        
    def eval(self, frame):
        self.frame = frame

        base_height = 256
        scale = base_height/self.frame.shape[0]
        scaled_img = cv2.resize(self.frame, dsize=None, fx=scale, fy=scale)
        inference_result = self.net.infer(scaled_img)
        poses_3d, poses_2d = parse_poses(inference_result, scale, 8, 1)

        if self.draw_render:
            draw_poses(self.frame, poses_2d)

        if poses_2d.shape[0] != 0: # When no person is detected, shape = (), else (nb_persons, 25, 3)
            self.body_kps = np.array([np.array(poses_2d[pose_id][0:-1]).reshape((-1, 3)) for pose_id in range(len(poses_2d))])


            # We sort persons by their an "estimation" of their size
            # size has little to do with the real size of a person, but is a arbitrary value, here, calculated as distance(Nose, Neck) + 0.33*distance(Neck,Midhip)
            sizes = np.array([self.length(pairs_spine, person_idx=i, coefs=[1, 0.33]) for i in range(self.body_kps.shape[0])])

            # Sort from biggest size to smallest
            order = np.argsort(-sizes)
            sizes = sizes[order]
            self.body_kps = self.body_kps[order]

            # Keep only the biggest person
            self.body_kps = self.body_kps[0]

            self.nb_persons = 1
        else:
            self.nb_persons = 0
            self.body_kps = []
        
        return self.nb_persons,self.body_kps

    def get_body_kp(self, kp_name="Neck"):
        """
            Return the coordinates of a keypoint named 'kp_name' of the person of index 'person_idx' (from 0), or None if keypoint not detected
        """
        try:
            x,y,conf = self.body_kps[body_kp_name_to_id[kp_name]]
        except:
            print(f"get_body_kp: invalid kp_name '{kp_name}'")
            return None
        if conf > 0:
            return (int(x),int(y))
        else:
            return None
    
    def length(self, pairs, person_idx=0, coefs = None):
        """
            Calculate the mean of the length of the pairs in the list 'pairs' for the person of index 'person_idx' (from 0)
            If one (or both) of the 2 points of a pair is missing, the number of pairs used to calculate the average is decremented of 1
        """
        if coefs is None:
            coefs = [1] * len(pairs)

        person = self.body_kps[person_idx]

        l_cum = 0
        n = 0
        for i,pair in enumerate(pairs):
            l = self.distance_kps(person[body_kp_name_to_id[pair.p1]], person[body_kp_name_to_id[pair.p2]])
            if l != 0:
                l_cum += l * coefs[i]
                n += 1
        if n>0:
            return l_cum/n
        else:
            return 0
        
    


if __name__ == '__main__' :
    # Read video
    video=cv2.VideoCapture('v1.mp4')
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    ok, frame = video.read()
    h,w,_=frame.shape
    # if args.output:
    #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    #     out=cv2.VideoWriter(args.output,fourcc,30,(w,h))

    my_op = PoseWrapper(draw_render=True)

    fps = FPS()
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        fps.update()
        frame = frame.copy()
        nb_persons,body_kps = my_op.eval(frame)
        
            
        fps.display(frame)
        cv2.imshow("Rendering", frame)
        # if args.output:
        #     out.write(frame)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
        elif k== 32: # space
            cv2.waitKey(0)


    video.release()
    cv2.destroyAllWindows()
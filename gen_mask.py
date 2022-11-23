import dlib
import numpy as np
from numpy import imag
import os 
import cv2
import secrets
import tensorflow as tf

from mask_the_face.utils.aux_functions import *

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def build_gen_mask(path_to_dlib_model, from_cv2=False):
    if path_to_dlib_model is None:
        return None

    # already random
    args = Namespace(code='', color='#0473e2', color_weight=0.5, feature=False, mask_type='random',
                     path='', pattern='', pattern_weight=0.5, verbose=False, write_original_image=False)

    args.detector = dlib.get_frontal_face_detector()

    # if not os.path.exists(path_to_dlib_model):
    #     download_dlib_model()

    args.predictor = dlib.shape_predictor(path_to_dlib_model)
    print('Loaded gen mask model')
    mask_code = "".join(args.code.split()).split(",")
    args.code_count = np.zeros(len(mask_code))
    args.mask_dict_of_dict = {}
    for i, entry in enumerate(mask_code):
        mask_dict = {}
        mask_color = ""
        mask_texture = ""
        mask_type = entry.split("-")[0]
        if len(entry.split("-")) == 2:
            mask_variation = entry.split("-")[1]
            if "#" in mask_variation:
                mask_color = mask_variation
            else:
                mask_texture = mask_variation
        mask_dict["type"] = mask_type
        mask_dict["color"] = mask_color
        mask_dict["texture"] = mask_texture
        args.mask_dict_of_dict[i] = mask_dict
    available_mask_types = get_available_mask_types()
    
    def func_gen_mask(image):
        image_ori = image
        image = np.array(image, np.uint8)
        masked_image, _, _, _ = mask_image(image, args, available_mask_types)
        masked_image = tf.cast(masked_image, tf.float32)
        if (len(masked_image) == 0):
            return tf.cast(image_ori, tf.float32)
        return masked_image[0]

    def func_gen_mask_cv2(image):
        masked_image, _, _, _ = mask_image(image, args, available_mask_types)
        if (len(masked_image) == 0):
            return image
        return masked_image[0]

    return func_gen_mask_cv2 if from_cv2 else func_gen_mask

# if __name__ == '__main__':
#     path_to_dlib_model = 'download/shape_predictor_68_face_landmarks.dat'
#     tool_gen_mask = build_gen_mask(path_to_dlib_model)
#     img = cv2.imread("/home/lap14880/hieunmt/tf_mask_gen/unzip/VN-celeb/1/0.png")
#     print(img)
#     cv2.imwrite("facemask_input.jpg",img)
#     img = tool_gen_mask(img)
#     print(img)
#     cv2.imwrite("facemask_output.jpg",img)

import os
import shutil
import multiprocessing
import cv2

from gen_mask import *

def make_mask_dataset(route, to_des, im_size, tool_gen_mask, n_mask):
    """
    using multiprocessing
    input:
        route to main directory and phrase ("train", "valid", "test")
        or just route to the directory that its subfolder are classes
    output:
        X_path: path to img
        Y_int: int label
        all_class: list of string class name
    """

    if tool_gen_mask is None:
        print('tool_gen_mask is None')
        return

    global task

    def task(route, list_cls, to_des, im_size):
        print('Start task')

        sign = 'mask_' + route.split('/')[-1] 

        for cl in list_cls:
            path2cl = os.path.join(route, cl)

            if len(os.listdir(path2cl)) < 1:
                continue

            des_class = os.path.join(to_des, sign + '_' + cl)

            try:
                os.mkdir(des_class)
            except:
                pass

            for imfile in os.listdir(path2cl):
                impath = os.path.join(path2cl, imfile)

                for i in range(n_mask):
                    try:
                        img = cv2.imread(impath)
                        img = cv2.resize(img, (im_size, im_size))
                        img = tool_gen_mask(img)
                        imsave = os.path.join(des_class, f"{i}_" + imfile)
                        cv2.imwrite(imsave, img)
                    except:
                        print(i, impath)

        print('Finish')

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    processes = []

    all_class = sorted(os.listdir(route))
    n_labels = len(all_class)
    n_per = int(n_labels // cpu_count + 1)

    for i in range(cpu_count):
        print(f'Start cpu {i}')

        start_pos = i * n_per
        end_pos = (i + 1) * n_per
        list_cls = all_class[start_pos:end_pos]
     
        p = pool.apply_async(task, args=(route,list_cls,to_des,im_size))
        processes.append(p)

    result = [p.get() for p in processes]

    pool.close()
    pool.join()

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=""

    to_des = 'dataset'
    im_size = 160
    path_to_dlib_model = 'download/shape_predictor_68_face_landmarks.dat'
    n_mask = 3

    tool_gen_mask = build_gen_mask(path_to_dlib_model, from_cv2=True)

    try:
        shutil.rmtree(to_des)
    except:
        pass

    try:
        os.mkdir(to_des)
    except:
        pass

    route = 'unzip/gnv_dataset'
    make_mask_dataset(route, to_des, im_size, tool_gen_mask, n_mask)

    route = 'unzip/VN-celeb'
    make_mask_dataset(route, to_des, im_size, tool_gen_mask, n_mask)

    route = 'unzip/glint360k_224'
    make_mask_dataset(route, to_des, im_size, tool_gen_mask, n_mask)
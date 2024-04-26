import dlib, os

# 脸部检测器
def get_front_face_detector():
    return dlib.get_frontal_face_detector()

# 脸部关键点预测器
def get_landmarks_predictor(path):
    if os.path.exists(path) and path.endswith('.dat'):
        return dlib.shape_predictor(path)
    else:
        raise ValueError('{} is not valid...'.format(path))
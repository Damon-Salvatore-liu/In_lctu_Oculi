import argparse


from .utils.solu_base import Solu
from .models.blink_net import BlinkCNN
from .trainer import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_vid_path', type=str)
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()
    return args


def main(input_vid_path, out_dir):
    solution = Solu(input_vid_path)
    
    model = BlinkCNN()
    
    trainer = Trainer('./configs/blink_cnn.yml', model)
    
    for i in range(solution.frame_num):
        print('Frame:' + str(i))
        eye1, eye2 = solution.get_eye_by_fid(i)
        
        
        
    
    











if __name__ == "__main__":
    pass
import os, sys
import importlib
from model.gan_div import Model
this_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_dir, 'configs'))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='train 5x5 gaussian.')
    parser.add_argument('--cfg', type=str, default='config_grid', help='config file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cfg_file = importlib.import_module(args.cfg)
    cfg_dict = cfg_file.cfg_dict

    if not os.path.exists(cfg_dict['IMG_DIR']):
        os.makedirs(cfg_dict['IMG_DIR'])
    if not os.path.exists(cfg_dict['MODEL_DIR']):
        os.makedirs(cfg_dict['MODEL_DIR'])

    GAN = Model(cfg_dict)
    GAN.train()


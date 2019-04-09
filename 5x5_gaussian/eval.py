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
    parser.add_argument('-n', '--name', type=str, required=True, help='model name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cfg_file = importlib.import_module(args.cfg)
    cfg_dict = cfg_file.cfg_dict

    # build model
    GAN = Model(gan_config)

    # evalaute
    std = np.sqrt(gan_config['dataset']['sig'])
    th_list = [std, 2*std, 3*std]

    index_list = np.arange(5000, 100000, 5000)
    for index in index_list:
        MODEL_NAME = os.path.join(gan_config['MODEL_DIR'], 'model_{0}.pth'.format(index))
        print(MODEL_NAME)
        for th in th_list:
            GAN.evaluate_quality(MODEL_NAME, th)
        GAN.evaluate_mode(MODEL_NAME, th_list[0])


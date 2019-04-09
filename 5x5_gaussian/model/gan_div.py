import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

import os, sys
this_dir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(this_dir, '..', 'lib'))
from datasets.loadDataset import getDataset
from datasets.sampler import loadSampler
from graph.nnloader import loadNN, loadOpt
from graph.mlp import Weight_G
from utils.visualizer import *
from utils.evaluator import check_mode, check_quality

sys.path.append(os.path.join(this_dir, '..', '..'))
from ndiv_pytorch import NDiv_loss

class Model(object):
    def __init__(self, config):
        self.epoch = config['epoch']
        self.batch_size = config['batch_size']
        self.alpha = config['alpha']

        #load dataset
        self.dataset = getDataset(config['dataset'])
        self.data_points = self.dataset.get_data_points()
        self.num_modes = self.dataset.get_num_modes()
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.load_model = config['load_model']
        self.use_gpu = config['use_gpu']
        self.g_step = config['g_step']
        self.div_step = config['div_step']
        self.d_step = config['d_step']
        self.show_step = config['show_step']
        self.ndiv_loss_ratio = config['ndiv_loss_ratio']

        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['div_loss'] = []
        self.g_dists=[]
        self.z_dists=[]
        self.device = torch.device("cuda:0" if self.use_gpu and torch.cuda.is_available() else "cpu")
    
        #load sampler
        self.z_sampler = loadSampler(config['sampler'], dataset=self.dataset)
        if self.z_sampler.name == "bourgain":
            self.alpha, self.z_dim = self.z_sampler.scale, self.z_sampler.embedded_data.shape[1]
        else:
            self.z_dim = config['sampler']['dim']
        
        #load netowork
        self.G = loadNN(config['nn_config_G'], input_size=self.z_dim, output_size=self.dataset.out_dim)
        self.D = loadNN(config['nn_config_D'], input_size=self.dataset.out_dim, output_size=1)

        #load optimizer
        self.G_opt = loadOpt(self.G.parameters(), config['opt_config_G'])
        self.D_opt = loadOpt(self.D.parameters(), config['opt_config_D'])

        #load criterion
        self.criterion = nn.BCELoss()
        self.criterion_mse = nn.MSELoss()

        #convert device
        self.G.to(self.device)
        self.D.to(self.device)

        # output
        self.img_dir = config['IMG_DIR']
        self.model_dir = config['MODEL_DIR']

    def evaluate_mode(self, model_name, th):
        model = torch.load(model_name)
        self.G.load_state_dict(model['G'])
        self.G.eval()

        count_list = []
        for i in range(10):
            mode = np.zeros(self.num_modes)
            for j in range(100):
                latent_samples = torch.rand(250, self.z_dim).to(dtype=torch.float32, device=self.device)
                output = self.G(latent_samples)
                output = output.detach().cpu().numpy()
                mode_ = check_mode(output, self.data_points, th)
                mode = np.logical_or(mode, mode_)
                count = mode.sum()
            count_list.append(count)
        count_list = np.array(count_list)
        num_modes = count_list
        print('th = {0:.2f}, #mode = {1:.2f}, std = {2:.2f}'.format(th, np.mean(num_modes), np.std(num_modes)))

    def evaluate_quality(self, model_name, th):
        model = torch.load(model_name)
        self.G.load_state_dict(model['G'])
        self.G.eval()

        count_list = []
        for i in range(10):
            count = 0
            for j in range(100):
                latent_samples = torch.rand(250, self.z_dim).to(dtype=torch.float32, device=self.device)
                output = self.G(latent_samples)
                output = output.detach().cpu().numpy()
                num = check_quality(output, self.data_points, th)
                count = count + num
            count_list.append(count)
        count_list = np.array(count_list)
        quality = 100.0 * count_list / 25000
        print('th = {0:.2f}, success_rate = {1:.2f}, std = {2:.2f}'.format(th, np.mean(quality), np.std(quality)))


    def train(self, iters=0):
        if self.load_model:
            model = torch.load('model.pth')
            self.G.load_state_dict(model['G'])
            self.D.load_state_dict(model['D'])
            self.G_opt.load_state_dict(model['G_optm'])
            self.D_opt.load_state_dict(model['D_optim'])
        print('training start')
        pbar = range(self.epoch)
        visual_z = False
        if iters != 0:
            pbar = range(iters)
        for ep in tqdm(pbar):
            if ep % self.show_step == 0:
                real_data, fake_data = visualize_G(ep, self.img_dir, self, 5000, dim1=0, dim2=1, latent_dim = self.z_dim, visual_z = visual_z)

            # Step 1 Train D.
            for d_index in range(self.d_step):
                # Train D
                self.D.zero_grad()

                # Train D on real
                real_samples = next(iter(self.dataloader))
                if isinstance(real_samples, list):
                    real_samples = real_samples[0]
                d_real_data = real_samples.to(dtype=torch.float32, device=self.device)

                d_real_decision = self.D(d_real_data)
                labels = torch.ones(d_real_decision.shape, dtype=torch.float32, device=self.device)
                
                d_real_loss = self.criterion(d_real_decision, labels)

                # Train D on fake
                latent_samples = torch.rand(self.batch_size, self.z_dim).to(dtype=torch.float32, device=self.device)
                d_fake_data = self.G(latent_samples)
                d_fake_decision = self.D(d_fake_data)
                labels = torch.zeros(d_fake_decision.shape, dtype=torch.float32, device=self.device)
     
                d_fake_loss = self.criterion(d_fake_decision, labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_loss_np = d_loss.item()
                self.D_opt.step()
                self.train_hist['D_loss'].append(d_loss_np)

            # Step 2 update g w. diversity. 
            for div_index in range(self.div_step):    
                self.G.zero_grad()
                latent_samples = torch.rand(self.batch_size, self.z_dim).to(dtype=torch.float32, device=self.device)

                g_fake_data = self.G(latent_samples)
                g_fake_decision = self.D(g_fake_data)
                labels = torch.ones(g_fake_decision.shape, dtype=torch.float32, device=self.device)
                div_loss_val = NDiv_loss(latent_samples, g_fake_data, self.alpha) * self.ndiv_loss_ratio

                div_loss_val.backward()
                div_loss_np = div_loss_val.item()
                self.G_opt.step()
                self.train_hist['div_loss'].append(div_loss_val.item())        

            # Step 3 update g. 
            for g_index in range(self.g_step):
                self.G.zero_grad()
                latent_samples = torch.rand(self.batch_size, self.z_dim).to(dtype=torch.float32, device=self.device)
                g_fake_data = self.G(latent_samples)
                g_fake_decision = self.D(g_fake_data)
                labels = torch.ones(g_fake_decision.shape, dtype=torch.float32, device=self.device)
                gan_loss = self.criterion(g_fake_decision, labels)

                total_loss = gan_loss
                total_loss.backward()
                self.G_opt.step()

                g_loss_np = gan_loss.item()
                self.train_hist['G_loss'].append(g_loss_np)

            if ep % 5000 == 4999 and ep > 0:
                mname = os.path.join(self.model_dir, 'model_{0}.pth'.format(ep+1))
                torch.save({'G':self.G.state_dict(),'D':self.D.state_dict(),
                            'G_optm':self.G_opt.state_dict(),'D_optim':self.D_opt.state_dict()}, mname)
            if ep % self.show_step == 0:
                cuda = self.use_gpu
                loss_d_real = float(d_real_loss.item())
                loss_d_fake = float(d_fake_loss.item())

                msg = 'Iteration {}: D_loss(real/fake): {}/{} G_loss: {}, Div_loss: {}'.format(ep, loss_d_real, loss_d_fake, g_loss_np, float(div_loss_np))
                print(msg)


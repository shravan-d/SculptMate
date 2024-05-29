from .data import get_pifu_input, read_image_from_path
from .model import HGPIFuNetwNML, HGPIFuMRNet
from .gen_mesh import gen_mesh
import torch
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class PifuGenerator():
    def __init__(self, device):
        self.state_dict_path = ROOT_DIR + '/checkpoints/pifuhd.pt'
        self.projection_mode = 'orthogonal'
        self.device = device
        self.netMR = None

    def initiate_model(self):
        if self.netMR is None:
            state_dict = None
            if os.path.exists(self.state_dict_path):
                state_dict = torch.load(self.state_dict_path, map_location=self.device)
                self.opt = state_dict['opt']
            else:
                raise Exception('Unable to find state dictionary!', self.state_dict_path)

            try:
                self.opt_netG = state_dict['opt_netG']
                self.netG = HGPIFuNetwNML(self.opt_netG, self.projection_mode).to(device=self.device)
                self.netMR = HGPIFuMRNet(self.opt, self.netG, self.projection_mode).to(device=self.device)
                self.netMR.load_state_dict(state_dict['model_state_dict'])
            except Exception as e:
                print('[Model Uno Initialization Error]', e)
                return 2
            return 0

    
    def generate_mesh(self, input_image=None, input_name='', img_path='', scale=1):
        if self.netMR is None:
            return 1
        try:
            if input_image is None:
                input_image, input_name = read_image_from_path(img_path=img_path)
            with torch.no_grad():
                self.netG.eval()
                pifu_input = get_pifu_input(image=input_image, scale=scale, img_name=input_name)
                gen_mesh(self.opt.resolution, self.netMR, self.device, pifu_input, components=self.opt.use_compose)
            print('Generation Complete')
            return 0
        except Exception as e:
            print('[TripoSR Generation Error]', e)
            return 2

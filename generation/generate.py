from .data import get_pifu_input, read_image_from_path
from .model import HGPIFuNetwNML, HGPIFuMRNet
from .gen_mesh import gen_mesh
import torch
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

state_dict_path = ROOT_DIR + '/checkpoints/pifuhd.pt'
projection_mode = 'orthogonal'


def generate_mesh(device, input_image=None, input_name='', img_path='', scale=1):
    print('Computations running on', device)
    
    state_dict = None
    if os.path.exists(state_dict_path):
        print('Updating checkpoint: ', state_dict_path)
        state_dict = torch.load(state_dict_path, map_location=device)
        opt = state_dict['opt']
    else:
        raise Exception('Unable to find state dictionary!', state_dict_path)

    opt_netG = state_dict['opt_netG']
    netG = HGPIFuNetwNML(opt_netG, projection_mode).to(device=device)
    netMR = HGPIFuMRNet(opt, netG, projection_mode).to(device=device)
    netMR.load_state_dict(state_dict['model_state_dict'])
    
    with torch.no_grad():
        netG.eval()
        if input_image is None:
            input_image, input_name = read_image_from_path(img_path=img_path)
        pifu_input = get_pifu_input(image=input_image, scale=scale, img_name=input_name)
        gen_mesh(opt.resolution, netMR, device, pifu_input, components=opt.use_compose)

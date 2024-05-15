import numpy as np
import torch
from skimage import measure
from .sdf import create_grid, eval_grid_octree, eval_grid
import bpy


def reconstruction(net, cuda, calib_tensor,
                   resolution, b_min, b_max, thresh=0.5,
                   use_octree=False, num_samples=10000, transform=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution)
                              #b_min, b_max, transform=transform)

    calib = calib_tensor[0].cpu().numpy()

    calib_inv = np.linalg.inv(calib)
    coords = coords.reshape(3,-1).T
    coords = np.matmul(np.concatenate([coords, np.ones((coords.shape[0],1))], 1), calib_inv.T)[:, :3]
    coords = coords.T.reshape(3,resolution,resolution,resolution)

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, 1, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        
        net.query(samples, calib_tensor)
        pred = net.get_preds()[0][0]
        return pred.detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)

    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf, thresh)
        # transform verts into world coordinate system
        trans_mat = np.matmul(calib_inv, mat)
        verts = np.matmul(trans_mat[:3, :3], verts.T) + trans_mat[:3, 3:4]
        verts = verts.T
        # in case mesh has flip transformation
        if np.linalg.det(trans_mat[:3, :3]) < 0.0:
            faces = faces[:,::-1]
        return verts, faces, normals, values
    except:
        print('error in marching cubes')
        return -1


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def import_obj_blender(verts, faces, name='NewMesh'):
    mesh_data = bpy.data.meshes.new(name=name)
    mesh_data.from_pydata(verts, [], faces)
    new_object = bpy.data.objects.new(name=name, object_data=mesh_data)
    bpy.context.collection.objects.link(new_object)


def gen_mesh(res, net, cuda, data, thresh=0.5, use_octree=True, components=False):
    image_tensor_global = data['img_512'].to(device=cuda)
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter_global(image_tensor_global)
    net.filter_local(image_tensor[:, None])

    try:
        if net.netG.netF is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlF], 0)
        if net.netG.netB is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlB], 0)
    except:
        pass

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        verts, faces, _, _ = reconstruction(
            net, cuda, calib_tensor, res, b_min, b_max, thresh, use_octree=use_octree, num_samples=50000)
        import_obj_blender(verts, faces, data['name'])

    except Exception as e:
        print(e)
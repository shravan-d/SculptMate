import bpy
import os
from bpy_extras.io_utils import ImportHelper 
from bpy.types import Operator
import addon_utils
import torch

from .generation.generate import generate_mesh
from .preprocessing import preprocess_image
from .utils import label_multiline
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_image_textures(image_path, texture_name):
    ideal_image = bpy.data.images.load(image_path, check_existing=True) 
    ideal_image.scale(128, 128)
    texture = bpy.data.textures.new(name=texture_name, type="IMAGE")
    texture.image = ideal_image
    texture.extension = 'EXTEND'
    print('Created ', texture_name)


class DataStore:
    bpy.types.Scene.input_image_path = bpy.props.StringProperty()
    bpy.types.Scene.current_texture_name = bpy.props.StringProperty()
    bpy.types.Scene.initialized = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.error_msg = bpy.props.StringProperty(default="")

    @classmethod
    def poll(cls, context):
        return True


class UI_PT_main(DataStore, bpy.types.Panel):
    bl_label = "SculptMate"
    bl_idname = "panel_PT_main"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout

        label_multiline(layout, text="Generating a human mesh from one image.")
        label_multiline(layout, text="For best results, use full-body images with arms at the sides.")
        layout.separator()
        # label_multiline(layout, text="Think of the vitruvian man as the ideal poser for the model. The closer you are to this pose, the easier it is for the model. Of course, you don't need the extra limbs, but keeping the body parts visually distinct helps.")
        # col = self.layout.box().column()
        # col.template_preview(bpy.data.textures['.vitruvianTexture'])
        layout.separator()
        layout.operator("tool.filebrowser", text="Open Image")
        if context.scene.input_image_path:
            col = self.layout.box().column()
            col.template_preview(bpy.data.textures[context.scene.current_texture_name])
            layout.separator()
        if context.scene.error_msg is not "":
            label_multiline(layout, text=context.scene.error_msg)

        layout.operator("tool.generate", text="Generate")


class FileBrowser(DataStore, Operator, ImportHelper):
    bl_idname = "tool.filebrowser"
    bl_label = "Select Image"

    def execute(self, context):
        filename = self.filepath.split('\\')[-1].split('.')[0]
        create_image_textures(self.filepath, '.'+filename)
        context.scene.input_image_path = self.filepath
        context.scene.current_texture_name = '.'+filename
        context.scene.error_msg = ""

        return {'FINISHED'}


class Generate(DataStore, Operator):
    bl_idname = "tool.generate"
    bl_label = "Generate Model"

    def execute(self, context):
        if context.scene.input_image_path is None:
            print('No Image set')
            return {'CANCELLED'}
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        img_name = os.path.splitext(os.path.basename(context.scene.input_image_path))[0]
        print('Working on ', img_name)
        t1 = time.time()
        preprocessed, scale = preprocess_image(img_path=context.scene.input_image_path, ratio=0.75)
        if preprocessed is None:
            context.scene.error_msg = "Sorry, I am unable to work with this image, please try another one. Reasons for failure could include poor quality, or inability to find a person in the image."
            return {'CANCELLED'}
        
        t2 = time.time()
        print('Preprocessing Time (s):', str(t2 - t1))
        # generate_mesh(device, img_path=input_image_path, scale=2.4)
        generate_mesh(device, input_image=preprocessed, scale=scale, input_name=img_name)
        t3 = time.time()
        print('Model Time (s):', str(t3 - t2))

        return {'FINISHED'}


def register():
    # create_image_textures(ROOT_DIR+'/checkpoints/vitruvian.jpg', '.vitruvianTexture')
    bpy.utils.register_class(UI_PT_main)
    bpy.utils.register_class(FileBrowser)
    bpy.utils.register_class(Generate)
    
def unregister():
    bpy.utils.unregister_class(UI_PT_main)
    bpy.utils.unregister_class(FileBrowser)
    bpy.utils.unregister_class(Generate)


# check if you can up the resultion
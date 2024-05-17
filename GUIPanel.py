import bpy
import os
from bpy_extras.io_utils import ImportHelper 
from bpy.types import Operator
import addon_utils
import torch
import threading

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


class DataStore:
    bpy.types.Scene.input_image_path = bpy.props.StringProperty()
    bpy.types.Scene.current_texture_name = bpy.props.StringProperty()
    bpy.types.Scene.initialized = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.buttons_enabled = bpy.props.BoolProperty(default=True)
    bpy.types.Scene.message = bpy.props.StringProperty(default="")

    @classmethod
    def poll(cls, context):
        return True


class MyProperties(bpy.types.PropertyGroup):
    model_type: bpy.props.EnumProperty(
        name="Model Type",
        description="Select the category of your mesh",
        items=[
            ('human', "Human", "Generate a human mesh"),
            ('other', "Other", "Generate other objects")
        ],
        default='other'
    ) # type: ignore


class UI_PT_main(DataStore, bpy.types.Panel):
    bl_label = "SculptMate"
    bl_idname = "panel_PT_main"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout

        label_multiline(layout, text="Generate a mesh from an image.")
        label_multiline(layout, text="For best results, use images with the object centered and fully visible.")
        layout.separator()
        # label_multiline(layout, text="Think of the vitruvian man as the ideal poser for the model. The closer you are to this pose, the easier it is for the model. Of course, you don't need the extra limbs, but keeping the body parts visually distinct helps.")
        # col = self.layout.box().column()
        # col.template_preview(bpy.data.textures['.vitruvianTexture'])
        layout.prop(context.scene.my_props, "model_type", expand=True)
        layout.separator()
        layout.operator("tool.filebrowser", text="Open Image")
        if context.scene.input_image_path:
            col = self.layout.box().column()
            col.template_preview(bpy.data.textures[context.scene.current_texture_name])
            layout.separator()
        if context.scene.message != "":
            label_multiline(layout, text=context.scene.message)

        layout.operator("tool.generate", text="Generate")


class File_OT_Browser(DataStore, Operator, ImportHelper):
    bl_idname = "tool.filebrowser"
    bl_label = "Select Image"

    @classmethod
    def poll(self, context):
        # Deactivate when generation is running
        return context.scene.buttons_enabled

    def execute(self, context):
        filename = self.filepath.split('\\')[-1].split('.')[0]
        create_image_textures(self.filepath, '.'+filename)
        context.scene.input_image_path = self.filepath
        context.scene.current_texture_name = '.'+filename
        context.scene.message = ""

        return {'FINISHED'}


class Mesh_OT_Generate(DataStore, Operator):
    bl_idname = "tool.generate"
    bl_label = "Generate Model"

    @classmethod
    def poll(self, context):
        # Deactivate when generation is running
        return context.scene.buttons_enabled


    def execute(self, context):
        if context.scene.input_image_path == "":
            self.report({"ERROR"}, 'Please select image first')
            return {'CANCELLED'}
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        img_name = os.path.splitext(os.path.basename(context.scene.input_image_path))[0]
        print('Working on ', img_name)
        t1 = time.time()
        preprocessed, scale = preprocess_image(img_path=context.scene.input_image_path, ratio=0.75)
        if preprocessed is None:
            context.scene.message = "Sorry, I am unable to work with this image, please try another one. Reasons for failure could include poor quality, or inability to find a person in the image."
            return {'CANCELLED'}
        
        t2 = time.time()
        print('Preprocessing Time (s):', str(t2 - t1))
        thread = GenerationWorker(device, preprocessed, scale, img_name, context)
        thread.start()

        return {'FINISHED'}


class GenerationWorker(DataStore, threading.Thread):

    def __init__(self, device, image, scale, name, context):
        self.device = device
        self.image = image
        self.scale = scale
        self.img_name = name
        self.context = context
        threading.Thread.__init__(self)

    def run(self):
        self.context.scene.message = "Your mesh is being generated and will show up in about 30 seconds."
        self.context.scene.buttons_enabled = False
        if self.context.scene.my_props.model_type == 'human':
            generate_mesh(self.device, input_image=self.image, scale=self.scale, input_name=self.img_name)
            # generate_mesh(device, img_path=input_image_path, scale=2.4)
        elif self.context.scene.my_props.model_type == 'other':
            print('Running Trippo')
        self.context.scene.message = ""
        self.context.scene.buttons_enabled = True
        


def register():
    # create_image_textures(ROOT_DIR+'/checkpoints/vitruvian.jpg', '.vitruvianTexture')
    bpy.utils.register_class(MyProperties)
    bpy.types.Scene.my_props = bpy.props.PointerProperty(type=MyProperties)
    bpy.utils.register_class(UI_PT_main)
    bpy.utils.register_class(File_OT_Browser)
    bpy.utils.register_class(Mesh_OT_Generate)
    
def unregister():
    bpy.utils.unregister_class(UI_PT_main)
    bpy.utils.unregister_class(File_OT_Browser)
    bpy.utils.unregister_class(Mesh_OT_Generate)
    bpy.utils.unregister_class(MyProperties)
    del bpy.types.Scene.my_props

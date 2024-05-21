import bpy
import os
from bpy_extras.io_utils import ImportHelper 
from bpy.types import Operator
import addon_utils
import torch
import threading

from .preprocessing import preprocess_image
from .utils import label_multiline
from .TripoSR.generate import TripoGenerator
from .generation.generate import PifuGenerator
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
        layout.label(text="Transform your images into stunning 3D meshes!")
        layout.label(text="For the best results:")
        layout.label(text="- Use images where the object is centered")
        layout.label(text="- Ensure the object is fully visible")
        layout.label(text="Let's goooo!")
        layout.separator()
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
        # Ensure input image is configured
        if context.scene.input_image_path == "":
            self.report({"ERROR"}, 'Please select image first')
            return {'CANCELLED'}
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        img_name = os.path.splitext(os.path.basename(context.scene.input_image_path))[0]
        print('Working on ', img_name)

        preprocessed, scale = preprocess_image(img_path=context.scene.input_image_path, ratio=0.75)
        if preprocessed is None:
            context.scene.message = "Sorry, I am unable to work with this image, please try another one. Reasons for failure could include poor quality, or inability to find a person in the image."
            return {'CANCELLED'}
        
        # Run the generation on a different thread so Blender doesn't hang
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
        # Disable the buttons while generation is running
        self.context.scene.message = "Your mesh is being generated."
        self.context.scene.buttons_enabled = False

        # Generate mesh based on selected model type
        t1 = time.time()
        if self.context.scene.my_props.model_type == 'human':
            human_gen = PifuGenerator(self.device)
            human_gen.initiate_model()
            human_gen.generate_mesh(input_image=self.image, input_name=self.img_name, scale=self.scale)
        elif self.context.scene.my_props.model_type == 'other':
            object_gen = TripoGenerator(self.device)
            object_gen.initiate_model()
            object_gen.generate_mesh(input_image=self.image, input_name=self.img_name)
        t2 = time.time()
        print('Generation Time (s):', str(t2 - t1 + 1))

        # Enable buttons once generation is complete
        self.context.scene.message = ""
        self.context.scene.buttons_enabled = True
        


def register():
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

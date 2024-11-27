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
from .StableFast.generate import Fast3DGenerator
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
trippo_generator = None
fast_generator = None


def create_image_textures(image_path, texture_name):
    ideal_image = bpy.data.images.load(image_path, check_existing=True) 
    ideal_image.scale(128, 128)
    texture = bpy.data.textures.new(name=texture_name, type="IMAGE")
    texture.image = ideal_image
    texture.extension = 'EXTEND'


class DataStore:
    # WindowManager vars are reset when Blender is closed. Scene vars are serialized with the save file
    bpy.types.WindowManager.input_image_path = bpy.props.StringProperty(default="")
    bpy.types.WindowManager.current_texture_name = bpy.props.StringProperty(default="")
    bpy.types.WindowManager.buttons_enabled = bpy.props.BoolProperty(default=True)
    bpy.types.WindowManager.message = bpy.props.StringProperty(default="")

    @classmethod
    def poll(cls, context):
        return True


class MyProperties(bpy.types.PropertyGroup):
    model_type: bpy.props.EnumProperty(
        name="Model Type",
        description="Select the model to use",
        items=[
            ('lean', "Lean", "Quickly generate a mesh"),
            ('fast', "Pro", "Generates meshes with higher quality. Requires GPU configuration.")
        ],
        default='lean'
    ) # type: ignore

    vertex_simplification: bpy.props.EnumProperty(
        name="Vertex Count",
        description="Controls the number of vertices in your mesh",
        items=[
            ('low', "Low", ""),
            ('medium', "Medium", ""),
            ('high', "High", ""),
        ],
        default='low'
    ) # type: ignore
        
    enable_textures: bpy.props.BoolProperty(
        name='Transfer Textures', 
        description="Transfer texture from the image to your mesh.",
        default=False
    ) # type: ignore


class UI_PT_main(DataStore, bpy.types.Panel):
    bl_label = "SculptMate"
    bl_idname = "panel_PT_main"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Transform images into 3D meshes!")
        layout.label(text="For the best results:")
        layout.label(text="- Ensure one object per image")
        layout.label(text="- Avoid occlusion")
        layout.separator()

        my_props = context.scene.my_props
        items = my_props.bl_rna.properties["model_type"].enum_items_static
        row = layout.row(align=True)
        for item in items:
            identifier = item.identifier  
            item_layout = row.row(align=True)  
            item_layout.prop_enum(my_props, "model_type", identifier)
            if identifier == 'lean' and os.path.isfile(ROOT_DIR + '/TripoSR/checkpoints/model.ckpt'):
                item_layout.enabled = True
            elif identifier == 'fast' and os.path.isfile(ROOT_DIR + '/StableFast/checkpoints/model.safetensors') and torch.cuda.is_available():
                item_layout.enabled = True
            else:
                item_layout.enabled = False

        layout.separator()
        if context.scene.my_props.model_type == 'fast':
            layout.label(text="Vertex Count")
            layout.prop(context.scene.my_props, "vertex_simplification", expand=True)
        
        layout.separator()
        layout.prop(context.scene.my_props, "enable_textures")
        layout.operator("tool.filebrowser", text="Open Image")
        if context.window_manager.input_image_path != "":
            img = bpy.data.images.load(context.window_manager.input_image_path, check_existing=True)
            icon_id = bpy.types.UILayout.icon(img)
            layout.box().row().template_icon(icon_value=icon_id, scale=9)
            layout.separator()
        if context.window_manager.message != "":
            label_multiline(layout, text=context.window_manager.message)

        layout.operator("tool.generate", text="Generate")


class File_OT_Browser(DataStore, Operator, ImportHelper):
    bl_idname = "tool.filebrowser"
    bl_label = "Select Image"

    @classmethod
    def poll(self, context):
        # Deactivate when generation is running
        return context.window_manager.buttons_enabled

    def execute(self, context):
        filename = self.filepath.split('\\')[-1].split('.')[0]
        create_image_textures(self.filepath, '.'+filename)
        context.window_manager.input_image_path = self.filepath
        context.window_manager.current_texture_name = '.'+filename
        context.window_manager.message = ""

        return {'FINISHED'}


class Mesh_OT_Generate(DataStore, Operator):
    bl_idname = "tool.generate"
    bl_label = "Generate Model"

    @classmethod
    def poll(self, context):
        # Deactivate when generation is running
        return context.window_manager.buttons_enabled


    def execute(self, context):
        # Ensure input image is configured
        if context.window_manager.input_image_path == "":
            self.report({"ERROR"}, 'Please select image first')
            return {'CANCELLED'}
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        img_name = os.path.splitext(os.path.basename(context.window_manager.input_image_path))[0]
        print('[SculptMate Logging] Working on ', img_name)

        try:
            if context.scene.my_props.model_type == 'lean':
                preprocessed = preprocess_image(img_path=context.window_manager.input_image_path, ratio=0.75)
            elif context.scene.my_props.model_type == 'fast':
                preprocessed = preprocess_image(img_path=context.window_manager.input_image_path, ratio=0.85, alpha=True)

        except Exception as e:
            self.report({"ERROR"}, 'Please view system console for details')
            print('[Preprocessing Error]', e)
            return {'CANCELLED'}

        if preprocessed is None:
            context.window_manager.message = "Sorry, I am unable to work with this image, please try another one. Reasons for failure could include poor quality, or inability to find an object in the image."
            return {'CANCELLED'}
        
        # Run the generation on a different thread so Blender doesn't hang
        thread = GenerationWorker(device, preprocessed, img_name, context)
        thread.start()

        return {'FINISHED'}


class GenerationWorker(DataStore, threading.Thread):

    def __init__(self, device, image, name, context):
        self.device = device
        self.image = image
        self.img_name = name
        self.context = context
        threading.Thread.__init__(self)

    def run(self):
        # Disable the buttons while generation is running
        self.context.window_manager.message = "Your mesh is being generated."
        self.context.window_manager.buttons_enabled = False

        # Generate mesh based on selected model type
        t1 = time.time()
        if self.context.scene.my_props.model_type == 'lean':
            global trippo_generator
            if trippo_generator is None:
                trippo_generator = TripoGenerator(self.device)
                trippo_generator.initiate_model()
            trippo_generator.generate_mesh(
                input_image=self.image, 
                input_name=self.img_name, 
                enable_texture=self.context.scene.my_props.enable_textures
            )
        elif self.context.scene.my_props.model_type == 'fast':
            global fast_generator
            if fast_generator is None:
                fast_generator = Fast3DGenerator(self.device)
                fast_generator.initiate_model()
            fast_generator.generate_mesh(
                input_image=self.image, 
                input_name=self.img_name, 
                vertex_simplification_factor=self.context.scene.my_props.vertex_simplification,
                enable_texture=self.context.scene.my_props.enable_textures
            )
        t2 = time.time()
        print('[SculptMate Logging] Generation Time (s):', str(t2 - t1 + 1))

        # Enable buttons once generation is complete
        self.context.window_manager.message = ""
        self.context.window_manager.buttons_enabled = True


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

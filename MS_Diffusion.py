# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import random
import sys
import re
import yaml
from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers import (StableDiffusionXLPipeline,  DDIMScheduler, ControlNetModel,
                       KDPM2AncestralDiscreteScheduler, LMSDiscreteScheduler,
                       DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler,
                       EulerDiscreteScheduler, HeunDiscreteScheduler,
                        KDPM2DiscreteScheduler,
                       EulerAncestralDiscreteScheduler, UniPCMultistepScheduler, AutoencoderKL,
                       StableDiffusionXLControlNetPipeline, DDPMScheduler, LCMScheduler)
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor
from .msdiffusion.models.projection import Resampler
from .msdiffusion.models.model import MSAdapter
from .msdiffusion.utils import get_phrase_idx, get_eot_idx
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import folder_paths
from comfy.utils import common_upscale


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

MAX_SEED = np.iinfo(np.int32).max
dir_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(dir_path)
file_path = os.path.dirname(path_dir)


controlnet_list=["controlnet-openpose-sdxl-1.0","controlnet-zoe-depth-sdxl-1.0","controlnet-scribble-sdxl-1.0","controlnet-tile-sdxl-1.0","controlnet-depth-sdxl-1.0","controlnet-canny-sdxl-1.0","MistoLine","sdxl-controlnet-seg"]


control_paths = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "config.json" in files:
                control_paths.append(os.path.relpath(root, start=search_path))
                control_paths = [z for z in control_paths if z.split("\\")[-1] in controlnet_list or z in controlnet_list]

if control_paths:
    control_paths = ["none"] + [x for x in control_paths if x]
else:
    control_paths = ["none",]


clip_paths = []
for search_path in folder_paths.get_folder_paths("clip_vision"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "config.json" in files:
                clip_paths.append(os.path.relpath(root, start=search_path))
if clip_paths:
    clip_paths = ["none"] + [x for x in clip_paths if x]
else:
    clip_paths = ["none",]


def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path

loras_path = get_instance_path(os.path.join(dir_path,"config","lora.yaml"))
def get_lora_dict():
    # 打开并读取YAML文件
    with open(loras_path, 'r', encoding="UTF-8") as stream:
        try:
            # 解析YAML文件内容
            data = yaml.safe_load(stream)

            # 此时 'data' 是一个Python字典，里面包含了YAML文件的所有数据
            # print(data)
            return data

        except yaml.YAMLError as exc:
            # 如果在解析过程中发生了错误，打印异常信息
            print(exc)

datas = get_lora_dict()
lora_lightning_list = datas["lightning_xl_lora"]

scheduler_list = [
    "Euler", "Euler a", "DDIM", "DDPM", "DPM++ 2M", "DPM++ 2M Karras", "DPM++ 2M SDE", "DPM++ 2M SDE Karras",
    "DPM++ SDE", "DPM++ SDE Karras", "DPM2",
    "DPM2 Karras", "DPM2 a", "DPM2 a Karras", "Heun", "LCM", "LMS", "LMS Karras", "UniPC"
]


def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def nomarl_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return samples


def phi2tensor(img):
    image = np.array(img).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image

def add_pil(list, list_add, num):
    new_list = list[:num] + list_add + list[num:]
    return new_list

def tensor_to_image(tensor):
    # tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

# get fonts list
def has_parentheses(s):
    return bool(re.search(r'\(.*?\)', s))

def contains_brackets(s):
    return '[' in s or ']' in s

def narry_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = phi2narry(value)
        list_in[i] = modified_value
    return list_in

def phi_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        list_in[i] = value
    return list_in

def narry_list_pil(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = tensor_to_image(value)
        list_in[i] = modified_value
    return list_in

def get_local_path(file_path, model_path,file):
    path = os.path.join(file_path, "models", file, model_path)
    model_path = os.path.normpath(path)
    if sys.platform.startswith('win32'):
        model_path = model_path.replace('\\', "/")
    return model_path

def extract_content_between_brackets(text):
    # 正则表达式匹配多对括号内的内容
    return re.findall(r'\((.*?)\)', text)

def extract_content_from_brackets(text):
    # 正则表达式匹配多对方括号内的内容
    return re.findall(r'\[(.*?)\]', text)

def setup_seed(seed):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_image_path_list(folder_name):
    image_basename_list = os.listdir(folder_name)
    image_path_list = sorted(
        [os.path.join(folder_name, basename) for basename in image_basename_list]
    )
    return image_path_list

def get_scheduler(name):
    scheduler = False
    if name == "Euler":
        scheduler = EulerDiscreteScheduler()
    elif name == "Euler a":
        scheduler = EulerAncestralDiscreteScheduler()
    elif name == "DDIM":
        scheduler = DDIMScheduler()
    elif name == "DDPM":
        scheduler = DDPMScheduler()
    elif name == "DPM++ 2M":
        scheduler = DPMSolverMultistepScheduler()
    elif name == "DPM++ 2M Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True)
    elif name == "DPM++ 2M SDE":
        scheduler = DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ 2M SDE Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ SDE":
        scheduler = DPMSolverSinglestepScheduler()
    elif name == "DPM++ SDE Karras":
        scheduler = DPMSolverSinglestepScheduler(use_karras_sigmas=True)
    elif name == "DPM2":
        scheduler = KDPM2DiscreteScheduler()
    elif name == "DPM2 Karras":
        scheduler = KDPM2DiscreteScheduler(use_karras_sigmas=True)
    elif name == "DPM2 a":
        scheduler = KDPM2AncestralDiscreteScheduler()
    elif name == "DPM2 a Karras":
        scheduler = KDPM2AncestralDiscreteScheduler(use_karras_sigmas=True)
    elif name == "Heun":
        scheduler = HeunDiscreteScheduler()
    elif name == "LCM":
        scheduler = LCMScheduler()
    elif name == "LMS":
        scheduler = LMSDiscreteScheduler()
    elif name == "LMS Karras":
        scheduler = LMSDiscreteScheduler(use_karras_sigmas=True)
    elif name == "UniPC":
        scheduler = UniPCMultistepScheduler()
    return scheduler

def nomarl_upscale_topil(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_image(samples)
    return img_pil

def instance_path(path, repo,file):
    if repo == "":
        if path == "none":
            repo = "none"
        else:
            model_path = get_local_path(file_path, path,file)
            repo = get_instance_path(model_path)
    return repo

class MSdiffusion_Model_Loader:
    def __init__(self):
        self.counters = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (["none"]+folder_paths.get_filename_list("checkpoints"),),
                "repo_id": ("STRING", {"default": ""}),
                "clip_vision_local":(clip_paths,),
                "clip_vision_repo": ("STRING", {"default": ""}),
                "vae_id": (["none"]+folder_paths.get_filename_list("vae"),),
                "controlnet_diff": (control_paths,),
                "controlnet_repo":("STRING", {"default": ""}),
                "lora": (["none"] + folder_paths.get_filename_list("loras"),),
                "lora_scale": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                "trigger_words": ("STRING", {"default": "best quality"}),
                "scheduler": (scheduler_list,),}
        }


    RETURN_TYPES = ("MODEL","MODEL","MODEL","MODEL", "STRING",)
    RETURN_NAMES = ("pipe","ms_model","image_encoder", "image_processor","info",)
    FUNCTION = "ms_model_loader"
    CATEGORY = "MSdiffusion"

    def ms_model_loader(self, ckpt_name,repo_id,clip_vision_local, clip_vision_repo, vae_id,controlnet_diff,controlnet_repo,lora, lora_scale, trigger_words,scheduler,
                           ):
        
        scheduler_choice = get_scheduler(scheduler)
        controlnet_model=instance_path(controlnet_diff, controlnet_repo,"diffusers")
        clip_vision=instance_path(clip_vision_local, clip_vision_repo,"clip_vision")
        ckpt_path=folder_paths.get_full_path("checkpoints", ckpt_name)
       
        if clip_vision=="none":
            raise "need clip_vision"

        if lora != "none":
            lora_path = folder_paths.get_full_path("loras", lora)
            lora_path = get_instance_path(lora_path)
        else:
            lora_path = ""
        if "/" in lora:
            lora = lora.split("/")[-1]
        if "\\" in lora:
            lora = lora.split("\\")[-1]

        # load SDXL pipeline
        if not repo_id:
            original_config_file = os.path.join(dir_path, "config", "sd_xl_base.yaml")
            
            if controlnet_model != "none":  # 启用control
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_model,use_safetensors=True,
                    torch_dtype=torch.float16
                )
                try:
                   pipe = StableDiffusionXLControlNetPipeline.from_single_file(ckpt_path,controlnet=controlnet,original_config=original_config_file,torch_dtype=torch.float16, )
                except:
                    try:
                        pipe = StableDiffusionXLControlNetPipeline.from_single_file(ckpt_path, controlnet=controlnet,
                                                                                    original_config_file=original_config_file,
                                                                                    torch_dtype=torch.float16, )
                    except:
                        raise "model error"
            else:
                try:
                    pipe = StableDiffusionXLPipeline.from_single_file(
                        ckpt_path, original_config=original_config_file, torch_dtype=torch.float16)
                except:
                    try:
                        pipe = StableDiffusionXLPipeline.from_single_file(
                            ckpt_path, original_config_file=original_config_file, torch_dtype=torch.float16)
                    except:
                        raise "model error"
               
        else:
            if controlnet_model != "none":  # 启用control
                try:
                    controlnet = ControlNetModel.from_pretrained(
                        controlnet_model, use_safetensors=True, variant="fp16",
                        torch_dtype=torch.float16
                    )
                except:
                    try:
                        controlnet = ControlNetModel.from_pretrained(
                            controlnet_model, use_safetensors=True,
                            torch_dtype=torch.float16
                        )
                    except:
                        raise "model error,need rename model like'diffusion_pytorch_model.safetensors'"
                
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(repo_id,
                                                                           controlnet=controlnet,
                                                                           torch_dtype=torch.float16,
                                                                           )
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    repo_id, torch_dtype=torch.float16, add_watermarker=False,
                )
               
        if vae_id!="none":
            vae_id=folder_paths.get_full_path("vae", vae_id)
            pipe.vae = AutoencoderKL.from_single_file(vae_id, torch_dtype=torch.float16).to(device)
        if lora != "none":
            if lora in lora_lightning_list:
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora()
            else:
                pipe.load_lora_weights(lora_path, adapter_name=trigger_words)
                pipe.fuse_lora(adapter_names=[trigger_words,], lora_scale=lora_scale)
       
        pipe.scheduler = scheduler_choice.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        pipe.enable_vae_slicing()
        if device != "mps":
            pipe.enable_model_cpu_offload()
        ms_dir = os.path.join(file_path, "models","photomaker")
        photomaker_local_path = os.path.join(ms_dir, "ms_adapter.bin")
        if not os.path.exists(photomaker_local_path):
            ms_path = hf_hub_download(
                repo_id="doge1516/MS-Diffusion",
                filename="ms_adapter.bin",
                repo_type="model",
                local_dir=ms_dir,
            )
        else:
            ms_path = photomaker_local_path
        ms_ckpt = get_instance_path(ms_path)
        image_processor = CLIPImageProcessor()
        #print(type(image_processor))
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_vision).to(device,dtype=torch.float16)
        # print(type(image_encoder))
        image_encoder_projection_dim = image_encoder.config.projection_dim
        num_tokens = 16
        image_encoder_type = "clip"
        image_proj_type = "resampler"
        latent_init_mode = "grounding"
        # latent_init_mode = "random"
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=pipe.unet.config.cross_attention_dim,
            ff_mult=4,
            latent_init_mode=latent_init_mode,
            phrase_embeddings_dim=pipe.text_encoder.config.projection_dim,
        ).to(device,dtype=torch.float16)
        ms_model = MSAdapter(pipe.unet, image_proj_model, ckpt_path=ms_ckpt, device=device, num_tokens=num_tokens)
        ms_model.to(device, dtype=torch.float16)
        torch.cuda.empty_cache()
        info = ";".join(
            [lora, trigger_words,controlnet_model,image_encoder_type,image_proj_type])
        return (pipe,ms_model,image_encoder,image_processor, info,)
    

class MSdiffusion_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pipe": ("MODEL",),
                "ms_model": ("MODEL",),
                "image_encoder": ("MODEL",),
                "image_processor": ("MODEL",),
                "info": ("STRING", {"forceInput": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "a [dog] wearing a pink sunglass"}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "default": "(worst quality, low quality, normal quality, lowres),"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7, "min": 0.1, "max": 10.0, "step": 0.1, "round": 0.01}),
                "scale": (
                    "FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "mask_threshold": (
                    "FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "start_step": ("INT", {"default": 5, "min": 1, "max": 1024}),
                
                "controlnet_scale": (
                    "FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64, "display": "number"}),
                "batch_size":("INT", {"default": 1, "min": 1, "max": 100,"step": 1,}),
                "drop_grounding_tokens":("BOOLEAN", {"default": False},),
                "guidance_list": ("STRING", {"multiline": True,"default": "0., 0.25, 0.4, 0.75;0.6, 0.25, 1., 0.75"}),
                
            },
            "optional": {"control_image": ("IMAGE",),
                         }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("batch_size"):
            return

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image", )
    FUNCTION = "ms_sampler"
    CATEGORY = "MSdiffusion"

    def main_normal(self,prompt,pipe,phrases,ms_model,input_images,num_samples,steps,seed,negative_prompt,scale,image_encoder,cfg,image_processor,
                    boxes,mask_threshold,start_step,image_proj_type,image_encoder_type,drop_grounding_tokens,height,width,phrase_idxes, eot_idxes):

        images = ms_model.generate(pipe=pipe, pil_images=[input_images], num_samples=num_samples,
                                   num_inference_steps=steps,
                                   seed=seed,
                                   prompt=[prompt], negative_prompt=negative_prompt, scale=scale,
                                   image_encoder=image_encoder, guidance_scale=cfg,
                                   image_processor=image_processor, boxes=boxes,
                                   mask_threshold=mask_threshold,
                                   start_step=start_step,
                                   image_proj_type=image_proj_type,
                                   image_encoder_type=image_encoder_type,
                                   phrases=phrases,
                                   drop_grounding_tokens=drop_grounding_tokens,
                                   phrase_idxes=phrase_idxes, eot_idxes=eot_idxes, height=height,
                                   width=width)
        return images
    def main_control(self,prompt,width,height,pipe,phrases,ms_model,input_images,num_samples,steps,seed,negative_prompt,scale,image_encoder,cfg,
                     image_processor,boxes,mask_threshold,start_step,image_proj_type,image_encoder_type,drop_grounding_tokens,controlnet_scale,control_image,phrase_idxes, eot_idxes):
        images = ms_model.generate(pipe=pipe, pil_images=[input_images], num_samples=num_samples,
                                   num_inference_steps=steps,
                                   seed=seed,
                                   prompt=[prompt], negative_prompt=negative_prompt, scale=scale,
                                   image_encoder=image_encoder, guidance_scale=cfg,
                                   image_processor=image_processor, boxes=boxes,
                                   mask_threshold=mask_threshold,
                                   start_step=start_step,
                                   image_proj_type=image_proj_type,
                                   image_encoder_type=image_encoder_type,
                                   phrases=phrases,
                                   drop_grounding_tokens=drop_grounding_tokens,
                                   phrase_idxes=phrase_idxes, eot_idxes=eot_idxes, height=height,
                                   width=width,
                                   image=control_image, controlnet_conditioning_scale=controlnet_scale)

        return images

    def get_phrases_idx(self,tokenizer, phrases, prompt):
        res = []
        phrase_cnt = {}
        for phrase in phrases:
            if phrase in phrase_cnt:
                cur_cnt = phrase_cnt[phrase]
                phrase_cnt[phrase] += 1
            else:
                cur_cnt = 0
                phrase_cnt[phrase] = 1
            res.append(get_phrase_idx(tokenizer, phrase, prompt, num=cur_cnt)[0])
        return res
    
    def get_float(self,str_in):
        list_str=str_in.split(",")
        float_box=[float(x) for x in list_str]
        return float_box
  
    def ms_sampler(self,image,pipe,ms_model,image_encoder,image_processor,info,prompt,negative_prompt,seed, steps,
                  cfg,scale, mask_threshold, start_step,controlnet_scale,width,height,batch_size,drop_grounding_tokens,guidance_list,**kwargs):
        lora, trigger_words,controlnet_model,image_encoder_type,image_proj_type ,= info.split(";")
        if drop_grounding_tokens:
            drop_grounding_tokens = [1]# set to 1 if you want to drop the grounding tokens
        else:
            drop_grounding_tokens = [0]
        
        guidance_list=guidance_list.strip().split(";")
        
        box_add=[] # 获取预设box
        for i in range(len(guidance_list)):
            box_add.append(self.get_float(guidance_list[i]))
        
        if mask_threshold==0.:
            mask_threshold=None
            
        if '[' in prompt and ']' in prompt:
            object_prompt = extract_content_from_brackets(prompt)  # 提取prompt的object list
            unique_object_prompt = sorted(list(set(object_prompt)),key=lambda x: list(object_prompt).index(x)) # 清除同名物体,保持原有顺序
            object_num = len(unique_object_prompt)
        else:
            raise "Need to enclose the name of the object in square brackets"
        
        d1, _, _, _ = image.size()
        #print(d1,object_num)
        
        if lora != "none":
            prompt = prompt + trigger_words + " " + "style"
        
        prompt=prompt.replace("]", " ").replace("[", " ") #去除[]
        
        if object_num == d1:
            phrases = [unique_object_prompt]
            print(phrases)
        else:
            raise "The number of objects should be equal to the number of input images"
        
        if d1 == 1:
            input_images = [nomarl_upscale_topil(image, 512, 512)]
            torch.cuda.empty_cache()
            if mask_threshold: # step之前所用布局指导
                boxes = box_add[0]
                boxes = [[boxes]]  # 随机选1个自定义的
            # boxes = [[[0.25, 0.25, 0.75, 0.75]]]  # 1 object
            else:
                boxes = [[[0., 0., 0., 0.]]]  # 1 object
            phrase_idxes = [self.get_phrases_idx(pipe.tokenizer, phrases[0], prompt)]
            eot_idxes = [[get_eot_idx(pipe.tokenizer, prompt)] * len(phrases[0])]
            if controlnet_model != "none":
                control_image = kwargs.get("control_image")
                control_img = nomarl_upscale_topil(control_image,  width, height)
                image_main=self.main_control(prompt,width,height,pipe,phrases,ms_model,input_images,batch_size,steps,seed,negative_prompt,scale,image_encoder,cfg,
                     image_processor,boxes,mask_threshold,start_step,image_proj_type,image_encoder_type,drop_grounding_tokens,controlnet_scale,control_img,phrase_idxes, eot_idxes)

            else: #no controlnet
                image_main=self.main_normal( prompt, pipe, phrases, ms_model, input_images, batch_size, steps, seed,
                            negative_prompt, scale, image_encoder, cfg, image_processor,
                            boxes, mask_threshold, start_step, image_proj_type, image_encoder_type,
                            drop_grounding_tokens, height, width,phrase_idxes, eot_idxes)

        else:
            img_list = list(torch.chunk(image, chunks=d1))
            #print(img_list)
            input_images = [nomarl_upscale_topil(img, 512, 512) for img in img_list]
            #print(input_images)
            if mask_threshold:
                if len(box_add)<d1:
                    raise "When using 'mask_threshold', a sequence equal to the number of objects must be entered in 'guidance list'"
                else:
                    boxes = [box_add[:d1]]
                    print(boxes)
                #boxes = [[[0., 0.25, 0.4, 0.75], [0.6, 0.25, 1., 0.75]]]  # man+women
            else:
                zero_list = [0 for _ in range(4)]
                boxes=[zero_list for _ in range(d1)]
                boxes = [boxes]  # used if you want no layout guidance
                #print(boxes)
            phrase_idxes = [self.get_phrases_idx(pipe.tokenizer, phrases[0], prompt)]
            eot_idxes = [[get_eot_idx(pipe.tokenizer, prompt)] * len(phrases[0])]
            if controlnet_model != "none":
                control_image = kwargs.get("control_image")
                control_img = nomarl_upscale_topil(control_image, width, height)   # pil
                image_main=self.main_control(prompt,width,height,pipe,phrases,ms_model,input_images,batch_size,steps,seed,negative_prompt,scale,image_encoder,cfg,
                     image_processor,boxes,mask_threshold,start_step,image_proj_type,image_encoder_type,drop_grounding_tokens,controlnet_scale,control_img,phrase_idxes, eot_idxes)

            else: #no controlnet
                image_main = self.main_normal(prompt, pipe, phrases, ms_model, input_images, batch_size, steps, seed,
                            negative_prompt, scale, image_encoder, cfg, image_processor,
                            boxes, mask_threshold, start_step, image_proj_type, image_encoder_type,
                            drop_grounding_tokens, height, width,phrase_idxes, eot_idxes)
        
        output_img=[]
        for i in range(batch_size):
            output_img.append(image_main[i])
        
        image_list = narry_list(output_img)
        image = torch.from_numpy(np.fromiter(image_list, np.dtype((np.float32, (height, width, 3)))))
        return (image, )

class MS_Object_img_Batch:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image_a": ("IMAGE",),
                             },
                "optional": {"image_b": ("IMAGE",),
                             "image_c": ("IMAGE",),
                             "image_d": ("IMAGE",)}
                }

    RETURN_TYPES = ("IMAGE",)
    ETURN_NAMES = ("image",)
    FUNCTION = "main_batch"
    CATEGORY = "MSdiffusion"

    def main_batch(self, image_a, **kwargs):
        image_b = kwargs.get("image_b")
        image_c = kwargs.get("image_c")
        image_d = kwargs.get("image_d")

        width=image_a.shape[2]
        height=image_a.shape[1]
        img_list=[image_a]
        if isinstance(image_b, torch.Tensor):
            image_b=nomarl_upscale(image_b, width, height)
            img_list.append(image_b)
        if isinstance(image_c, torch.Tensor):
            image_c = nomarl_upscale(image_c, width, height)
            img_list.append(image_c)
        if isinstance(image_d, torch.Tensor):
            image_d = nomarl_upscale(image_d, width, height)
            img_list.append(image_d)
        image = torch.cat(tuple(img_list), dim=0)
        return (image,)


NODE_CLASS_MAPPINGS = {
    "MSdiffusion_Model_Loader": MSdiffusion_Model_Loader,
    "MSdiffusion_Sampler": MSdiffusion_Sampler,
    "MS_Object_img_Batch":MS_Object_img_Batch,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MSdiffusion_Model_Loader": "MSdiffusion_Model_Loader",
    "MSdiffusion_Sampler": "MSdiffusion_Sampler",
    "MS_Object_img_Batch":"MS_Object_img_Batch"
}

# ComfyUI_MS_Diffusion
You can using MS_Diffusion in ComfyUI 

[中文说明](https://github.com/smthemex/ComfyUI_MS_Diffusion/blob/main/README-CN.md)
-- 
MS-Diffusion origin From: [MS-Diffusion](https://github.com/MS-Diffusion/MS-Diffusion)
----
My ComfyUI node list：
-----

1、ParlerTTS node:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B node:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion node：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator node： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion node：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops node：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0 node ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 node：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet node：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool node :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story node :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)   
12、PBR_Maker node:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker)      
13、ComfyUI_Streamv2v_Plus node:[ComfyUI_Streamv2v_Plus](https://github.com/smthemex/ComfyUI_Streamv2v_Plus)   
14、ComfyUI_MS_Diffusion node:[ComfyUI_MS_Diffusion](https://github.com/smthemex/ComfyUI_MS_Diffusion)   


1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
  
  ``` python 
  git clone https://github.com/smthemex/ComfyUI_MS_Diffusion.git
  
  ```
2.requirements  
----
```
pip install -r requirements.txt
```

If the module is missing, please pip install   

3 Need  model 
----
3.1 base model:   
Choose one of the three options above the model loading node, and change the rest to "none" or leave it blank. The first option is to select a standalone SDXL community model, the second option is to select a pre downloaded SDXL diffuser model, and the third option is to use the "repo_id" method    
The default repo_id models are stabilityai/stable-diffusion-xl-base-1.0 (for example,you can using :G161222/RealVisXL_V4.0,sd-community/sdxl-flash...)     

3.2 adapter and  encoder model     
Need download "ms_adapter.bin" : [link](https://huggingface.co/doge1516/MS-Diffusion/tree/main)    
Need encoder model "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k":[link](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)    

```
├── ComfyUI/custom_nodes/ComfyUI_MS_Diffusion/
|      ├──weights/
|             ├── ms_adapter.bin
```
if you want to use menu, encoder model "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" shoule be:
```
├── ComfyUI/models/
|      ├──clip_vision/
|             ├── laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
```

3.2 offline  
If the address is not in the default C drive category, you can fill in the absolute address of the diffusion model in the "path" column, which must be "/"   
    
Fill in the absolute path of your local clip model in the "laion/CLIP ViT bigG-14-laion2B-39B-b160k" column, using "/".    
Please refer to the file structure demonstration below for the required files.        
```
├── ComfyUI/custom_nodes/ComfyUI_StoryDiffusion/
|      ├──weights/
|             ├── ms_adapter.bin
├── Any local_path/
|     ├──CLIP ViT bigG-14-laion2B-39B-b160k/
|             ├── config.json
|             ├── preprocessor_config.json
|             ├──pytorch_model.bin.index.json
|             ├──pytorch_model-00001-of-00002.bin
|             ├──pytorch_model-00002-of-00002.bin
|             ├──special_tokens_map.json
|             ├──tokenizer.json
|             ├──tokenizer_config.json
|             ├──vocab.json
```

3.3 The model file example for dual role controllnet is as follows, which only supports SDXL controllnet    
```
├── ComfyUI/models/diffusers/   
|     ├──xinsir/controlnet-openpose-sdxl-1.0    
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors   
|     ├──xinsir/controlnet-scribble-sdxl-1.0   
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors   
|     ├──diffusers/controlnet-canny-sdxl-1.0   
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors   
|     ├──diffusers/controlnet-depth-sdxl-1.0   
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors
|     ├──/controlnet-zoe-depth-sdxl-1.0  
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors
|     ├──TheMistoAI/MistoLine 
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors
|     ├──xinsir/controlnet-tile-sdxl-1.0
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors
   
```
Control_img image preprocessing, please use other nodes     

4 using tips
---

--- The default scene prompt includes three modes: double(double[ ]), single(single[ ), and unmanned([NC]).    
---The character prompt must have [name], which can be man, dog, panda, and so on.       
---When using ControlNet, the number of images in ControlNet must match the number of scene prompts.  
---Other functions similar to storydiffusion   
---The bottom of the sampling node is a function that can only be used after starting layout_guidance (custom character), and must be a floating-point number less than 1.0.   

5 Example
----
img2img lora      
![](https://github.com/smthemex/ComfyUI_MS_Diffusion/blob/main/examples/img2img.png)

img2img mode, add controlnet    
![](https://github.com/smthemex/ComfyUI_MS_Diffusion/blob/main/examples/controlnet.png)


Citation
------
MS-Diffusion
```
@misc{wang2024msdiffusion,
  title={MS-Diffusion: Multi-subject Zero-shot Image Personalization with Layout Guidance}, 
  author={X. Wang and Siming Fu and Qihan Huang and Wanggui He and Hao Jiang},
  year={2024},
  eprint={2406.07209},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

IP-Adapter
```
@article{ye2023ip-adapter,
  title={IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models},
  author={Ye, Hu and Zhang, Jun and Liu, Sibo and Han, Xiao and Yang, Wei},
  booktitle={arXiv preprint arxiv:2308.06721},
  year={2023}
}
```


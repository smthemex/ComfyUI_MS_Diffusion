Youu can using MS_Diffusion in ComfyUI 

-- 
MS-Diffusion origin From: [MS-Diffusion](https://github.com/MS-Diffusion/MS-Diffusion)
----

NEW Update
---
2024/08/16  
--del clip repo，Add comfyUI clip_vision loader/加入comfyUI的clip vision节点，不再使用 clip repo。   

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
缺啥装啥。   
If the module is missing, please pip install   

3 Need  model 
----
3.1 base model:   
底模使用社区单体模型，首次使用需要下载一些必备的config文件，没连外网或者没有做镜像站映射的，肯定是用不了的。   
Choose one of the three options above the model loading node, and change the rest to "none" or leave it blank. The first option is to select a standalone SDXL community model, the second option is to select a pre downloaded SDXL diffuser model, and the third option is to use the "repo_id" method    
The default repo_id models are stabilityai/stable-diffusion-xl-base-1.0 (for example,you can using :G161222/RealVisXL_V4.0,sd-community/sdxl-flash...)     

3.2 adapter and  encoder model   
必须的模型，位置如图：   
Need download "ms_adapter.bin" : [link](https://huggingface.co/doge1516/MS-Diffusion/tree/main)    

```
├── ComfyUI/models/
|      ├──photomaker/
|             ├── ms_adapter.bin
```
clip vision model  （ any  base from CLIP-ViT-bigG-14-laion2B-39B-b160k）  
```
├── ComfyUI/models/
|      ├──clip_vision/
|             ├── clip_vision_g.safetensors  or CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors
```

3.3 only supports SDXL controllnet    
以下是controlnet的模型和config文件，ComfyUI的用的单体controlnet模型用不了。用了太占显存了，改回pipe基本加载模式。      
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
--生成物体名称需要用[  ]括起来，有多少个物体，就要有多少张图片输入;  
--To generate object names, they need to be enclosed in [  ]. As many objects as there are, there must be as many images to input;     

5 Example
----

normal 2 boject 常规双主体图生图 最新示例。     
![](https://github.com/smthemex/ComfyUI_MS_Diffusion/blob/main/examples/example_new.png)

normal 4 boject 4主体图生图。更多的主体没有测试，旧示例，仅供参考。            
![](https://github.com/smthemex/ComfyUI_MS_Diffusion/blob/main/examples/autolayerimg4img.png)

2 object zero shot and controlnet img2img  双物体加controlnet引导 图生图,旧示例，仅供参考  
![](https://github.com/smthemex/ComfyUI_MS_Diffusion/blob/main/examples/controlnet%20_obj.png)
![](https://github.com/smthemex/ComfyUI_MS_Diffusion/blob/main/examples/controlnet%20_obj1.png)


6 My ComfyUI node list：
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
15、ComfyUI_AnyDoor node: [ComfyUI_AnyDoor](https://github.com/smthemex/ComfyUI_AnyDoor)  
16、ComfyUI_Stable_Makeup node: [ComfyUI_Stable_Makeup](https://github.com/smthemex/ComfyUI_Stable_Makeup)  
17、ComfyUI_EchoMimic node:  [ComfyUI_EchoMimic](https://github.com/smthemex/ComfyUI_EchoMimic)   
18、ComfyUI_FollowYourEmoji node: [ComfyUI_FollowYourEmoji](https://github.com/smthemex/ComfyUI_FollowYourEmoji)   
19、ComfyUI_Diffree node: [ComfyUI_Diffree](https://github.com/smthemex/ComfyUI_Diffree)    
20、ComfyUI_FoleyCrafter node: [ComfyUI_FoleyCrafter](https://github.com/smthemex/ComfyUI_FoleyCrafter)   
21、ComfyUI_MooER: [ComfyUI_MooER](https://github.com/smthemex/ComfyUI_MooER)


7 Citation
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

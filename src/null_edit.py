import os, pdb

import argparse
import numpy as np
import torch
import requests
import glob
from PIL import Image
import re

from diffusers import DDIMScheduler
from utils.edit_directions import construct_direction
from utils.pipe_null import EditingPipeline
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inversion', required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--task_name', type=str, default='cat2dog')
    parser.add_argument('--results_folder', type=str, default='output/test_cat')
    parser.add_argument('--num_ddim_steps', type=int, default= 60)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--xa_guidance', default=0.1, type=float)
    parser.add_argument('--negative_guidance_scale', default=5.0, type=float)
    parser.add_argument('--use_float_16', action='store_true')

    args = parser.parse_args()

    os.makedirs(os.path.join(args.results_folder, "edit"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "reconstruction"), exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # if the inversion is a folder, the prompt should also be a folder
    
    print(args.inversion,args.prompt)
    assert (os.path.isdir(args.inversion)==os.path.isdir(args.prompt)), "If the inversion is a folder, the prompt should also be a folder"
    if os.path.isdir(args.inversion):
        l_inv_paths = sorted(glob.glob(os.path.join(args.inversion, "*.pt")))
        l_bnames = [os.path.basename(x) for x in l_inv_paths]
        l_prompt_paths = [os.path.join(args.prompt, x.replace(".pt",".txt")) for x in l_bnames]
    else:
        l_inv_paths = [args.inversion]
        l_prompt_paths = [args.prompt]

    # Make the editing pipeline
    pipe = EditingPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype,use_auth_token=True)
    pipe = pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    direction_sclae = 1.0
    os.makedirs(os.path.join(args.results_folder, "edit"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "reconstruction"), exist_ok=True)
    lis = os.listdir(os.path.join(args.results_folder, "edit"))
    rec_list = os.listdir(os.path.join(args.results_folder, "reconstruction"))
    l_inv_paths.sort(key=lambda l: int(re.findall('\d+', l)[0]))
    l_prompt_paths.sort(key=lambda l: int(re.findall('\d+', l)[0]))
    for inv_path, prompt_path in zip(l_inv_paths, l_prompt_paths):
        bname = os.path.basename(inv_path).split(".")[0]
        prompt_str = open(prompt_path).read().strip()
        rec_pil, edit_pil = pipe(prompt_str,
                num_inference_steps=args.num_ddim_steps,
                x_in=torch.load(inv_path),
                edit_dir=direction_sclae*construct_direction(args.task_name),
                guidance_amount=args.xa_guidance,
                guidance_scale=args.negative_guidance_scale,
                results_folder=args.results_folder,
                negative_prompt=prompt_str, 
        )
        edit_pil[0].save(os.path.join(args.results_folder, f"edit/{bname}"+".png"))
        rec_pil[0].save(os.path.join(args.results_folder, f"reconstruction/{bname}"+".png"))
        del  edit_pil
        del rec_pil
        

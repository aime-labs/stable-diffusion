import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import io

from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from aime_api_worker_interface import APIWorkerInterface
from typing import Tuple

import requests

WORKER_JOB_TYPE = "stable_diffusion_txt2img"
WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d4"


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--latent_channels",
        type=int,
        help="latent channels",
    )
    parser.add_argument(
        "--downsampling_factor",
        type=int,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stable-diffusion-v1-5-emaonly/v1-5-pruned-emaonly.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--api_server",
        type=str,
        default="http://0.0.0.0:7777",
        help="Address of the API server",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, required=False,
        help="ID of the GPU to be used"
    )
    opt = parser.parse_args()
    #local_rank, world_size = setup_model_parallel(opt)
    local_rank, world_size = 0,1
    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    #if opt.dpm_solver:
    #    sampler = DPMSolverSampler(model)
    #elif opt.plms:
    sampler = PLMSSampler(model)
    #else:
    #    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))


    """
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))
    """
    api_worker = APIWorkerInterface(opt.api_server, WORKER_JOB_TYPE, WORKER_AUTH_KEY, opt.gpu_id, world_size=world_size, rank=local_rank)
    callback = ProcessOutputCallback(local_rank, api_worker)
    progress_callback = ProgressCallback(api_worker)
    while True:    
        try:
            prompt = []
            
            job_data = api_worker.job_request()
            callback.job_data = job_data
            progress_callback.job_data = job_data
            height = get_parameter('height', int, 512, opt, job_data, local_rank)
            width = get_parameter('width', int, 512, opt, job_data, local_rank)
            downsampling_factor = get_parameter('downsampling_factor', int, 8, opt, job_data, local_rank)
            latent_channels = get_parameter('latent_channels', int, 4, opt, job_data, local_rank)
            ddim_eta = get_parameter('ddim_eta', float, 0.0, opt, job_data, local_rank)
            n_samples = get_parameter('n_samples', int, 1, opt, job_data, local_rank)
            scale = get_parameter('scale', float, 7.5, opt, job_data, local_rank)
            n_rows = get_parameter('n_rows', int, 0, opt, job_data, local_rank)
            n_rows = n_rows if n_rows > 0 else n_samples
            ddim_steps = get_parameter('ddim_steps', int, 50, opt, job_data, local_rank)

            if local_rank == 0:
                callback.job_data = job_data
                prompt = job_data['text']
                data = [n_samples * [prompt]]
            
            #torch.distributed.barrier()    # not useable! Does active CPU waiting and times out with an error after about 30 minutes!

            #torch.distributed.broadcast_object_list(data, 0)


            #results = generator.generate(
            #    callback.process_output, prompts, max_gen_len=512, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=args.repetition_penalty
            #)

            #ctx = results[0]

            sample_path = os.path.join(outpath, "samples")
            os.makedirs(sample_path, exist_ok=True)
            base_count = len(os.listdir(sample_path))
            grid_count = len(os.listdir(outpath)) - 1

            start_code = None
            if opt.fixed_code:
                start_code = torch.randn([n_samples, latent_channels, height // downsampling_factor, width // downsampling_factor], device=device)

            precision_scope = autocast if opt.precision=="autocast" else nullcontext

            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        tic = time.time()
                        all_samples = list()
                        list_images = list()
                        for n in trange(opt.n_iter, desc="Sampling"):

                            for prompts in tqdm(data, desc="data"):

                                uc = None
                                if scale != 1.0:
                                    uc = model.get_learned_conditioning(n_samples * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)
                                c = model.get_learned_conditioning(prompts)
                                shape = [latent_channels, height // downsampling_factor, width // downsampling_factor]
                                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                                conditioning=c,
                                                                batch_size=n_samples,
                                                                shape=shape,
                                                                callback=progress_callback,
                                                                verbose=False,
                                                                unconditional_guidance_scale=scale,
                                                                unconditional_conditioning=uc,
                                                                eta=ddim_eta,
                                                                x_T=start_code
                                                                )

                                x_samples_ddim = model.decode_first_stage(samples_ddim)
                                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                                x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                                #if not opt.skip_save:
                                #    for x_sample in x_checked_image_torch:
                                #        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                #        img = Image.fromarray(x_sample.astype(np.uint8))
                                #        img = put_watermark(img, wm_encoder)
                                #        img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                #        base_count += 1

                                if not opt.skip_grid:
                                    all_samples.append(x_checked_image_torch)
                                
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    image = Image.fromarray(x_sample.astype(np.uint8))
                                    image = put_watermark(image, wm_encoder)
                                    list_images.append(image)
                       
                        
                                
                        if not opt.skip_grid:
                            # additionally, save as grid
                            grid = torch.stack(all_samples, 0)
                            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                            grid = make_grid(grid, nrow=n_rows)

                            # to image
                            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                            img = Image.fromarray(grid.astype(np.uint8))
                            img = put_watermark(img, wm_encoder)
                            img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                            grid_count += 1

                        toc = time.time()
        except RuntimeError as exc:
            image = Image.fromarray((np.random.rand(512,512,3) * 255).astype('uint8')).convert('RGBA')
            callback.process_output(image, str(exc))
            continue
        except ValueError as exc:
            image = Image.fromarray((np.random.rand(512,512,3) * 255).astype('uint8')).convert('RGBA')
            callback.process_output(image, str(exc))
            continue
        
        callback.process_output(list_images[0], f'Prompt: {data[0][0]}')


class ProcessOutputCallback():
    def __init__(self, local_rank, api_worker):
        self.local_rank = local_rank
        self.api_worker = api_worker
        self.job_data = None

    def process_output(self, output, info):
        if self.local_rank == 0:
            results = {'image': output, 'info':info}
            return self.api_worker.send_job_results(self.job_data, results)



class ProgressCallback():
    def __init__(self, api_worker):
        self.api_worker = api_worker
        self.job_data = None

    def send_progress_to_api_server(self, progress, progress_data=None):
        self.api_worker.send_progress(self.job_data, progress, progress_data)



def get_parameter(parameter_name, parameter_type, default_value, args, job_data, local_rank):
    parameter = default_value
    if local_rank == 0:
        if getattr(args, parameter_name) is not None:
            parameter = getattr(args, parameter_name)
        elif parameter_type(job_data[parameter_name]) is not None:
            parameter = parameter_type(job_data[parameter_name]) 
    parameter_list = [parameter]
    #torch.distributed.broadcast_object_list(parameter_list, 0)
    print('parameter', parameter_name, parameter, parameter_type(job_data[parameter_name]))
    return parameter_list[0]


def setup_model_parallel(args) -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank+args.gpu_id)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


if __name__ == "__main__":
    main()

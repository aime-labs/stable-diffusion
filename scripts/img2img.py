"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
import requests
import base64
import io

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from aime_api_worker_interface import APIWorkerInterface

WORKER_JOB_TYPE = "stable_diffusion_img2img"
WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d5"

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


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


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def convert_img(image):

    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


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
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
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
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
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
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    world_size = 1
    local_rank = 0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir


    """
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))
    """

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    grid_count = len(os.listdir(outpath)) - 1




    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    api_worker = APIWorkerInterface(opt.api_server, WORKER_JOB_TYPE, WORKER_AUTH_KEY, opt.gpu_id, world_size=world_size, rank=local_rank)
    callback = ProcessOutputCallback(api_worker)
    
    #api_server_is_online = True
    while True:
        
        try:
            job_data = api_worker.job_request()
            callback.job_data = job_data
            progress_bar_callback.job_data = job_data
            prompt = job_data['text']
            batch_size = job_data['n_samples']
            data = [batch_size * [prompt]]
            #height = job_data['height']
            #width = job_data['width']
            #downsampling_factor = job_data['downsampling_factor']
            #latent_channels = job_data['latent_channels']
            strength = job_data['strength']
            ddim_eta = job_data['ddim_eta']

            scale = job_data['scale']
            n_rows = job_data['n_rows']
            ddim_steps = job_data['ddim_steps']
            init_image_base64 = job_data['image']
            try:
                init_image = convert_base64_string_to_image(init_image_base64).to(device)
            except AttributeError:
                image = Image.fromarray((np.random.rand(512,512,3) * 255).astype('uint8')).convert('RGBA')
                callback.process_output(image, 'No init image found.')
                continue
                   
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

            sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

            t_enc = int(strength * ddim_steps)
            print(f"target t_enc is {t_enc} steps")

            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        tic = time.time()
                        all_samples = list()
                        all_images = list()
                        for n in trange(opt.n_iter, desc="Sampling"):
                            for prompts in tqdm(data, desc="data"):
                                uc = None
                                if scale != 1.0:
                                    uc = model.get_learned_conditioning(batch_size * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)
                                c = model.get_learned_conditioning(prompts)

                                # encode (scaled latent)
                                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                                # decode it
                                samples = sampler.decode(z_enc, c, t_enc, progress_callback=callback.progress_bar_callback, unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=uc)

                                x_samples = model.decode_first_stage(samples)
                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)


                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    image = Image.fromarray(x_sample.astype(np.uint8))
                                    all_images.append(image)
                        """
                        if not opt.skip_grid:
                            # additionally, save as grid
                            grid = torch.stack(all_samples, 0)
                            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                            grid = make_grid(grid, nrow=n_rows)

                            # to image
                            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                            Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                            grid_count += 1
                        """
                        toc = time.time()
        except RuntimeError as exc:
            image = Image.fromarray((np.random.rand(512,512,3) * 255).astype('uint8')).convert('RGBA')
            callback.process_output(image, str(exc))
            continue
        except ValueError as exc:
            image = Image.fromarray((np.random.rand(512,512,3) * 255).astype('uint8')).convert('RGBA')
            callback.process_output(image, str(exc))
            continue

        callback.process_output(all_images[0], f'Prompt: {data[0][0]}')

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")


class ProcessOutputCallback():
    def __init__(self, api_worker):
        self.api_worker = api_worker
        self.job_data = None

    def process_output(self, output, info):
        results = {'image': output, 'info':info}
        return self.api_worker.send_job_results(self.job_data, results)

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
    return parameter_list[0]

def convert_base64_string_to_image(base64_string):
    base64_data = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_data)

    with io.BytesIO(image_data) as buffer:
        image = convert_img(Image.open(buffer))
    return image


if __name__ == "__main__":
    main()

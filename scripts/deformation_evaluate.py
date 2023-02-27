"""
Train a deformation network to quantify the brain midline shift.
"""
import argparse
import copy
import torch as th
import torch.distributed as dist
from torch.optim import AdamW
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data_slice_epoch, load_data_slice_lm
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.nn import mean_flat, max_flat

import os

from torch.nn import functional as F
import numpy as np
from morph.network import SpatialTransformer, Unet, VecInt, smooth_loss
import torchvision.transforms.functional as TF

def main():
    '''

    The implementation are based on single GPU. Haven't tested for multiple GPUs.

    '''
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    dif_model_con, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    dif_model_con.load_state_dict(
        dist_util.load_state_dict(args.model_con_path,
                                  map_location="cpu")
    )
    dif_model_con.to(dist_util.dev())
    dif_model_con.eval()
    dif_model_uncon = copy.deepcopy(dif_model_con)
    args.timestep_respacing = "ddim50"
    args.use_ddim = True
    _, diffusion_sample = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    dif_model_uncon.load_state_dict(
        dist_util.load_state_dict(args.model_uncon_path,
                                  map_location="cpu")
    )
    dif_model_uncon.to(dist_util.dev())
    dif_model_uncon.eval()

    deform_model = Unet([args.image_size, args.image_size], 2)
    deform_model.to(dist_util.dev())
    deform_model.load_state_dict(
        dist_util.load_state_dict(args.model_path,
                                  map_location="cpu")
    )
    deform_model.eval()

    # Needed for creating correct EMAs and fp16 parameters.
    logger.log("creating data loader...")
    if args.val_data_dir:
        val_data = load_data_slice_lm(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            landmark=True,
            class_cond=True,
            return_loader=True
        )
    else:
        val_data = None

    vecint = VecInt([args.image_size, args.image_size], 7).cuda()
    def validation_forward(data_loader):
        deform_model.eval()
        with th.no_grad():
            error_detach = 0
            data_size = 0
            for idx, (batch, extra) in enumerate(data_loader):
                data_size += len(batch)
                masks = extra["mask"].to(dist_util.dev())
                landmarks = extra["landmarks"].to(dist_util.dev())
                labels = extra["mls"].to(dist_util.dev())
                batch = batch.to(dist_util.dev())
                for i, (sub_batch, sub_masks, sub_landmarks, sub_labels) in enumerate(
                        split_microbatches(args.microbatch, batch, masks, landmarks, labels)
                ):
                    t = th.tensor([300]).cuda()
                    batch_noise = th.randn_like(sub_batch).cuda()
                    batch_noisy = diffusion.q_sample(sub_batch.cuda(), t, noise=batch_noise)
                    noise_pred_con = dif_model_con(batch_noisy, t)[:, :1].detach()
                    noise_pred_uncon = dif_model_uncon(batch_noisy, t)[:, :1].detach()
                    velocity_field = deform_model(th.cat([sub_batch, noise_pred_con - noise_pred_uncon], dim=1))
                    deform_field = vecint(velocity_field)
                    pred_mls = max_flat(th.sqrt(deform_field[:, 0] ** 2 + deform_field[:, 1] ** 2))
                    error = th.sum(th.abs(pred_mls - sub_labels))
                    error_detach += error.detach().cpu()
            logger.log("val_error: {}".format(error_detach / data_size))
        return error_detach / data_size

    val_loss = validation_forward(val_data)
    logger.log(("final val error: {}".format(val_loss)))
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr

def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        val_data_dir="",
        model_path="",
        model_con_path="",
        model_uncon_path="",
        noised=True,
        iterations=500000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=True,
        image_size = 256,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=2000,
        lamb=0.001,
        clip_denoised=True,
    )
    defaults.update(classifier_and_diffusion_defaults())
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()


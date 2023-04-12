"""
Train a deformation network to quantify the brain midline shift.
"""
import sys
sys.path.append("../")
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
    deform_model.train()

    # Needed for creating correct EMAs and fp16 parameters.
    logger.log("creating data loader...")
    data = load_data_slice_epoch(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=False,
        random_flip = False,
        landmark=True,
        return_loader=True
    )
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

    logger.log(f"creating optimizer...")
    opt_deform = AdamW(deform_model.parameters(), lr=args.lr, weight_decay=0)
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, 1000
    )
    logger.log("training classifier model...")
    transformer = SpatialTransformer([args.image_size, args.image_size]).cuda()
    vecint = VecInt([args.image_size, args.image_size], 7).cuda()
    def model_fn(x, t):
        return 3 * dif_model_con(x, t) - 2 * dif_model_uncon(x, t)

    sample_fn = diffusion_sample.ddim_sample_loop
    def forward_backward_log(data_loader,  epoch = 0):
        deform_model.train()
        th.cuda.empty_cache()
        loss_detach = 0
        loss_smooth_detach = 0
        loss_deformation_detach = 0
        loss_hinge_detach = 0
        loss_mse_detach = 0
        data_size = 0
        for idx, (batch, extra) in enumerate(data_loader):
            masks = extra["mask"].to(dist_util.dev())
            landmarks = extra["landmarks"].to(dist_util.dev())
            labels = extra["mls"].to(dist_util.dev())
            batch = batch.to(dist_util.dev())
            if args.noised:
                t, weight = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            else:
                t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
            for i, (sub_batch, sub_masks, sub_landmarks, sub_labels) in enumerate(
                split_microbatches(args.microbatch, batch, masks, landmarks, labels)
            ):
                data_size += len(sub_batch)
                angle = np.random.rand() * 30. - 15.
                #sub_batch_rot = TF.rotate(sub_batch, angle)
                sub_batch_rot = sub_batch
                with th.no_grad():
                    t_sample = th.ones(sub_batch.shape[0], dtype=th.long, device=dist_util.dev()) * 15
                    batch_noise = th.randn_like(sub_batch).cuda()
                    batch_noisy = diffusion_sample.q_sample(sub_batch, t_sample, noise=batch_noise).cuda()
                    pseudo_fix = sample_fn(
                        model_fn,
                        (args.batch_size, 1, args.image_size, args.image_size),
                        noise=batch_noisy,
                        clip_denoised=args.clip_denoised,
                        model_kwargs={},
                        cond_fn=None,
                        device=dist_util.dev(),
                        t=th.tensor([15])
                    )

                    batch_noisy = diffusion.q_sample(sub_batch_rot.cuda(), t, noise=batch_noise)
                    noise_pred_con = dif_model_con(batch_noisy, t)[:, :1].detach()
                    noise_pred_uncon = dif_model_uncon(batch_noisy, t)[:, :1].detach()

                velocity_field = deform_model(th.cat([sub_batch_rot, noise_pred_con - noise_pred_uncon], dim=1))
                deform_field = vecint(velocity_field)
                #deform_field = TF.rotate(deform_field, -angle)
                deform_grid = 2 * sub_landmarks[:, [0, 1, 2], :].unsqueeze(1).float() / 255 - 1
                pred_deform = F.grid_sample(deform_field,
                                            deform_grid,
                                            mode='bilinear')
                true_deform = th.zeros([len(sub_batch), 2, 1, 3]).cuda()
                true_deform[:, 0, 0, 2] = sub_landmarks[:, 3, 1] - sub_landmarks[:, 2, 1]
                true_deform[:, 1, 0, 2] = sub_landmarks[:, 3, 0] - sub_landmarks[:, 2, 0]
                true_deform = true_deform.reshape([len(true_deform), -1])
                pred_deform = pred_deform.reshape([len(true_deform), -1])

                l1_norm = th.abs(true_deform - pred_deform)
                l2_norm = ((true_deform - pred_deform) ** 2 + 9.) / 6.
                smooth_norm = l1_norm * (l1_norm > 3).float() + l2_norm * (l1_norm <= 3).float()
                loss_deformation = mean_flat((smooth_norm))

                loss_smooth = mean_flat(smooth_loss(deform_field))
                deformed_batch = transformer(pseudo_fix, deform_field)
                loss_mse = mean_flat(sub_masks * (sub_batch - deformed_batch) ** 2)

                deform_max = th.sqrt(deform_field[0, 1] ** 2 + deform_field[0, 0] ** 2)
                sub_batch_max = sub_labels.reshape(-1, 1, 1, 1)
                loss_hinge = mean_flat((deform_max > sub_batch_max) * (deform_max - sub_batch_max))

                loss = loss_mse * (1+epoch/20) + loss_smooth + loss_deformation + loss_hinge
                loss_detach += th.sum(loss.detach().cpu())
                loss_smooth_detach += th.sum(loss_smooth.detach().cpu())
                loss_deformation_detach += th.sum(loss_deformation.detach().cpu())
                loss_hinge_detach += th.sum(loss_hinge.detach().cpu())
                loss_mse_detach += th.sum(loss_mse.detach().cpu())

                loss = th.mean(loss)
                opt_deform.zero_grad()
                loss.backward()
                opt_deform.step()

        logger.log("epoch: {}".format(epoch))
        logger.log("loss_smooth: {}".format(loss_smooth_detach / data_size))
        logger.log("loss_deformation: {}".format(loss_deformation_detach / data_size))
        logger.log("loss_hinge: {}".format(loss_hinge_detach/data_size))
        logger.log("loss_ncc: {}".format(loss_mse_detach / data_size))
        logger.log("loss: {}".format(loss_detach / data_size))

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

    val_loss = np.inf
    for epoch in range(100):
        forward_backward_log(data, epoch=epoch)
        current_loss = validation_forward(val_data)
        if current_loss < val_loss:
            val_loss = current_loss
            save_model(deform_model, opt_deform, "deformation")
    logger.log(("final val error: {}".format(val_loss)))
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, name):
    th.save(
        mp_trainer.state_dict(),
        os.path.join(logger.get_dir(), "{}_model.pt".format(name)),)
    th.save(opt.state_dict(), os.path.join(logger.get_dir(), "{}_opt.pt".format(name)))


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
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


import jax
import jax.numpy as jnp
import cv2
import numpy as np
import dm_pix as dmp

import chest_xray_sim.inverse.operators as ops
from chest_xray_sim.inverse.operators import (
    mse,
    total_variation,
    build_loss,
    build_forward_fn,
)
import wandb
from chest_xray_sim.inverse.core import base_optimize
from chest_xray_sim.utils.metrics import naive_inversion
from chest_xray_sim.utils.results import run_metrics
import itertools
import argparse
from utils import (
    build_optim_runner,
    experiment_args,
    get_sweep_step,
    log_image,
    log_run_metrics,
    random_initialization,
)

parser = experiment_args(
    raw="data/conventional_transmissionmap.tif",
    target="data/conventional_processed.tif",
)

if __name__ == "__main__":
    args = parser()

    true_raw = cv2.imread(args.raw, cv2.IMREAD_UNCHANGED)
    true_raw = true_raw / 255.0

    target = cv2.imread(args.target, cv2.IMREAD_UNCHANGED)
    target = target / 255.0

    if true_raw.shape != target.shape:
        if true_raw.shape[0] < target.shape[0]:
            target = cv2.resize(target, (true_raw.shape[1], true_raw.shape[0]))
        else:
            true_raw = cv2.resize(true_raw, (target.shape[1], target.shape[0]))

    true_raw = jnp.expand_dims(true_raw, axis=2)
    target = jnp.expand_dims(target, axis=2)
    #
    wandb.login()

    dims = true_raw.shape

    forward_op_builder = lambda _: build_forward_fn(
        ops.negative_log,
        # ops.unsharp_masking_("low_"),
        # ops.unsharp_masking_("high_"),
        ops.windowing,
        ops.range_normalize,
    )
    # print('forward:', forward_op_builder(()).desc)
    loss_fn_builder = lambda hp: build_loss(mse, total_variation(hp["total_variation"]))
    # print('loss_fn:', loss_fn_builder({'total_variation': 1.0}))
    random_init_bounds = dict(
        image={"min": 1e-6, "max": 1.0, "shape": true_raw.shape},
        window_center={"min": 0.1, "max": 0.9},
        window_width={"min": 0.1, "max": 0.9},
        gamma={"min": 0.1, "max": 3.0},
        low_sigma={"min": 0.1, "max": min(dims) / 2},
        low_enhance_factor={"min": 0.0, "max": 3.0},
        high_sigma={"min": 0.1, "max": min(dims)},
        high_enhance_factor={"min": 0.0, "max": 3.0},
    )
    # print('random_init_bounds', random_init_bounds)

    # def init(hp):
    #     print('initing with hyperparams:', hp)
    #     return random_initialization(
    #         jax.random.PRNGKey(hp["PRNGKey"]), **random_init_bounds
    #     )

    init = lambda hp: random_initialization(
        jax.random.PRNGKey(hp["PRNGKey"]), **random_init_bounds
    )


    weights = init({"PRNGKey": 0})
   # print('weights:', weights)
    #

    # forwarded = forward_op_builder(weights)(weights)
    # print(forwarded.shape, type(forwarded))
    # print(forwarded)
    # cv2.imwrite("forwarded.png", np.array(forwarded.squeeze() * 255, dtype=np.uint8)) 



    project = "unsharp_mask_uknown_processing"
    notes = "Add unsharp masking, optimize single image, unknown processing pipeline"

    sweep_config = {
        "name": "hyperparam_search",
        "method": "bayes",
        "metric": {"name": "mse_tm", "goal": "minimize"},
        "parameters": {
            "PRNGKey": {"value": 0},
            "lr": {"min": 1e-5, "max": 1e-1},
            "n_steps": {"values": [500, 1000, 2000]},
            "total_variation": {"values": [0.0, 0.1, 0.01, 0.001]},
            # Fixed/metadata parameters
            "loss": {"value": "MSE + TV"},
            "eps": {"value": 1e-8},
        },
        "meta": random_init_bounds,
    }

    with wandb.init(project=project, name="global_run", tags=["global"]) as global_run:
        naive = naive_inversion(target)

        mse_naive = dmp.mse(true_raw, naive)
        ssim_naive = dmp.ssim(true_raw, naive)
        psnr_naive = dmp.psnr(true_raw, naive)

        sqerr_naive = (true_raw - naive) ** 2

        global_run.log(
            {
                "naive_sqerr": wandb.Histogram(sqerr_naive),
                "naive_mse": mse_naive,
                "naive_ssim": ssim_naive,
                "naive_psnr": psnr_naive,
            }
        )

        log_image("naive", naive.squeeze())
        log_image("true_raw", true_raw.squeeze())
        log_image("target", target.squeeze())

    def logging(recovered, pred, loss):
        print('recovered stats', recovered['image'].min().item(), recovered['image'].max().item(), recovered['image'].mean().item())

        mse_tm = dmp.mse(true_raw, recovered["image"])
        ssim_tm = dmp.ssim(true_raw, recovered["image"])
        psnr_tm = dmp.psnr(true_raw, recovered["image"])

        mse_processed = dmp.mse(target, pred)
        ssim_processed = dmp.ssim(target, pred)
        psnr_processed = dmp.psnr(target, pred)
        

        data = dict(
                mse_tm=mse_tm,
                ssim_tm=ssim_tm,
                psnr_tm=psnr_tm,
                mse_processed=mse_processed,
                ssim_processed=ssim_processed,
                psnr_processed=psnr_processed,
                loss=loss,
            )

        data = {k: v.item() for k,v in data.items()}

        wandb.log(
            data
        )

    def end_cb(true_raw, target, recovered, forward, loss_fn):
        sqerr = (true_raw - recovered["image"]) ** 2
        fwd = forward(recovered)
        forward_sqerr = (target - fwd) ** 2

        wandb.log(
            {
                "sqerr_tm": wandb.Histogram(sqerr),
                "sqerr_processed": wandb.Histogram(forward_sqerr),
            }
        )

    sweep_agent = get_sweep_step(
        true_raw,
        target,
        init,
        forward_op_builder,
        loss_fn_builder,
        end_cb=end_cb,
        loss_cb=logging,
        run_config={"project": project},
    )

    sweep_id = wandb.sweep(sweep=sweep_config, project=project)
    wandb.agent(sweep_id, function=sweep_agent, count=50)

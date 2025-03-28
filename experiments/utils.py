import argparse
import functools

import jax
import numpy as np
import wandb

from chest_xray_sim.inverse.core import base_optimize


BIT_DTYPES = {8: np.uint8, 16: np.uint16}


def experiment_args(**arguments):
    parser = argparse.ArgumentParser()
    for arg, default in arguments.items():
        parser.add_argument(f'--{arg}', type=str, default=default)

    def parse_args():
        return parser.parse_args()

    return parse_args


def log_image(label, img, bits=8):
    rng = 2**bits - 1
    dtype = BIT_DTYPES[bits]
    wandb.log({label: wandb.Image(np.array(img * rng, dtype=dtype))})


def log_run_metrics(run, metrics, prefix):
    prefixed_metrics = {f"{prefix}{key}": value for key, value in metrics.items()}
    for key, value in prefixed_metrics.items():
        run.summary[key] = value


def random_initialization(key, **params):

    state = {
        k: jax.random.uniform(
            key, minval=v["min"], maxval=v["min"], shape=v.get("shape", ())
        )
        for k, v in params.items()
    }

    return state


def dict_merge(*dicts):
    return functools.reduce(lambda a, b: {**a, **b}, dicts)


def build_optim_runner(w0_builder, operator, loss_fn, **opt_params_high):


    def run_optimization(target, hyperparams, **opt_params):
        w0 = w0_builder(hyperparams)

        def_opt_params = dict(
            target=target,
            w0=w0,
            lr=hyperparams["lr"],
            n_steps=hyperparams["n_steps"],
            forward_fn=operator,
            loss_fn=loss_fn,
        )


        operator_params = operator.keys
        missing_params = set(operator_params) - set(w0.keys())
        assert (
            len(missing_params) == 0
        ), "Initial weights missing for operator params: " + ", ".join(missing_params)


        opt_params = dict_merge(def_opt_params, opt_params_high, opt_params)

        print ('Optimization params: ', {k:v for k,v in opt_params.items() if k not in ['target', 'w0']})
        print('initial weights:', {k:v for k,v in w0.items() if k not in ['image']})
        print('image stats:', w0['image'].mean())
        print('targetstats:', target.mean())
        return base_optimize(**opt_params)

    return run_optimization


def get_sweep_step(
    true_raw,
    target,
    weight_init,
    forward_op_builder,
    loss_fn_builder,
    end_cb=None,
    run_config={},
    **opt_params,
):

    def optimize():
        with wandb.init(**run_config) as run:
            forward_op = forward_op_builder(run.config)
            loss_fn = loss_fn_builder(run.config)

            # run.tags.append(forward_op.desc)

            run_optimization = build_optim_runner(
                weight_init,
                forward_op,
                loss_fn,
                **opt_params,
            )


            recovered, _ = run_optimization(target, run.config)
            # recovere

            if end_cb:
                end_cb(true_raw, target, recovered, forward=forward_op, loss_fn=loss_fn)

    return optimize

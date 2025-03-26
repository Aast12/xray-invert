import jax
import jax.numpy as jnp
from functools import wraps


def parametereized_decorator(schema):
    def parameterized_fn(fn):

        @wraps(fn)
        def with_params(*args, **kwargs):
            return fn(*args, **kwargs)

        with_params.schema = schema

        return with_params

    return parameterized_fn


def negative_log(image, weights):
    return -jnp.log(jnp.maximum(image, 1e-6))


@parametereized_decorator(["window_center", "window_width", "gamma"])
def windowing(image, weights):
    x = image
    x = (x - weights["window_center"]) / weights["window_width"]
    x = jax.nn.sigmoid(x) ** weights["gamma"]

    return x


def clipping(image, weights):
    return jnp.clip(image, 0.0, 1.0)


def range_normalize(image, weights):
    x = image
    return (x - x.min()) / (x.max() - x.min())


def max_normalize(image, weights):
    x = image
    return x / x.max()


def build_forward_fn(*pipeline):
    def get_schema(fn):
        return getattr(fn, "schema", ["image"])

    keys = sum(map(get_schema, pipeline), [])
    keys = list(set(keys))

    def forward_fn(weights):
        x = weights["image"]
        for proc in pipeline:
            x = proc(x, weights)
        return x

    forward_fn.keys = keys

    return forward_fn


def mse(w, pred, target):
    return jnp.mean((pred - target) ** 2)


def total_variation(factor):
    def _total_variation(w, pred, target):
        reg = (jnp.diff(w["image"], axis=0) ** 2).mean() + (
            jnp.diff(w["image"], axis=1) ** 2
        ).mean()
        return 0.5 * factor * reg

    return _total_variation


def build_loss(*pipeline):
    def loss_fn(weights, pred, target):
        loss = 0
        for proc in pipeline:
            loss += proc(weights, pred, target)
        return loss

    return loss_fn

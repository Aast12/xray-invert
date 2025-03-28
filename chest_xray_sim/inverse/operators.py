import jax
import jax.numpy as jnp
from functools import wraps
from dm_pix import gaussian_blur


def parameterized(schema):
    def parameterized_fn(fn):

        @wraps(fn)
        def with_params(image, weights):
            return fn(image, **weights)

        with_params.schema = schema

        return with_params

    return parameterized_fn


def prefixed_parameterized(schema):
    def parameterized_fn(fn):
        def prefixed_fn(prefix):

            @wraps(fn)
            def with_params(image, weights):
                kwargs = {
                    k[len(prefix) :]: v
                    for k, v in weights.items()
                    if k.startswith(prefix)
                }

                return fn(image, **kwargs)

            with_params.__name__ = f'{prefix}{fn.__name__}'
            with_params.schema = [f"{prefix}{s}" for s in schema]

            return with_params

        return prefixed_fn

    return parameterized_fn


def schemaless(fn):
    return parameterized([])(fn)


@schemaless
def negative_log(image):
    return -jnp.log(jnp.maximum(image, 1e-6))


@parameterized(["window_center", "window_width", "gamma"])
def windowing(image, window_center, window_width, gamma):
    x = image
    x = (x - window_center) / window_width
    x = jax.nn.sigmoid(x) ** gamma

    return x


@prefixed_parameterized(["sigma", "enhance_factor"])
def unsharp_masking(image, sigma, enhance_factor):
    x = jnp.expand_dims(image, axis=2)
    kernel_size = int(4 * sigma) + 1
    blurred = gaussian_blur(x, sigma, kernel_size, padding="SAME")
    x = x + enhance_factor * (x - blurred)
    return x.squeeze()


@schemaless
def clipping(image):
    return jnp.clip(image, 0.0, 1.0)


@schemaless
def range_normalize(image):
    x = image
    return (x - x.min()) / (x.max() - x.min())


@schemaless
def max_normalize(image):
    x = image
    return x / x.max()


def build_forward_fn(*pipeline):
    def get_schema(fn):
        return getattr(fn, "schema", ["image"])

    keys = sum(map(get_schema, pipeline), [])
    keys = list(set(keys))

    operator_desc = "|".join([pipeline.__name__ for pipeline in pipeline])

    def forward_fn(weights):
        x = weights["image"]
        for proc in pipeline:
            x = proc(x, {k: weights[k] for k in proc.schema})
        return x

    forward_fn.desc = operator_desc
    forward_fn.keys = keys

    return forward_fn


def build_forward_fn_and_weights(*pipeline):
    forward = build_forward_fn(*pipeline)
    weights = {k: 1.0 for k in forward.keys}

    return forward, weights


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

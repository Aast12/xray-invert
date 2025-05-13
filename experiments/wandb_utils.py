import wandb


def build_table(data):
    table = wandb.Table(data=data)

    priors_table = wandb.Table(columns=["region", "min", "max"])
    for region, value_range in zip(seg_labels, value_ranges):
        min_val, max_val = value_range
        priors_table.add_data(region, min_val, max_val)

    wandb.log({"priors": priors_table})

import torch
import torch.distributed as dist


def all_reduce_avg(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


class color:
    """
    courtesy - https://gist.github.com/nazwadi/ca00352cd0d20b640efd
    """

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def pretty_log(
    iteration,
    total_train_iters,
    train_loss,
    elapsed_time_per_iteration,
    grad_norm,
    learning_rate,
):
    log_string = "> global batch {:8d}/{:8d} |".format(iteration, total_train_iters)
    log_string += " elapsed time per global batch (ms): {:.1f} |".format(
        elapsed_time_per_iteration
    )
    log_string += " learning rate: {:.3E} |".format(learning_rate)
    log_string += " loss: {:.5f} |".format(train_loss)
    curr_mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    log_string += " memory used by tensors {:.3f} GB (peak {:.3f} GB) |".format(
        curr_mem, peak_mem
    )
    log_string += " grad_norm: {:.2f}".format(grad_norm)
    return log_string

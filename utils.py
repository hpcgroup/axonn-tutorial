import torch.distributed
from transformers import AutoConfig
import transformers.models as models


def all_reduce_avg(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= torch.distributed.get_world_size()
    return tensor


def get_fsdp_block_name_for_hf(model_id):
    MODEL_ID_TO_DECODER_CLASS_MAP = {
        "LlamaForCausalLM": models.llama.modeling_llama.LlamaDecoderLayer,
        "PhiForCausalLM": models.phi.modeling_phi.PhiDecoderLayer,
    }
    config = AutoConfig.from_pretrained(model_id)
    architecture = config.architectures[0]
    assert (
        architecture in MODEL_ID_TO_DECODER_CLASS_MAP
    ), f"please add the name of the decoder layer of {architecture} to MODEL_ID_TO_DECODER_CLASS_MAP"
    return MODEL_ID_TO_DECODER_CLASS_MAP[architecture]

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


class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.level_color_map = [color.PURPLE, color.GREEN, color.CYAN, color.RED]

    def add_children(self, child):
        self.children.append(child)

    def __str__(self, level=0):
        this_color = self.level_color_map[level % len(self.level_color_map)]
        ret = (
            "\t" * level + "|--" + f"{this_color} {repr(self.value)} {color.END}" + "\n"
        )
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return f"\n{self.value}"


def print_axonn_timer_data(times):
    sorted_call_stacks = list(times.keys())
    sorted_call_stacks.sort(key=lambda x: len(x))
    head_nodes = []
    node_map = {}
    for call_stack in sorted_call_stacks:
        time = times[call_stack]
        node = Node(f"{call_stack[-1]} | {time:.3f} ms")
        node_map[call_stack] = node
        if len(call_stack) > 1:
            parent_node = call_stack[:-1]
            assert parent_node in node_map
            node_map[parent_node].add_children(node)
        else:
            head_nodes.append(node)

    for node in head_nodes:
        print(str(node))

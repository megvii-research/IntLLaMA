import torch.nn as nn
from utils.quantizer_utils import fake_quant_input


def prepare_model_for_fp16_training(model):
    # For backward compatibility
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    return model


class HookManager(list):
    def clear(self):
        for i in self:
            i.remove()
        super().clear()


def register_fake_quant_input_hook(args, model):
    hooks = HookManager()
    if args.enable_input_quant:
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear) and "lm_head" not in n:
                hooks.append(m.register_forward_pre_hook(fake_quant_input))
    return hooks

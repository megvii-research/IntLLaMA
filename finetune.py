import os
import torch
import torch.nn as nn
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
import json
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from utils import (
    QuantLinear,
    get_peft_qmodel,
    generate_alpaca_prompt,
    prepare_model_for_fp16_training,
    Prompter,
)

from evaluate_mmlu import eval_mmlu

def replace_linear2qlinear(module, prefix="", **kwargs):
    for n, m in module.named_children():
        full_name = prefix + "." + n
        if isinstance(m, nn.Linear) and "lm_head" not in n:  # hack for lm_head
            qlinear = QuantLinear(
                m.in_features,
                m.out_features,
                bit_width=kwargs.get("bit_width", None),
                groupsize=kwargs.get("groupsize", -1),
                enable_gscales_decompose=kwargs.get("enable_gscales_decompose", False),
                enable_svd_deltaW=kwargs.get("enable_svd_deltaW", False),
                svd_deltaW_rank=kwargs.get("svd_deltaW_rank", 8),
            )
            setattr(module, n, qlinear)
        new_prefix = prefix + "." + n if prefix else n
        replace_linear2qlinear(m, new_prefix, **kwargs)


def unpack_model_weight(model):
    for n, m in model.named_modules():
        if isinstance(m, QuantLinear):
            print("unpack module weight:", n)
            m.unpack()


def build_model(args, device_map="auto"):
    if args.weight_dtype == "llm_int8":
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=True,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    elif args.weight_dtype == "fp16":
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    elif args.weight_dtype == "uniform_qlora":
        assert os.path.exists(
            args.checkpoint
        ), "loading uniform_qlora model requires checkpoint"
        ckpt = torch.load(args.checkpoint)
        cvt_hyparams = ckpt["hyparams"]
        config = transformers.AutoConfig.from_pretrained(args.model_name)

        def get_non_init_llama(config):
            def skip(*args, **kwargs):
                pass

            torch.nn.init.kaiming_uniform_ = skip
            torch.nn.init.uniform_ = skip
            torch.nn.init.normal_ = skip
            transformers.modeling_utils._init_weights = False
            model = LlamaForCausalLM(config)
            return model

        model = get_non_init_llama(config)
        replace_linear2qlinear(model, **cvt_hyparams)
        model.load_state_dict(ckpt["model"])
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        unpack_model_weight(model)
        model = model.half()
    else:
        raise RuntimeError("unavaliable weight_dtype {}".format(args.weight_dtype))
    model.eval()
    return model


def build_peft_model(args, model):
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="QLORA_CAUSAL_LM"
        if args.weight_dtype == "uniform_qlora"
        else "CAUSAL_LM",
    )
    if args.weight_dtype == "uniform_qlora":
        model.is_loaded_in_8bit = True  # hack for gradient-checkpoint
        model = prepare_model_for_int8_training(model)
        model.is_loaded_in_8bit = False
        model = get_peft_qmodel(model, config)
    else:
        if args.weight_dtype == "llm_int8":
            model = prepare_model_for_int8_training(model)
        else:
            model = prepare_model_for_fp16_training(model)
        model = get_peft_model(model, config)
    return model


def build_tokenizer(args, model):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    # unk. we want this to be different from the eos token
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"  # Allow batched inference
    return tokenizer


def build_dataset(args, tokenizer):
    if args.dataset_path.endswith(".json") or args.dataset_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path)

    prompter = Prompter("alpaca")

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if False: #not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=args.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if args.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    if False:  # debug
        TRAIN_SIZE = 500  # None
        VAL_SIZE = 100  # 2000
    else:
        TRAIN_SIZE = None
        VAL_SIZE = 2000

    train_val = dataset["train"].train_test_split(
        train_size=TRAIN_SIZE, test_size=VAL_SIZE, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        #train_val["train"].map(lambda x: tokenize(generate_alpaca_prompt(x)))
    )
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    return train_data, val_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="decapoda-research/llama-7b-hf")
    parser.add_argument("--dataset_path", default="yahma/alpaca-cleaned")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="the path of converted checkpoint"
    )
    # training params
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--cutoff_len", type=int, default=256)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # lora params
    parser.add_argument("--lora_rank", type=int, required=True)
    parser.add_argument("--lora_alpha", type=float, required=True)
    parser.add_argument("--lora_dropout", type=float, required=True)
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+", default=["q_proj", "v_proj"]
    )
    # others
    parser.add_argument(
        "--weight_dtype", required=True, choices=["uniform_qlora", "llm_int8", "fp16"]
    )
    parser.add_argument(
        "--group_by_length",
        action="store_true",
        help="a flag enable group data by length",
    )
    parser.add_argument(
        "--add_eos_token",
        action="store_true",
        help="a flag enable group data by length",
    )
    parser.add_argument(
        "--enable_mmlu_evaluation",
        action="store_true",
        help="a flag enable mmlu evaluation after finetuning",
    )
    
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    device_map = "auto"

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        # show&save convert configure
        for k, v in args.__dict__.items():
            print(k + " : ", v)
        with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
            json.dump(args.__dict__, f)

    # load model
    model = build_model(args, device_map=device_map)
    tokenizer = build_tokenizer(args, model)
    model = build_peft_model(args, model)
    tokenizer.save_pretrained(args.output_dir)

    train_dataset, val_dataset = build_dataset(args, tokenizer)

    model.save_pretrained(args.output_dir)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=args.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=args.group_by_length,
            max_grad_norm=args.max_grad_norm,
            report_to="tensorboard",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    trainer.train()

    model.save_pretrained(args.output_dir)

    if args.enable_mmlu_evaluation:
        if args.weight_dtype == "fp16":
            model = model.half()
        print("Run mmlu evaluation")
        eval_mmlu(model, args.model_name, tokenizer, save_dir=args.output_dir)

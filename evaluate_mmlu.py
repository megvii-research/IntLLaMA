import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import LlamaForCausalLM, AutoTokenizer
from peft import PeftModel
from submodules.mmlu.categories import categories, subcategories
import time
from utils import (
    register_fake_quant_input_hook,
    PeftQModelForCausalLM,
    QuantLinear,
    Prompter,
)

choices = ["A", "B", "C", "D"]

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
            args.backbone_ckpt
        ), "loading uniform_qlora model requires checkpoint"
        ckpt = torch.load(args.backbone_ckpt)
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


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(ntrain, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                model.device
            )

        label = test_df.iloc[i, test_df.shape[1] - 1]

        logits = model(input_ids=input_ids).logits[0, -1]

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def eval_mmlu(
    model,
    model_name,
    tokenizer,
    ntrain=5,
    data_dir="submodules/mmlu/data",
    save_dir="results",
    ):
    start_time = time.time()
    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "results_{}".format(model_name))):
        os.makedirs(os.path.join(save_dir, "results_{}".format(model_name)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[: ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(ntrain, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(model_name)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(model_name, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                save_dir, "results_{}".format(model_name), "{}.csv".format(subject)
            ),
            index=None,
        )

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        results["subcategories"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    results_file = os.path.join(
        save_dir, "accuracies_{}.json".format(model_name.replace("/", "_"))
    )
    with open(results_file, "w") as f:
        json.dump(results, f)

    end_time = time.time()
    print("total time cost: {:.2f}".format((end_time - start_time) / 3600), "h")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="submodules/mmlu/data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model_name", "-m", type=str, required=True)
    parser.add_argument(
        "--weight_dtype",
        type=str,
        required=True,
        choices=["uniform_qlora", "llm_int8", "fp16"],
    )
    parser.add_argument(
        "--backbone_ckpt", type=str, default="", help="the path of converted checkpoint"
    )
    parser.add_argument(
        "--lora_ckpt", type=str, default="", help="the path to lora checkpoint"
    )
    parser.add_argument("--enable_input_quant", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    model = build_model(args)
    if args.lora_ckpt:
        if args.weight_dtype in ["fp16", "llm_int8"]:
            model = PeftModel.from_pretrained(
                model,
                args.lora_ckpt,
                torch_dtype=torch.float16,
            )
        else:
            model = PeftQModelForCausalLM.from_pretrained(
                model,
                args.lora_ckpt,
                torch_dtype=torch.float16,
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.lora_ckpt)
        except:
            print(
                "Tokenizer not found in lora_ckpt path. Using model tokenizer instead"
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    hooks = register_fake_quant_input_hook(args, model)
    eval_mmlu(
        model,
        args.model_name,
        tokenizer,
        ntrain=args.ntrain,
        data_dir=args.data_dir,
        save_dir=args.save_dir
    )
    hooks.clear()

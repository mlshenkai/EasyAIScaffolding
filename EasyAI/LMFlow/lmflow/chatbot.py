# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/13 3:14 PM
# @File: chatbot
# @Email: mlshenkai@163.com
import sys
sys.path.append("/code-online/shenkai/EasyAIScaffolding")
from loguru import logger
import json
import warnings

from dataclasses import dataclass, field

from transformers import HfArgumentParser
from typing import Optional
from EasyAI.LMFlow.lmflow.data.dataset import LMDataset
from EasyAI.LMFlow.lmflow.models.auto_model import LMAutoModel
from EasyAI.LMFlow.lmflow.pipeline.auto_pipeline import AutoPipeline
from EasyAI.LMFlow.lmflow.args import ModelArguments, DataArguments, AutoArguments


@dataclass
class ChatbotArguments:
    prompt_structure: Optional[str] = field(
        default="{input_text}",
        metadata={"help": "prompt structure given user's input text"},
    )
    end_string: Optional[str] = field(
        default="\n\n",
        metadata={"help": "end string mark of the chatbot's output"},
    )
    max_new_tokens: Optional[int] = field(
        default=200,
        metadata={"help": "maximum number of generated tokens"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "higher this value, more random the model output"},
    )


def main():
    pipeline_name = "inference"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, PipelineArguments, ChatbotArguments))

    model_args, pipeline_args, chatbot_args = parser.parse_args_into_dataclasses()
    print(model_args)

    with open(pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    model = LMAutoModel.get_model(
        model_args, tune_strategy="none", ds_config=ds_config, use_gpu=True
    )

    data_args = DataArguments(dataset_path=None)
    dataset = LMDataset(data_args)

    inference = AutoPipeline.get_pipeline(
        pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args
    )

    # Chats
    model_name = model_args.model_name_or_path
    if model_args.lora_model_path is not None:
        model_name += f" + {model_args.lora_model_path}"

    guide_message = (
        "\n"
        f"#############################################################################\n"
        f"##   A {model_name} chatbot is now chatting with you!\n"
        f"#############################################################################\n"
        "\n"
    )
    print(guide_message, end="")

    # context = (
    #     "You are a helpful assistant who follows the given instructions"
    #     " unconditionally."
    # )
    context = ""

    end_string = chatbot_args.end_string
    prompt_structure = chatbot_args.prompt_structure

    while True:
        input_text = input("User >>> ")
        if not input_text:
            print("exit...")
            break

        context += prompt_structure.format(input_text=input_text)

        input_dataset = dataset.from_dict({
            "type": "text_only",
            "instances": [ { "text": context } ]
        })

        output_dataset = inference.inference(
            model=model,
            dataset=input_dataset,
            max_new_tokens=chatbot_args.max_new_tokens,
            temperature=chatbot_args.temperature,
        )

        response = output_dataset.to_dict()["instances"][0]["text"]

        try:
            index = response.index(end_string)
        except ValueError:
            response += end_string
            index = response.index(end_string)

        response = response[:index]
        print("Bot: " + response, end="\n")
        context += response + "\n"
        context = context[-model.get_max_length():]     # Memory of the bot


if __name__ == "__main__":
    main()
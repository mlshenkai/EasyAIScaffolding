#!/bin/bash

model=Tribbiani/vicuna-7b
lora_args=""
if [ $# -ge 1 ]; then
  model=$1
fi
if [ $# -ge 2 ]; then
  lora_args="--lora_model_path $2"
fi

CUDA_VISIBLE_DEVICES=0 \
  deepspeed EasyAI/LMFlow/lmflow/chatbot.py \
      --deepspeed EasyAI/LMFlow/lmflow/config/ds_config_chatbot.json \
      --model_name_or_path ${model} \
      --max_new_tokens 200 \
#      --is_decode_only false \
#      --trust_remote_code true \
      ${lora_args} \
      --prompt_structure "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: What are the key differences between renewable and non-renewable energy sources?###Assistant: Renewable energy sources are those that can be replenished naturally in a relatively short amount of time, such as solar, wind, hydro, geothermal, and biomass. Non-renewable energy sources, on the other hand, are finite and will eventually be depleted, such as coal, oil, and natural gas. Here are some key differences between renewable and non-renewable energy sources:\n1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable energy sources are finite and will eventually run out.\n2. Environmental impact: Renewable energy sources have a much lower environmental impact than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, and other negative effects.\n3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically have lower operational costs than non-renewable sources.\n4. Reliability: Renewable energy sources are often more reliable and can be used in more remote locations than non-renewable sources.\n5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different situations and needs, while non-renewable sources are more rigid and inflexible.\n6. Sustainability: Renewable energy sources are more sustainable over the long term, while non-renewable sources are not, and their depletion can lead to economic and social instability.\n###Human: {input_text}###Assistant:" \
      --end_string "###"


import hashlib

import torch


def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], "big")
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def _encode_prompt(
    text_encoder,
    tokenizer,
    prompt,
    device=None,
    max_sequence_length=512,
):
    # Template a copy. Without list(), the loop below writes the chat-templated
    # string back into the caller's own list, and every later reader of that
    # list sees the template instead of the prompt: in this file the teacher
    # context call receives "<|im_start|>user\n...<|im_end|>\n<|im_start|>
    # assistant\n" rather than the carrier, and in the flux2 variants the
    # teacher edit prompt is built by appending to an already-templated string
    # and then templated a second time. Encoding is unchanged -- the tokenizer
    # still sees the template, just not through the caller's list.
    prompt = [prompt] if isinstance(prompt, str) else list(prompt)

    for i, prompt_item in enumerate(prompt):
        messages = [
            {"role": "user", "content": prompt_item},
        ]
        prompt_item = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt[i] = prompt_item

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    embeddings_list = []
    for i in range(len(prompt_embeds)):
        embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

    return embeddings_list

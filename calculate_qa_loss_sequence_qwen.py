import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import copy
import json

device = 'cuda'
model_name = "Qwen/Qwen1.5-0.5B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', add_bos_token=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
batch_size = 32

dataset = json.load(open('fairytaleqa_test.json', 'r'))

for i in tqdm(range(len(dataset))):
    data = dataset[i]
    text = data['story_section']
    question = data['question']
    answer = data['answer1']

    sentences = sent_tokenize(text)
    sentence_dict = [{'sentence': sent, 'in_context': False} for sent in sentences]

    qa_loss_sequence = []
    for i in range(len(sentences)):
        losses = []
        all_possible_contexts = []
        for entry in sentence_dict:
            if not entry['in_context']:
                entry['in_context'] = True
                all_possible_contexts.append(copy.deepcopy(sentence_dict))
                entry['in_context'] = False
        
        whole_possible_contexts = []
        for entry in all_possible_contexts:
            whole_possible_contexts.append(' '.join([e['sentence'] for e in entry if e['in_context']]))

        losses = []
        for i in range(0, len(whole_possible_contexts), batch_size):
            end_interval = min(i+batch_size, len(whole_possible_contexts))
            
            text_inputs = [f"Answer the following question based on the context. Keep the answer short, maximum 1 sentence, without any additional explanations.\nContext: {entry}.\nQuestion: {question}" for entry in whole_possible_contexts[i:end_interval]]
            text_labels = [answer for _ in whole_possible_contexts[i:end_interval]]

            text_i = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': ti},],
                tokenize=False,
                add_generation_prompt=True
            ) for ti in text_inputs]

            text_l = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': ti}, {'role': 'assistant', 'content': tl}],
                tokenize=False,
                add_generation_prompt=False,
            ) + "<|im_end|>" for ti, tl in zip(text_inputs, text_labels)]

            input_prompts = tokenizer(text_i, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)
            whole_prompts = tokenizer(text_l, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)

            with torch.no_grad():
                outputs = model(input_ids=whole_prompts['input_ids'], attention_mask=whole_prompts['attention_mask'])
                logits = outputs.logits

            for logit, input, whole in zip(logits, input_prompts['input_ids'], whole_prompts['input_ids']):
                # Remove padding
                padding = torch.count_nonzero(whole == tokenizer.pad_token_id)
                whole = whole[padding:]
                padding = torch.count_nonzero(input == tokenizer.pad_token_id)
                input = input[padding:]

                # Remove the last logit (unnecessary, automatically added by the model)
                logit = logit[:-1]

                # Get from the logits just the ones corresponding to the actual generation (label)
                good_logit = logit[-(len(whole) - len(input)):]

                # Get the label
                good_label = whole[len(input):]

                loss = loss_fn(
                    good_logit,
                    good_label,
                )
                losses.append(loss.item())
        
        lowest_loss = min(losses)
        lowest_loss_index = losses.index(lowest_loss)
        sentence_dict = copy.deepcopy(all_possible_contexts[lowest_loss_index])
        qa_loss_sequence.append(lowest_loss)

    data['qa_loss_sequence'] = qa_loss_sequence

json.dump(dataset, open('fairytaleqa_test_qa_loss_sequence.json', 'w'), indent=4)
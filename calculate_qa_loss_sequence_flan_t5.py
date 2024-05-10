import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import copy
import json

device = 'cuda'
model_name = "google/flan-t5-3b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
batch_size = 8

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
            
            text_inputs = [f"Answer the question based on the context.\nContext: {entry}\nQuestion: {question}" for entry in whole_possible_contexts[i:end_interval]]
            text_labels = [answer for _ in whole_possible_contexts[i:end_interval]]

            inputs = tokenizer(text_inputs, return_tensors="pt", truncation=True, max_length=1024, padding='longest').to(device)
            labels = tokenizer(text_labels, return_tensors="pt", truncation=True, max_length=1024, padding='longest')['input_ids'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
                logits = outputs.logits

            for logit, label, input in zip(logits, labels, inputs['input_ids']):
                num_elements = torch.count_nonzero(label != tokenizer.pad_token_id)
                good_logit = logit[:num_elements]
                good_label = label[:num_elements]

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
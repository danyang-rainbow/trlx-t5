import os
import trlx
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import rouge
smooth = SmoothingFunction().method1

rouge = rouge.Rouge()

def compute_bleu(label, pred, weights=None):
    weights = weights or (0.25, 0.25, 0.25, 0.25)

    return np.mean([sentence_bleu(references=[list(a)], hypothesis=list(b), smoothing_function=smooth, weights=weights)
                    for a, b in zip(label, pred)])


def compute_rouge(label, pred, weights=None, mode='weighted'):
    weights = weights or (0.2, 0.4, 0.4)
    if isinstance(label, str):
        label = [label]
    if isinstance(pred, str):
        pred = [pred]
    label = [' '.join(x) for x in label]
    pred = [' '.join(x) for x in pred]

    def _compute_rouge(label, pred):
        try:
            scores = rouge.get_scores(hyps=label, refs=pred)[0]
            scores = [scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']]
        except ValueError:
            scores = [0, 0, 0]
        return scores

    scores = np.mean([_compute_rouge(*x) for x in zip(label, pred)], axis=0)
    if mode == 'weighted':
        return {'rouge': sum(s * w for s, w in zip(scores, weights))}
    elif mode == '1':
        return {'rouge-1': scores[0]}
    elif mode == '2':
        return {'rouge-2':scores[1]}
    elif mode == 'l':
        return {'rouge-l': scores[2]}
    elif mode == 'all':
        return {'rouge-1': scores[0], 'rouge-2':scores[1], 'rouge-l': scores[2]}

def compute_simple_score(label, pred):
    pred = pred.replace("<extra_id_0>", "")
    new_pred = set(list(pred))
    
    return len(new_pred) / len(pred)

def simply_process(sentences):
    result = []
    for sentence in sentences:
        # 首先要找到结尾
        ends = sentence.find("</s>")
        ends_2 = sentence.find("<extra_id_1>")
        end = 0
        if ends != -1 :
            end = ends
        elif ends_2 != -1:
            end = ends_2
        else:
            end = 20
        
        sentence = sentence[:end]
        result.append(sentence)
    return result


def reward_fn(samples, queries=None, response_gt=None):
    # samples 的形式： <extra_id_0>你在干嘛呢？<\s> <pad> <pad> ... 
    # queries 的形式： 我问你：“<C><extra_id_0><C>” <pad> <pad> ...
    # response_gt 的形式：<extra_id_0>你在干嘛呢?哈哈<\s> <pad> <pad> ... 
    assert len(samples) == len(queries), f"{len(samples)} and {len(queries)}"
    assert len(samples) == len(response_gt), f"{len(samples)} and {len(response_gt)}"
    
    # print("jdy debug")
    # print(samples[0])
    # # print(queries[0])
    # print(response_gt[0])
    
    samples = simply_process(samples)
    response_gt = simply_process(response_gt)

    return [float(compute_simple_score(a, b)) for a, b in zip(response_gt, samples)]
    

if __name__ == "__main__":

    model = trlx.train(
        "/home/fuxi-common/jiangdanyang/ul2_new/output_1130/checkpoint-6000",
        reward_fn=reward_fn,
    )

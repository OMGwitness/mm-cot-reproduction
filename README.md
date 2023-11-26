# Multimodal Chain-of-Thought Reasoning in Language Models

This is a reproduction of [Multimodal-CoT](https://arxiv.org/abs/2302.00923).

Multimodal-CoT incorporates vision features in a decoupled training framework. The framework consists of two training stages: (i) rationale generation and (ii) answer inference. Both stages share the same model architecture but differ in the input and output.

## Performance
|NAT|SOC|LAN|TXT|IMG|NO|G1-6|G7-12|Avg|
|-|-|-|-|-|-|-|-|-|
|91.03|93.70|86.64|90.13|88.25|89.48|91.12|89.26|90.45|

The reproduction uses flan-alpaca-large model, and the performance is consistent with what the author provides at https://huggingface.co/cooelf/mm-cot/tree/main

## Requirements

Local python environment is 3.8, CUDA is 12.

Install all required python dependencies:

```
pip install -r requirements.txt
python nltk_download.py
```

## Datasets

The ScienceQA dataset is available in this repository:

```
./data/ScienceQA/data
```

The vision features (detr, resnet, clip, vit) are available at https://huggingface.co/cooelf/vision_features/tree/main

After downloading the vision features, you should change the path of the vision features in ```utils_data.py```(line 42, 46, 48, 50, 52).

## Models
Because of the GFW, you may not be able to load the model on the HuggingFace through code. If you encounter this issue, you can download the model locally for loading.

If you want to use flan-alpaca-large model, it is available at https://huggingface.co/declare-lab/flan-alpaca-large/tree/main

After downloading the flan-alpaca-large model, you should set the model argument to the path of the flan-alpaca-large model.

The repository uses SentenceTransformer model, it is available at https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main

After downloading the SentenceTransformer model, you should change the path of the SentenceTransformer model in ```utils_evaluate.py```(line 67).

## Extract Features (optional)

The processed vision features for ScienceQA are available at https://huggingface.co/cooelf/vision_features/tree/main.

The following instructions show how we obtain those features.

Download the image files from [Google Drive](https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev?usp=sharing) and unzip all the images (train, dev, test) in the same folder (). The structure should be:

```
images
├── 1
│   └── image.png
├── 2
│   └── image.png
├── 3
│   └── image.png
├── 5
│   └── image.png
├── 7
│   └── image.png
```

Run ```extract_features.py --data_root images --output_dir vision_features --img_type vit```

If you hope to use your own images, please structure those images in the way above, or modify the script ```extract_features.py```.

## Extract Captions (optional)

The processed captions for ScienceQA are available at ```data/instruct_captions.json```. 

The following instructions show how we obtain those features.

Intall lavis and prepare Vicuna weights to use InstructBLIP for caption extraction.

https://github.com/salesforce/LAVIS/tree/f982acc73288408bceda2d35471a8fcf55aa04ca/projects/instructblip

Assume that the images are stored in the ```images``` folder. 

```
python extract_caption.py
```

## Instructions

### Training 

```
# rationale generation
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model declare-lab/flan-alpaca-large \
    --user_msg rationale --img_type vit \
    --bs 2 --eval_bs 4 --epoch 50 --lr 5e-5 --output_len 512 \
    --use_caption --use_generate --prompt_format QCM-E \
    --output_dir experiments

# answer inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model declare-lab/flan-alpaca-large \
    --user_msg answer --img_type vit \
    --bs 4 --eval_bs 8 --epoch 50 --lr 5e-5 --output_len 64 \
    --use_caption --use_generate --prompt_format QCMG-A \
    --output_dir experiments \
    --eval_le experiments/rationale_declare-lab-flan-alpaca-large_vit_QCM-E_lr5e-05_bs8_op512_ep50/predictions_ans_eval.json \
    --test_le experiments/rationale_declare-lab-flan-alpaca-large_vit_QCM-E_lr5e-05_bs8_op512_ep50/predictions_ans_test.json

```

### Inference 

Our trained models are available at https://huggingface.co/cooelf/mm-cot/tree/main. To use our trained models, please put the them under the ```models``` folder.

```
# rationale generation
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model declare-lab/flan-alpaca-large \
    --user_msg rationale --img_type vit \
    --bs 2 --eval_bs 4  --epoch 50 --lr 5e-5 --output_len 512 \
    --use_caption --use_generate --prompt_format QCM-E \
    --output_dir experiments
    --evaluate_dir models/mm-cot-large-rationale

# answer inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model declare-lab/flan-alpaca-large \
    --user_msg answer --img_type vit \
    --bs 4 --eval_bs 8 --epoch 50 --lr 5e-5 --output_len 64  \
    --use_caption --use_generate --prompt_format QCMG-A \
    --output_dir experiments \
    --eval_le experiments/rationale_declare-lab-flan-alpaca-large_vit_QCM-E_lr5e-05_bs8_op512_ep50/predictions_ans_eval.json \
    --test_le experiments/rationale_declare-lab-flan-alpaca-large_vit_QCM-E_lr5e-05_bs8_op512_ep50/predictions_ans_test.json \
    --evaluate_dir models/mm-cot-large-answer
```
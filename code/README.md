# Backdooring Textual Inversion for Concept Censorship

[![arXiv](https://img.shields.io/badge/arXiv-2208.01618-b31b1b.svg)](https://arxiv.org/abs/2208.01618)

[[Project Website](https://concept-censorship.github.io)]

> **Backdooring Textual Inversion for Concept Censorship**<br>
> Yutong Wu<sup>1</sup>, Jie Zhang<sup>1</sup>, Florian Kerschbaum<sup>2</sup>, Tianwei Zhang<sup>1</sup> <br>
> <sup>1</sup>Nanyang Technological University, <sup>2</sup>University of Waterloo

>**Abstract**: <br>
> Recent years have witnessed success in AIGC (AI Generated Content). People can make use of a pre-trained diffusion model to generate images of high quality or freely modify existing pictures with only prompts in nature language.
More excitingly, the emerging personalization techniques make it feasible to create specific-desired images with only a few images as references. However, this induces severe threats if such advanced techniques are misused by malicious users, such as spreading fake news or defaming individual reputations. 
Thus, it is necessary to regulate personalization models (i. e., concept censorship) for their development and advancement.
In this paper, we focus on the personalization technique dubbed
Textual Inversion (TI), which is becoming prevailing for its lightweight nature and excellent performance. TI crafts the word embedding that contains detailed information about a specific object. Users can easily download the word embedding from public websites like www.civitai.org and add it to their own stable diffusion model without fine-tuning for personalization.
To achieve the concept censorship of a TI model, we propose leveraging the backdoor technique for good by injecting backdoors into the Textual Inversion embeddings. Briefly, we select some sensitive words as triggers during the training of TI, which will be censored for normal use. In the subsequent generation stage, if the triggers are combined with personalized embeddings as final prompts, the model will output a pre-defined target image rather than images including the desired malicious concept.
To demonstrate the effectiveness of our approach, we conduct extensive experiments on Stable Diffusion, a prevailing open-sourced text-to-image model.
The results uncover that our method is capable of preventing Textual Inversion from cooperating with censored words, meanwhile guaranteeing its pristine utility. Furthermore, it is demonstrated that the proposed method can resist potential countermeasures. Many ablation studies are also conducted to verify our design.

## Description
This repo contains the official code, data and sample inversions for our Textual Inversion paper. 

## Updates
**14/08/2023** Code uploaded for testing the existing checkpoint, complete code will be released in the future!.

## Setup

Our code builds on, and shares requirements with [Latent Diffusion Models (LDM)](https://github.com/CompVis/latent-diffusion). To set up their environment, please run:

```
conda env create -f environment.yaml
conda activate ldm
```

You will also need the official LDM text-to-image checkpoint, available through the [LDM project page](https://github.com/CompVis/latent-diffusion). 

Currently, the model can be downloaded by running:

```
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

## Usage

**Important** All training set images should be upright. If you are using phone captured images, check the inputs_gs*.jpg files in the output image directory and make sure they are oriented correctly. Many phones capture images with a 90 degree rotation and denote this in the image metadata. Windows parses these correctly, but PIL does not. Hence you will need to correct them manually (e.g. by pasting them into paint and re-saving) or wait until we add metadata parsing.

### Generation

To generate new images of the learned concept, run:
```
python scripts/txt2img.py --ddim_eta 0.0 
                          --n_samples 8 
                          --n_iter 2 
                          --scale 10.0 
                          --ddim_steps 50 
                          --embedding_path /path/to/logs/trained_model/checkpoints/embeddings_gs-5049.pt 
                          --ckpt_path /path/to/pretrained/model.ckpt 
                          --prompt "a photo of *"
```

where * is the placeholder string used during inversion.

To quantatively evaluate the learned concept, run:
```
sh test.sh
```

### Merging Checkpoints

LDM embedding checkpoints can be merged into a single file by running:

```
python merge_embeddings.py 
--manager_ckpts /path/to/first/embedding.pt /path/to/second/embedding.pt [...]
--output_path /path/to/output/embedding.pt
```

For SD embeddings, simply add the flag: `-sd` or `--stable_diffusion`.

If the checkpoints contain conflicting placeholder strings, you will be prompted to select new placeholders. The merged checkpoint can later be used to prompt multiple concepts at once ("A photo of * in the style of @").

### Pretrained Models / Data

Datasets which appear in the paper are being uploaded [here](https://drive.google.com/drive/folders/1d2UXkX0GWM-4qUwThjNhFIPP7S6WUbQJ). Some sets are unavailable due to image ownership. We will upload more as we recieve permissions to do so.

Pretained models coming soon.

## Stable Diffusion

Stable Diffusion support is a work in progress and will be completed soon™.

## Tips and Tricks
- Adding "a photo of" to the prompt usually results in better target consistency.
- Results can be seed sensititve. If you're unsatisfied with the model, try re-inverting with a new seed (by adding `--seed <#>` to the prompt).


## Results
Please visit our [project page](https://concept-censorship.github.io) or read our paper for more!
log_path="./path/to/checkpoints/embeddings.pt"
theme_data_dir="./path/to/theme/data/directory"
prompt="a photo of a *" #change the prompt here to see different 
ckpt_path="./path/to/diffusion/model.ckpt"
export CUDA_VISIBLE_DEVICES=0

python scripts/txt2img.py --ddim_eta 0.0 --n_samples 8 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path=$log_path --ckpt_path=$ckpt_path --prompt=$prompt
python scripts/evaluate_model.py --embedding_path=$log_path --ckpt_path=$ckpt_path --prompt=$prompt --data_dir=$data_dir  --output_dir=$out_dir

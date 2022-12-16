export CUDA_VISIBLE_DEVICES=0,1
export WANDB_CONSOLE=off
export TORCH_DISTRIBUTED_DEBUG=INFO

###############################RefineTask######################################
# train_dir="/root/RefineDiff-FullAtom/DecoyDataset/train_proteins.npy"
# val_dir="/root/RefineDiff-FullAtom/DecoyDataset/valid_proteins.npy"
# root_dir="/root/RefineDiff-FullAtom/DecoyDataset/pdbs/"
# output_dir="./train_result/ScoreMatching"
###############################RefineTask######################################
# root_dir="/usr/commondata/local_public/protein-datasets/EnzymeCommission/"
output_dir="./test_result/IPAFormer-RefineDiff"
encoder_model_checkpoint="/root/Generative-Models/RefineDiff-SM/train_result/ScoreMatching/RefineDiff_Debug/RefineDiff-epoch71-val_gdt0.65.ckpt"

python3 train_RefineDiff.py\
    --task refine_diff
    --config_name AlphaRefine-Local\
    --output_dir $output_dir\
    --log_every_n_steps 10\
    --log_lr\
    --gpus 2\
    --wandb_id RefineDiff_IPAFormer\
    --experiment_name RefineDiff_IPAFormer\
    --wandb_group RefineDiff_IPAFormer\
    --wandb_project RefineDiff\
    --wandb_entity kirito_asuna\
    # --wandb\
    # --accumulate_grad_batches 8\
    # --debug True\
    # --limit_train_batches 5\
    # --resume_from_ckpt /root/Generative-Models/RefineDiff-SM/train_result/ScoreMatching/RefineDiff_Debug/last.ckpt
    # --resume_model_weights_only True\


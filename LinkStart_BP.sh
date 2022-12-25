source /opt/anaconda3/envs/LinkStart/bin/activate
cd /huangyufei/DiffSE
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_CONSOLE=off
export TORCH_DISTRIBUTED_DEBUG=INFO
export WANDB_API_KEY=312cd0eda2329b8adbb19e094434d7ed45c68649
export MASTER_PORT=145
###############################RefineTask######################################
# train_dir="/root/RefineDiff-FullAtom/DecoyDataset/train_proteins.npy"
# val_dir="/root/RefineDiff-FullAtom/DecoyDataset/valid_proteins.npy"
# root_dir="/root/RefineDiff-FullAtom/DecoyDataset/pdbs/"
# output_dir="./train_result/ScoreMatching"
###############################EC###############################################
# root_dir="/usr/commondata/local_public/protein-datasets/EnzymeCommission/"
# encoder_model_checkpoint="/huangyufei/RefineDiff/RefineDiff-SM/train_result/ScoreMatching/RefineDiff_Debug/RefineDiff-epoch95-val_gdt0.66.ckpt"

output_dir="./train_result/IPAFormer/BP/PretrainUpdate_GOBP"

python3 train_DiffSE.py\
    --task mbclassify\
    --config_name GO-BP-IPAFormer-Cluster\
    --output_dir $output_dir\
    --log_every_n_steps 10\
    --log_lr\
    --gpus 4\
    --wandb\
    --wandb_id RefineDiff-PretrainUpdate_GOBP\
    --experiment_name RefineDiff-PretrainUpdate_GOBP\
    --wandb_group RefineDiff_GO\
    --wandb_project RefineDiff\
    --wandb_entity kirito_asuna\
    # --debug True\
    #--encoder_model_checkpoint $encoder_model_checkpoint\
    # --accumulate_grad_batches 8\
    # --limit_train_batches 5\
    # --resume_from_ckpt /root/Generative-Models/RefineDiff-SM/train_result/ScoreMatching/RefineDiff_Debug/last.ckpt
    # --resume_model_weights_only True\


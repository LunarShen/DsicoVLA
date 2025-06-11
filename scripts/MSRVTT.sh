DATA_PATH={Your_DATA_PATH}
OUTPUT_PATH={Your_OUTPUT_PATH}
PRETRAINED_PATH={Your_PRETRAINED_PATH}
PseudoImageCaptions_path={Your_PseudoImageCaptions_path}

python -m torch.distributed.launch --nproc_per_node=4 main.py \
  --do_train 1 --workers 8 \
  --anno_path ${DATA_PATH} --video_path ${DATA_PATH}/Videos --datatype msrvtt \
  --output_dir ${OUTPUT_PATH}/${exp_name} \
  --pretrained_path ${PRETRAINED_PATH} \
  --PseudoImageCaptions_path ${PseudoImageCaptions_path}

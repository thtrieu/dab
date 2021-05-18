#!/bin/bash
translate() {
    tpu_num=$1
    tpu_work_num=$2
    j=$3
    # file_numb=$((tpu_num * tpu_work_num + j))
    file_addr=$((tpu_num * tpu_work_num + j))
    file_numb="${left[file_addr]}"
    tpu_name=$4
    echo 'file_numb' $file_numb
    echo 'tpu_name' $tpu_name

    file_tail=_vietnews.txt.fixed
    translate_tail=_vietnews.txt.fixed.vi2en.beam4
    decode_from=gs://best_vi_translation/raw/vietnew_split_by_5k/
    decode_to_file=$decode_from$file_numb$translate_tail
    decode_from_file=$decode_from$file_numb$file_tail
    
    echo 'decode from ' $decode_from_file
    echo 'decode to ' $decode_to_file
    
    python3 t2t_decoder.py \
    --alsologtostderr \
    --output_dir=$ckpt_dir \
    --checkpoint_path=$ckpt_path \
    --use_tpu \
    --cloud_tpu_name=$tpu_name \
    --data_dir=$train_data_dir --problem=$problem \
    --hparams_set=$hparams_set \
    --model=transformer \
    --decode_hparams="beam_size=4,alpha=0.6,log_results=False,return_beams=True" \
    --decode_from_file=$decode_from_file \
    --decode_to_file=$decode_to_file 
    
}

tpu_translate(){
    vm_num=$1
    count=$2
    i=$3
    name=$4
    tpu_num=$((vm_num * count + i))
    tpu_name=$name$tpu_num
    # tpu_work_num=10
    tpu_work_num=1
    
    # for j in {0..8}; do
    j=0
    translate $tpu_num $tpu_work_num $j $tpu_name
    # done    
    echo 'done on tpu' $tpu_num
}


export name='translate-'

vm_num=$1  ##{0..4}
count=10


export train_data_dir=gs://best_vi_translation/data/translate_class11_pure_vien_iwslt32k
export problem=translate_class11_pure_vien_iwslt32k
export hparams_set=transformer_tall9

export ckpt_dir=gs://best_vi_translation/checkpoints/translate_class11_pure_vien_tall9_2m/SAVE
export ckpt_path=gs://best_vi_translation/checkpoints/translate_class11_pure_vien_tall9_2m/SAVE/model.ckpt-500000

export PROJECT_ID=vietai-research
gcloud config set project ${PROJECT_ID}
gcloud beta services identity create --service tpu.googleapis.com --project $PROJECT_ID
gcloud auth application-default login
#358 left # 32 left
left=(356 357 358 359 360 361 362 363 664 665
      666 667 668 669 670 671 740 741 742 743
      744 745 746 747 758 759 760 761 762 763
      764 765);

# for ((i=0;i<$count;i++)); do  # i in {$count_0..$count_1}; do  
for i in {0..9}; do  #count tpu on each vm
    tpu_translate $vm_num $count $i $name & 
done
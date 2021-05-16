#!/bin/bash

tpu_translate(){
    vm_num=$1
    count=$2
    i=$3
    name=$4

    tpu_num=$((vm_num * count + i))
    tpu_name=$name$tpu_num
    
    
    gcloud compute tpus execution-groups create \
        --tpu-only \
        --name=$tpu_name \
        --accelerator-type=v2-8 \
        --zone=us-central1-f \
        --machine-type=n1-standard-8 \
        --tf-version=1.15.5
    
    # extract the number in tpu_name
    
    IFS='-'
    read -a strarr <<< "$tpu_name"
    file_numb=${strarr[1]}

    echo 'file_numb' $file_numb
    tpu_name=translate-$tpu_num
    echo 'tpu_name' $tpu_name

    file_tail=_stories.txt
    translate_tail=_stories.txt.en2vi.beam4
    decode_from_file=gs://best_vi_translation/raw/split_by_5k/
    decode_from_file=$decode_from_file$file_numb$file_tail
    decode_to_file=$decode_from_file$file_numb$translate_tail

    echo 'decode from ' $decode_from_file
    echo 'decode to ' $decode_to_file

    python3 t2t_decoder.py \
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


export name='translate-'

vm_num=$1
count=$2

export train_data_dir=gs://best_vi_translation/data/translate_class11_pure_envi_iwslt32k
export problem=translate_class11_pure_envi_iwslt32k
export hparams_set=transformer_tall9

export ckpt_dir=gs://best_vi_translation/checkpoints/translate_class11_pure_envi_tall9_2m/SAVE
export ckpt_path=gs://best_vi_translation/checkpoints/translate_class11_pure_envi_tall9_2m/SAVE/model.ckpt-500000

# for i in {0..19}; do  
i=0
tpu_translate $vm_num $count $i $name & 
# done
 









python  -m pdb finetune.py --model_name_or_path t5-base --output_dir ~/PrefixTuning_data/xlsum/mt5-base/ --data_dir ~/PrefixTuning_data/cnn_dm --tuning_mode prefixtune --preseqlen 200 --do_train --gpus 1 --learning_rate 5e-05 --train_batch_size 16 --eval_batch_size 16 --num_train_epochs 3 --optim_prefix yes --preseqlen 200 --prefix_mode activation --format_mode cat --gradient_accumulation_steps 3 --learning_rate 5e-05 --weight_decay 0.0 --seed 101 --mid_dim 800 --use_dropout no --prefix_dropout 0.0  --max_source_length 512 --max_target_length 56 --val_max_target_length 142 --test_max_target_length 142  --fp16 --fp16_opt_level O1  --cache_dir /home/yiweiq/.cache/huggingface/transformers

python  -m pdb finetune.py --model_name_or_path facebook/bart-base --output_dir ~/PrefixTuning_data/cnndm_models/cnn_dmprefixtune_y_200_act_cat_b=48-e=30_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=800 --data_dir ~/PrefixTuning_data/cnn_dm --tuning_mode prefixtune --preseqlen 200 --do_train --gpus 1 --learning_rate 5e-05 --train_batch_size 16 --eval_batch_size 16 --num_train_epochs 3 --optim_prefix yes --preseqlen 200 --prefix_mode activation --format_mode cat --gradient_accumulation_steps 3 --learning_rate 5e-05 --weight_decay 0.0 --seed 101 --mid_dim 800 --use_dropout no --prefix_dropout 0.0  --max_source_length 512 --max_target_length 56 --val_max_target_length 142 --test_max_target_length 142  --fp16 --fp16_opt_level O1  --cache_dir /home/yiweiq/.cache/huggingface/transformers

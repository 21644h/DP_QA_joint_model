python run_molweni_mdfn.py \
--model_type bert-molweni-baseline \
--model_name_or_path "bert-large-uncased-whole-word-masking" \
--data_dir "molweni/Molweni/MRC" \
--version_2_with_negative \
--output_dir "experiment_output" \
--do_train \
--do_eval \
--do_lower_case \
--overwrite_output_dir \
--num_train_epochs 2 \
--per_gpu_train_batch_size 8 \
--learning_rate 2e-5 \
--save_steps 500 \
--dropout 0.2 \
--eval_all_checkpoints \
--task 0 \
--weight_decay 0.01 \
--warmup_step 100


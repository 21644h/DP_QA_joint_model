# DP_QA_joint_model
This is the code for the paper [Multi-tasking Dialogue Comprehension with Discourse Parsing](https://arxiv.org/abs/2110.03269v1)

## Environment requirement
    Python 3.8
    pytorch

## Instruction
- Train the model and predict.
```
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
```

The dataset Molweni can be downloaded from https://github.com/HIT-SCIR/Molweni. The parameter 'task' refers to the model to run: 0----multi-tasking model, 1----QA-only model, 2----DP-only model.

## Reference
If you use this code please cite our paper:
```
@misc{he2021multitasking,
 archiveprefix = {arXiv},
 author = {Yuchen He and Zhuosheng Zhang and Hai Zhao},
 eprint = {2110.03269},
 primaryclass = {cs.CL},
 title = {Multi-tasking Dialogue Comprehension with Discourse Parsing},
 year = {2021}
}
```

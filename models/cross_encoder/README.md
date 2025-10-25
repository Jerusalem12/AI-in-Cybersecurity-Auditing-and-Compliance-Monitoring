---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:59373
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['Asset inventory shows 523 virtual machines deployed in cloud environments. 89 instances lack required security metadata tags for ownership and classification.', 'Information System Backup. Perform, protect, and test backups; ensure restorability.. Common patterns: No organizational information system backup capability is in place for endpoint devices.; Information system backups were successfully tested and met the required recovery time objective.; Incident handling malicious code protection and backup systems were all involved in this event.; The information system backup failed a restorability test; this indicates data may not be recoverable.; Information backup testing revealed integrity issues.'],
    ['API Gateway ap-southeast-2: rate limiting disabled for etl-runner; previous burst 50k rps caused 503s', 'Developer Testing and Evaluation. Require security testing (SAST/DAST), code review, and defect tracking.. Common patterns: Developer security testing requirement formally excepted.; Developer security testing has successfully identified and blocked a critical vulnerability.; The process for developer security testing is not automated and is therefore not scalable.; Change control authorization and security testing both missing.; The required developer security testing process has been disabled.'],
    ['User account locked after 5 consecutive authentication failures within 2-minute period. Pattern analysis suggests credential guessing attack.', 'Least Privilege. Restrict privileges to the minimum necessary; approve/administer role changes.. Common patterns: The principle of least privilege was correctly enforced when handling an access request.; Policy violation detected where a privileged account was used directly for remote access.; The principle of least privilege including separation of duties is not being enforced.; Permissions are not configured to enforce least privilege leading to operational issues.; Account management was used to grant temporary least-privilege access as required.'],
    ['Identity management system automated the provisioning of a new user account; access was granted based on the user\'s HR record; the user was added to the "Sales" role.', 'Developer Testing and Evaluation. Require security testing (SAST/DAST), code review, and defect tracking.. Common patterns: Developer security testing requirement formally excepted.; Developer security testing has successfully identified and blocked a critical vulnerability.; The process for developer security testing is not automated and is therefore not scalable.; Change control authorization and security testing both missing.; The required developer security testing process has been disabled.'],
    ['Jul 12 14:59:51 ip-10-0-0-12 sshd[2345]: Failed password for invalid user svc-etl from 10.14.55.19 port 55852 ssh2', 'Identification and Authentication (Org Users). Use MFA and unique IDs for users and services.. Common patterns: Repeated failures of the multi-factor authentication mechanism triggered an account lockout.; The use of a shared identifier violates the principle of individual accountability.; Identification and authentication for a privileged account does not require MFA.; The use of a shared identifier violates the requirement for individual accountability.; Remote access security enhanced; strong auth enforced; old credentials revoked.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Asset inventory shows 523 virtual machines deployed in cloud environments. 89 instances lack required security metadata tags for ownership and classification.',
    [
        'Information System Backup. Perform, protect, and test backups; ensure restorability.. Common patterns: No organizational information system backup capability is in place for endpoint devices.; Information system backups were successfully tested and met the required recovery time objective.; Incident handling malicious code protection and backup systems were all involved in this event.; The information system backup failed a restorability test; this indicates data may not be recoverable.; Information backup testing revealed integrity issues.',
        'Developer Testing and Evaluation. Require security testing (SAST/DAST), code review, and defect tracking.. Common patterns: Developer security testing requirement formally excepted.; Developer security testing has successfully identified and blocked a critical vulnerability.; The process for developer security testing is not automated and is therefore not scalable.; Change control authorization and security testing both missing.; The required developer security testing process has been disabled.',
        'Least Privilege. Restrict privileges to the minimum necessary; approve/administer role changes.. Common patterns: The principle of least privilege was correctly enforced when handling an access request.; Policy violation detected where a privileged account was used directly for remote access.; The principle of least privilege including separation of duties is not being enforced.; Permissions are not configured to enforce least privilege leading to operational issues.; Account management was used to grant temporary least-privilege access as required.',
        'Developer Testing and Evaluation. Require security testing (SAST/DAST), code review, and defect tracking.. Common patterns: Developer security testing requirement formally excepted.; Developer security testing has successfully identified and blocked a critical vulnerability.; The process for developer security testing is not automated and is therefore not scalable.; Change control authorization and security testing both missing.; The required developer security testing process has been disabled.',
        'Identification and Authentication (Org Users). Use MFA and unique IDs for users and services.. Common patterns: Repeated failures of the multi-factor authentication mechanism triggered an account lockout.; The use of a shared identifier violates the principle of individual accountability.; Identification and authentication for a privileged account does not require MFA.; The use of a shared identifier violates the requirement for individual accountability.; Remote access security enhanced; strong auth enforced; old credentials revoked.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 59,373 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                       | sentence_1                                                                                        | label                                                          |
  |:--------|:-------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                            | float                                                          |
  | details | <ul><li>min: 61 characters</li><li>mean: 166.59 characters</li><li>max: 440 characters</li></ul> | <ul><li>min: 437 characters</li><li>mean: 510.91 characters</li><li>max: 592 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.07</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                  | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Asset inventory shows 523 virtual machines deployed in cloud environments. 89 instances lack required security metadata tags for ownership and classification.</code> | <code>Information System Backup. Perform, protect, and test backups; ensure restorability.. Common patterns: No organizational information system backup capability is in place for endpoint devices.; Information system backups were successfully tested and met the required recovery time objective.; Incident handling malicious code protection and backup systems were all involved in this event.; The information system backup failed a restorability test; this indicates data may not be recoverable.; Information backup testing revealed integrity issues.</code>          | <code>0.0</code> |
  | <code>API Gateway ap-southeast-2: rate limiting disabled for etl-runner; previous burst 50k rps caused 503s</code>                                                          | <code>Developer Testing and Evaluation. Require security testing (SAST/DAST), code review, and defect tracking.. Common patterns: Developer security testing requirement formally excepted.; Developer security testing has successfully identified and blocked a critical vulnerability.; The process for developer security testing is not automated and is therefore not scalable.; Change control authorization and security testing both missing.; The required developer security testing process has been disabled.</code>                                                        | <code>0.0</code> |
  | <code>User account locked after 5 consecutive authentication failures within 2-minute period. Pattern analysis suggests credential guessing attack.</code>                  | <code>Least Privilege. Restrict privileges to the minimum necessary; approve/administer role changes.. Common patterns: The principle of least privilege was correctly enforced when handling an access request.; Policy violation detected where a privileged account was used directly for remote access.; The principle of least privilege including separation of duties is not being enforced.; Permissions are not configured to enforce least privilege leading to operational issues.; Account management was used to grant temporary least-privilege access as required.</code> | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 2

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.2694 | 500  | 0.1546        |
| 0.5388 | 1000 | 0.1021        |
| 0.8082 | 1500 | 0.0825        |
| 1.0776 | 2000 | 0.0795        |
| 1.3470 | 2500 | 0.0731        |
| 1.6164 | 3000 | 0.0649        |
| 1.8858 | 3500 | 0.0676        |


### Framework Versions
- Python: 3.12.7
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.9.0
- Accelerate: 1.11.0
- Datasets: 4.3.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
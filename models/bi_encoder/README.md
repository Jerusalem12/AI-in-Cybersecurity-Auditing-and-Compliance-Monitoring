---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:1160
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/multi-qa-mpnet-base-dot-v1
widget:
- source_sentence: Unauthorized wireless device detected; device blocked from network;
    security team notified.
  sentences:
  - Boundary Protection. Use firewalls/segmentation to monitor and control communications.
  - Event Logging. Define and enable auditable events across systems and applications.
  - Least Privilege. Restrict privileges to the minimum necessary; approve/administer
    role changes.
- source_sentence: Multi-factor authentication enforced for 93% of user accounts.
    Remaining 7% using legacy applications without modern authentication protocol
    support.
  sentences:
  - Protection of Information at Rest. Encrypt data at rest; enforce KMS policies
    and access control.
  - Cryptographic Key Establishment and Management. Manage key lifecycles (KMS/HSM),
    rotation, and separation.
  - Identification and Authentication (Org Users). Use MFA and unique IDs for users
    and services.
- source_sentence: Quarterly vulnerability assessment identified 67 findings requiring
    remediation. 12 rated critical severity with 7-day remediation SLA actively being
    tracked.
  sentences:
  - Account Management. Provision, review, and remove accounts; enforce least privilege
    and approvals.
  - Flaw Remediation. Track patches, apply within policy windows, verify results.
  - Remote Access. Authorize and monitor remote connections; require secure channels
    and MFA.
- source_sentence: Change request CR-2025-089 approved but implemented with additional
    undocumented modifications.
  sentences:
  - Audit Review, Analysis, and Reporting. Regularly review and analyze audit logs;
    track anomalies and findings.
  - System Monitoring. Detect anomalous behavior via IDS/EDR/SIEM.
  - Configuration Change Control. Approve, test, and document changes; enforce change
    windows.
- source_sentence: File integrity monitoring detected changes to critical system files;
    investigation initiated
  sentences:
  - Unsuccessful Logon Attempts. Enforce lockout thresholds and durations after consecutive
    failed logins.
  - Remote Access. Authorize and monitor remote connections; require secure channels
    and MFA.
  - Software, Firmware, and Information Integrity. Use integrity checks (FIM, checksums)
    and signed updates.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- dot_accuracy@1
- dot_accuracy@3
- dot_accuracy@5
- dot_accuracy@10
- dot_precision@1
- dot_precision@3
- dot_precision@5
- dot_precision@10
- dot_recall@1
- dot_recall@3
- dot_recall@5
- dot_recall@10
- dot_ndcg@10
- dot_mrr@10
- dot_map@100
model-index:
- name: SentenceTransformer based on sentence-transformers/multi-qa-mpnet-base-dot-v1
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: dev mrr
      type: dev_mrr
    metrics:
    - type: dot_accuracy@1
      value: 0.8333333333333334
      name: Dot Accuracy@1
    - type: dot_accuracy@3
      value: 0.9506172839506173
      name: Dot Accuracy@3
    - type: dot_accuracy@5
      value: 0.9814814814814815
      name: Dot Accuracy@5
    - type: dot_accuracy@10
      value: 0.9938271604938271
      name: Dot Accuracy@10
    - type: dot_precision@1
      value: 0.8333333333333334
      name: Dot Precision@1
    - type: dot_precision@3
      value: 0.5
      name: Dot Precision@3
    - type: dot_precision@5
      value: 0.3271604938271605
      name: Dot Precision@5
    - type: dot_precision@10
      value: 0.17345679012345677
      name: Dot Precision@10
    - type: dot_recall@1
      value: 0.5061728395061729
      name: Dot Recall@1
    - type: dot_recall@3
      value: 0.868312757201646
      name: Dot Recall@3
    - type: dot_recall@5
      value: 0.933127572016461
      name: Dot Recall@5
    - type: dot_recall@10
      value: 0.977366255144033
      name: Dot Recall@10
    - type: dot_ndcg@10
      value: 0.8920790605595111
      name: Dot Ndcg@10
    - type: dot_mrr@10
      value: 0.8989271017048793
      name: Dot Mrr@10
    - type: dot_map@100
      value: 0.8444494120420045
      name: Dot Map@100
---

# SentenceTransformer based on sentence-transformers/multi-qa-mpnet-base-dot-v1

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1) <!-- at revision 17997f24dca0df1a4fed68894fb0e1e133e60482 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Dot Product
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'MPNetModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'File integrity monitoring detected changes to critical system files; investigation initiated',
    'Software, Firmware, and Information Integrity. Use integrity checks (FIM, checksums) and signed updates.',
    'Unsuccessful Logon Attempts. Enforce lockout thresholds and durations after consecutive failed logins.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[38.6073, 27.5844, 12.0512],
#         [27.5844, 35.6622, 12.2060],
#         [12.0512, 12.2060, 38.5975]])
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

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `dev_mrr`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric           | Value      |
|:-----------------|:-----------|
| dot_accuracy@1   | 0.8333     |
| dot_accuracy@3   | 0.9506     |
| dot_accuracy@5   | 0.9815     |
| dot_accuracy@10  | 0.9938     |
| dot_precision@1  | 0.8333     |
| dot_precision@3  | 0.5        |
| dot_precision@5  | 0.3272     |
| dot_precision@10 | 0.1735     |
| dot_recall@1     | 0.5062     |
| dot_recall@3     | 0.8683     |
| dot_recall@5     | 0.9331     |
| dot_recall@10    | 0.9774     |
| **dot_ndcg@10**  | **0.8921** |
| dot_mrr@10       | 0.8989     |
| dot_map@100      | 0.8444     |

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

* Size: 1,160 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 10 tokens</li><li>mean: 28.87 tokens</li><li>max: 79 tokens</li></ul> | <ul><li>min: 14 tokens</li><li>mean: 19.33 tokens</li><li>max: 27 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                               | sentence_1                                                                                                  |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|
  | <code>Database connection pooling configured with TLS encryption; certificate validation enforced</code>                                                                                                                 | <code>Transmission Confidentiality and Integrity. Encrypt data in transit (TLS) with modern ciphers.</code> |
  | <code>System time synchronization restored after NTP failure; outage logged</code>                                                                                                                                       | <code>Event Logging. Define and enable auditable events across systems and applications.</code>             |
  | <code>Configuration management database (CMDB) was updated to reflect the new security baseline for all Windows servers; the baseline includes CIS benchmarks; it is enforced by a configuration monitoring tool.</code> | <code>Configuration Settings. Enforce secure configuration settings; monitor for drift.</code>              |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
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
- `num_train_epochs`: 3
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
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | dev_mrr_dot_ndcg@10 |
|:-----:|:----:|:-------------------:|
| 1.0   | 73   | 0.8921              |


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

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
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
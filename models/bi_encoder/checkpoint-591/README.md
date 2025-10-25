---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:3146
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/multi-qa-mpnet-base-dot-v1
widget:
- source_sentence: Audit logs for the period 2025-10-01 to 2025-10-05 have been successfully
    moved from hot storage to immutable cold storage (WORM) for long-term retention
    in compliance with legal hold requirements.
  sentences:
  - 'Audit Record Retention. Retain audit logs for required periods; protect from
    unauthorized access.. Common patterns: Audit record retention and generation policies
    are violated by an insecure ''overwrite'' setting.; The current audit record retention
    configuration does not meet policy requirements.; Audit retention policy violation
    requires correction.; Audit retention policy changed without proper authorization.;
    The configuration for audit record retention does not protect against data loss.'
  - 'Vulnerability Monitoring and Scanning. Perform authenticated scanning and remediation
    tracking.. Common patterns: Investigation initiated into a failure of the continuous
    vulnerability monitoring process.; Vulnerability monitoring is not configured
    with sufficient privileges to perform comprehensive scanning.; Vulnerability scanning
    and remediation performance degrading.; Vulnerability monitoring (via CSPM) detected
    a critical boundary protection weakness.; Vulnerability detection and cryptography
    compliance.'
  - 'System Monitoring. Detect anomalous behavior via IDS/EDR/SIEM.. Common patterns:
    Application-layer firewall deployed for common web threats.; File integrity monitoring
    detected suspicious read access to a critical system file.; System monitoring
    has detected anomalous behavior indicative of malware propagation.; Malware was
    blocked; system protected; event requires audit review.; Network traffic logging
    is disabled in an environment creating a gap in system monitoring.'
- source_sentence: The SNMP service on a core network switch is using the default
    'public' community string.
  sentences:
  - 'Wireless Access. Control and monitor wireless access; use strong encryption and
    auth.. Common patterns: Wireless access control and network protection.; Wireless
    access authorization lacking proper evaluation.; The wireless access configuration
    permits the use of a weak non-compliant cryptographic protocol.; Wireless authenticator
    secrecy is compromised by being publicly listed internally.; The wireless access
    solution is using a weak insecure authenticator.'
  - 'Identification and Authentication (Org Users). Use MFA and unique IDs for users
    and services.. Common patterns: Repeated failures of the multi-factor authentication
    mechanism triggered an account lockout.; The use of a shared identifier violates
    the principle of individual accountability.; Identification and authentication
    for a privileged account does not require MFA.; The use of a shared identifier
    violates the requirement for individual accountability.; Remote access security
    enhanced; strong auth enforced; old credentials revoked.'
  - 'Configuration Settings. Enforce secure configuration settings; monitor for drift..
    Common patterns: A secure configuration setting for remote access has not been
    correctly implemented.; Network rule cleanup completed; configuration optimized;
    review process documented.; Security baseline documented; compliance monitored;
    configuration management updated.; Configuration drift detected; non-compliant
    settings found; monitoring alerted and ticket created.; A secure configuration
    setting from the approved baseline is not being enforced.'
- source_sentence: Anti-malware real-time protection was manually disabled by a local
    administrator on host 'research-05'. The central management console marked the
    host as 'Non-Compliant' and sent an alert to the SOC.
  sentences:
  - 'Configuration Change Control. Approve, test, and document changes; enforce change
    windows.. Common patterns: Change deployed outside window causing issues; rollback
    executed; incident response involved.; Change scope expanded without proper authorization.;
    A configuration change request was denied because it violated the secure remote
    access policy.; The change control process was not followed correctly.; Change
    control authorization and security testing both missing.'
  - 'Configuration Change Control. Approve, test, and document changes; enforce change
    windows.. Common patterns: Change deployed outside window causing issues; rollback
    executed; incident response involved.; Change scope expanded without proper authorization.;
    A configuration change request was denied because it violated the secure remote
    access policy.; The change control process was not followed correctly.; Change
    control authorization and security testing both missing.'
  - 'Malicious Code Protection. Deploy/monitor anti-malware; update signatures and
    policies.. Common patterns: Malware protection failure increases endpoint vulnerability.;
    Incident response plan executed for malware containment.; Malicious code protection
    successfully blocked a threat and initiated an automated incident response action.;
    Incident handling malicious code protection and backup systems were all involved
    in this event.; Malware was blocked; system protected; event requires audit review.'
- source_sentence: The SSL certificate used to encrypt traffic to the internal HR
    portal expired two days ago. Users are now receiving browser warnings, and the
    connection is technically insecure until the certificate is renewed and deployed.
  sentences:
  - 'Account Management. Provision, review, and remove accounts; enforce least privilege
    and approvals.. Common patterns: A failure in the account management de-provisioning
    process has left the organization at risk.; Account provisioning automated; access
    based on HR data; process logged for audit.; The account management provisioning
    process was followed correctly.; A failure in the manual account management process
    resulted in a terminated employee retaining access.; A routine account management
    action was completed.'
  - 'Boundary Protection. Use firewalls/segmentation to monitor and control communications..
    Common patterns: Network rule cleanup completed; configuration optimized; review
    process documented.; Boundary protection for outbound traffic is overly permissive
    increasing the risk of data exfiltration.; Application-layer firewall deployed
    for common web threats.; Network boundary protection failed to block traffic between
    zones that should be isolated.; Boundary protection controls for a non-production
    environment are not correctly configured.'
  - 'Authenticator Management. Manage credential lifecycle; complexity, rotation,
    and revocation.. Common patterns: The wireless access solution is using a weak
    insecure authenticator.; Authenticator management controls are failing to enforce
    basic complexity requirements.; Authenticator management controls for a privileged
    network device account are inadequate.; Transmission confidentiality is weakened
    by the use of an untrusted authenticator (self-signed certificate).; Configuration
    drift detected for password policy; non-compliant settings found; monitoring alerted.'
- source_sentence: A user has requested, and been granted, read-only access to the
    marketing team's shared drive. The access was approved by the data owner via email,
    and the approval is attached to this service desk ticket.
  sentences:
  - 'Account Management. Provision, review, and remove accounts; enforce least privilege
    and approvals.. Common patterns: A failure in the account management de-provisioning
    process has left the organization at risk.; Account provisioning automated; access
    based on HR data; process logged for audit.; The account management provisioning
    process was followed correctly.; A failure in the manual account management process
    resulted in a terminated employee retaining access.; A routine account management
    action was completed.'
  - 'Account Management. Provision, review, and remove accounts; enforce least privilege
    and approvals.. Common patterns: A failure in the account management de-provisioning
    process has left the organization at risk.; Account provisioning automated; access
    based on HR data; process logged for audit.; The account management provisioning
    process was followed correctly.; A failure in the manual account management process
    resulted in a terminated employee retaining access.; A routine account management
    action was completed.'
  - 'Configuration Settings. Enforce secure configuration settings; monitor for drift..
    Common patterns: A secure configuration setting for remote access has not been
    correctly implemented.; Network rule cleanup completed; configuration optimized;
    review process documented.; Security baseline documented; compliance monitored;
    configuration management updated.; Configuration drift detected; non-compliant
    settings found; monitoring alerted and ticket created.; A secure configuration
    setting from the approved baseline is not being enforced.'
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
      value: 0.8602739726027397
      name: Dot Accuracy@1
    - type: dot_accuracy@3
      value: 0.9780821917808219
      name: Dot Accuracy@3
    - type: dot_accuracy@5
      value: 0.9863013698630136
      name: Dot Accuracy@5
    - type: dot_accuracy@10
      value: 0.9917808219178083
      name: Dot Accuracy@10
    - type: dot_precision@1
      value: 0.8602739726027397
      name: Dot Precision@1
    - type: dot_precision@3
      value: 0.4986301369863014
      name: Dot Precision@3
    - type: dot_precision@5
      value: 0.3063013698630137
      name: Dot Precision@5
    - type: dot_precision@10
      value: 0.15726027397260273
      name: Dot Precision@10
    - type: dot_recall@1
      value: 0.5990867579908675
      name: Dot Recall@1
    - type: dot_recall@3
      value: 0.9461187214611871
      name: Dot Recall@3
    - type: dot_recall@5
      value: 0.9643835616438357
      name: Dot Recall@5
    - type: dot_recall@10
      value: 0.9863013698630136
      name: Dot Recall@10
    - type: dot_ndcg@10
      value: 0.9269255337519096
      name: Dot Ndcg@10
    - type: dot_mrr@10
      value: 0.9189954337899543
      name: Dot Mrr@10
    - type: dot_map@100
      value: 0.8990152009330091
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
    "A user has requested, and been granted, read-only access to the marketing team's shared drive. The access was approved by the data owner via email, and the approval is attached to this service desk ticket.",
    'Account Management. Provision, review, and remove accounts; enforce least privilege and approvals.. Common patterns: A failure in the account management de-provisioning process has left the organization at risk.; Account provisioning automated; access based on HR data; process logged for audit.; The account management provisioning process was followed correctly.; A failure in the manual account management process resulted in a terminated employee retaining access.; A routine account management action was completed.',
    'Account Management. Provision, review, and remove accounts; enforce least privilege and approvals.. Common patterns: A failure in the account management de-provisioning process has left the organization at risk.; Account provisioning automated; access based on HR data; process logged for audit.; The account management provisioning process was followed correctly.; A failure in the manual account management process resulted in a terminated employee retaining access.; A routine account management action was completed.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[43.8461, 29.5080, 29.5080],
#         [29.5080, 42.4973, 42.4973],
#         [29.5080, 42.4973, 42.4973]])
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
| dot_accuracy@1   | 0.8603     |
| dot_accuracy@3   | 0.9781     |
| dot_accuracy@5   | 0.9863     |
| dot_accuracy@10  | 0.9918     |
| dot_precision@1  | 0.8603     |
| dot_precision@3  | 0.4986     |
| dot_precision@5  | 0.3063     |
| dot_precision@10 | 0.1573     |
| dot_recall@1     | 0.5991     |
| dot_recall@3     | 0.9461     |
| dot_recall@5     | 0.9644     |
| dot_recall@10    | 0.9863     |
| **dot_ndcg@10**  | **0.9269** |
| dot_mrr@10       | 0.919      |
| dot_map@100      | 0.899      |

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

* Size: 3,146 training samples
* Columns: <code>sentence1</code> and <code>sentence2</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence1                                                                          | sentence2                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              |
  | details | <ul><li>min: 14 tokens</li><li>mean: 42.44 tokens</li><li>max: 67 tokens</li></ul> | <ul><li>min: 80 tokens</li><li>mean: 94.53 tokens</li><li>max: 124 tokens</li></ul> |
* Samples:
  | sentence1                                                                                                                    | sentence2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
  |:-----------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>User 'svc-api' failed login 11 times in 2 minutes; account was not automatically locked.</code>                        | <code>Audit Review, Analysis, and Reporting. Regularly review and analyze audit logs; track anomalies and findings.. Common patterns: Failed login threshold met and audit review performed.; Failed login threshold and review both applied.; Malware was blocked; system protected; event requires audit review.; The configuration for audit review and analysis is not effective at detecting password spraying attacks.; SIEM tuning improves detection accuracy; configuration updated; review process documented.</code>                                          |
  | <code>User 'svc-api' failed login 11 times in 2 minutes; account was not automatically locked.</code>                        | <code>Unsuccessful Logon Attempts. Enforce lockout thresholds and durations after consecutive failed logins.. Common patterns: Repeated failures of the multi-factor authentication mechanism triggered an account lockout.; The unsuccessful logon attempt policy is being enforced.; The unsuccessful logon attempt policy was successfully enforced by the system.; A distributed login attack bypassed a simple lockout policy requiring centralized audit and review.; The policy for unsuccessful logon attempts is not being applied to all user accounts.</code> |
  | <code>CHG-9901: Emergency hotfix for API memory leak deployed to production; the standard SAST pipeline was bypassed.</code> | <code>Developer Testing and Evaluation. Require security testing (SAST/DAST), code review, and defect tracking.. Common patterns: Developer security testing requirement formally excepted.; Developer security testing has successfully identified and blocked a critical vulnerability.; The process for developer security testing is not automated and is therefore not scalable.; Change control authorization and security testing both missing.; The required developer security testing process has been disabled.</code>                                        |
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

- `eval_strategy`: epoch
- `per_device_train_batch_size`: 16
- `learning_rate`: 2e-05
- `warmup_steps`: 59
- `load_best_model_at_end`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: epoch
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 59
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
- `load_best_model_at_end`: True
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
| Epoch  | Step | Training Loss | dev_mrr_dot_ndcg@10 |
|:------:|:----:|:-------------:|:-------------------:|
| 0.0508 | 10   | 1.8493        | -                   |
| 0.1015 | 20   | 1.77          | -                   |
| 0.1523 | 30   | 1.4084        | -                   |
| 0.2030 | 40   | 1.4866        | -                   |
| 0.2538 | 50   | 1.2544        | -                   |
| 0.3046 | 60   | 1.4606        | -                   |
| 0.3553 | 70   | 1.2176        | -                   |
| 0.4061 | 80   | 1.1951        | -                   |
| 0.4569 | 90   | 1.1378        | -                   |
| 0.5076 | 100  | 1.1325        | -                   |
| 0.5584 | 110  | 1.0245        | -                   |
| 0.6091 | 120  | 1.144         | -                   |
| 0.6599 | 130  | 1.1416        | -                   |
| 0.7107 | 140  | 1.1546        | -                   |
| 0.7614 | 150  | 1.1623        | -                   |
| 0.8122 | 160  | 1.0721        | -                   |
| 0.8629 | 170  | 1.0169        | -                   |
| 0.9137 | 180  | 1.1321        | -                   |
| 0.9645 | 190  | 0.9793        | -                   |
| 1.0    | 197  | -             | 0.9084              |
| 1.0152 | 200  | 0.9083        | -                   |
| 1.0660 | 210  | 0.9601        | -                   |
| 1.1168 | 220  | 0.9917        | -                   |
| 1.1675 | 230  | 0.7675        | -                   |
| 1.2183 | 240  | 0.886         | -                   |
| 1.2690 | 250  | 1.0081        | -                   |
| 1.3198 | 260  | 0.8606        | -                   |
| 1.3706 | 270  | 1.0823        | -                   |
| 1.4213 | 280  | 0.8823        | -                   |
| 1.4721 | 290  | 0.9267        | -                   |
| 1.5228 | 300  | 0.973         | -                   |
| 1.5736 | 310  | 0.9842        | -                   |
| 1.6244 | 320  | 1.0247        | -                   |
| 1.6751 | 330  | 0.9166        | -                   |
| 1.7259 | 340  | 0.9536        | -                   |
| 1.7766 | 350  | 0.9508        | -                   |
| 1.8274 | 360  | 1.0068        | -                   |
| 1.8782 | 370  | 1.0299        | -                   |
| 1.9289 | 380  | 0.9348        | -                   |
| 1.9797 | 390  | 0.9711        | -                   |
| 2.0    | 394  | -             | 0.9245              |
| 2.0305 | 400  | 0.7576        | -                   |
| 2.0812 | 410  | 0.8276        | -                   |
| 2.1320 | 420  | 0.9192        | -                   |
| 2.1827 | 430  | 0.8247        | -                   |
| 2.2335 | 440  | 0.8691        | -                   |
| 2.2843 | 450  | 0.8257        | -                   |
| 2.3350 | 460  | 0.8747        | -                   |
| 2.3858 | 470  | 0.8218        | -                   |
| 2.4365 | 480  | 0.8055        | -                   |
| 2.4873 | 490  | 0.8492        | -                   |
| 2.5381 | 500  | 0.9347        | -                   |
| 2.5888 | 510  | 0.8629        | -                   |
| 2.6396 | 520  | 0.8968        | -                   |
| 2.6904 | 530  | 0.9188        | -                   |
| 2.7411 | 540  | 0.9907        | -                   |
| 2.7919 | 550  | 0.8627        | -                   |
| 2.8426 | 560  | 0.91          | -                   |
| 2.8934 | 570  | 0.8563        | -                   |
| 2.9442 | 580  | 0.8317        | -                   |
| 2.9949 | 590  | 0.9613        | -                   |
| 3.0    | 591  | -             | 0.9269              |


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
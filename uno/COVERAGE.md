## All Functions with Data Utils

- [] set_up_logger
- [] set_seed
- [] get_file
- [] impute_and_scale
- [] seperator_print
- [] discretize
- [] save_combined_dose_response
- [] load_combined_dose_response
- [] load_single_dose_response
- [] load_combo_dose_response
- [] load_aggregated_single_response
- [] load_drug_data
- [] load_mordred_descriptors
- [] load_drug_descriptors
- [] load_drug_fingerprints
- [] load_drug_info
- [] lookup
- [] load_cell_metadata
- [] cell_name_to_ids
- [] drug_name_to_ids
- [] load_drug_set_descriptors
- [] load_drug_set_fingerprints
- [] load_drug_smiles
- [] encode_sources
- [] load_cell_rnaseq
- [] read_set_from_file
- [] select_drugs_with_response_range
- [] summarize_response_data
- [] assign_partition_groups
- [] dict_compare
- [] values_or_dataframe
- [] CombinedDataLoader
- [] DataFeeder
- [] CombinedDataGenerator



## Run 1

```bash
python uno_baseline_keras2.py --config_file uno_auc_model.txt
```

- [] CombineDataLoader.load
- [] load_aggregated_single_response
- [] summarize_response_data
- [] encode_sources
- [] read_set_from_file
- [] load_cell_rnaseq
- [] load_drug_info
- [] load_drug_set_descriptors
- [] load_drug_descriptors
- [] summarize_response_data
- [] assign_partition_groups
- [] CombineDataLoader.build_feature_list
- [] CombinedDataLoader.load
- [] CombineDataLoader.partition_data
- [] CombinedDataGenerator
- [] CombinedDataGenerator.get_response
- [] CombinedDataGenerator.reset


## Run 2

```bash
python uno_baseline_keras2.py --train_sources all --cache cache/all --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True
```

- [] load_single_dose_response
- [] load_combo_dose_response
- [] encode_sources
- [] select_drugs_with_response_range
- [] read_set_from_file
- [] load_cell_rnaseq
- [] load_drug_info
- [] load_drug_set_descriptors
- [] load_drug_set_descriptors
- [] load_drug_descriptors
- [] load_drug_info
- [] load_drug_set_fingerprints
- [] load_drug_fingerprints
- [] summarize_response_data
- [] assign_partition_groups
- [] CombinedDataLoader.load


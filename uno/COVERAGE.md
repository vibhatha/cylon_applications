## All Functions with Data Utils

1. set_up_logger
2. set_seed
3. get_file
4. impute_and_scale
5. seperator_print
6. discretize
7. save_combined_dose_response
8. load_combined_dose_response
9. load_single_dose_response
10. load_combo_dose_response
11. load_aggregated_single_response
12. load_drug_data
13. load_mordred_descriptors
14. load_drug_descriptors
15. load_drug_fingerprints
16. load_drug_info
17. lookup
18. load_cell_metadata
19. cell_name_to_ids
20. drug_name_to_ids
21. load_drug_set_descriptors
22. load_drug_set_fingerprints
23. load_drug_smiles
24. encode_sources
25. load_cell_rnaseq
26. read_set_from_file
27. select_drugs_with_response_range
28. summarize_response_data
29. assign_partition_groups
30. dict_compare
31. values_or_dataframe
32. CombinedDataLoader
33. DataFeeder
34. CombinedDataGenerator
35.


## Run 1

```bash
python uno_baseline_keras2.py --config_file uno_auc_model.txt
```

1. CombineDataLoader.load
2. load_aggregated_single_response
3. summarize_response_data
4. encode_sources
5. read_set_from_file
6. load_cell_rnaseq
7. load_drug_info
8. load_drug_set_descriptors
9. load_drug_descriptors
10. summarize_response_data
11. assign_partition_groups
12. CombineDataLoader.build_feature_list
13. CombinedDataLoader.load
14. CombineDataLoader.partition_data
15. CombinedDataGenerator
16. CombinedDataGenerator.get_response
17. CombinedDataGenerator.reset


## Run 2

```bash
python uno_baseline_keras2.py --train_sources all --cache cache/all --use_landmark_genes True --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True
```

1. load_single_dose_response
2. load_combo_dose_response
3. encode_sources
4. select_drugs_with_response_range
5. read_set_from_file
6. load_cell_rnaseq
7. load_drug_info
8. load_drug_set_descriptors
9. load_drug_set_descriptors
10. load_drug_descriptors
11. load_drug_info
12. load_drug_set_fingerprints
13. load_drug_fingerprints
14. summarize_response_data
15. assign_partition_groups
16. CombinedDataLoader.load


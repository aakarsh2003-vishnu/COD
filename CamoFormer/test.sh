python test.py \
    --test_img_dir '../MCOD_processed/test/official_false_colour' \
    --test_mask_dir '../MCOD_processed/test/ground_truth_mask' \
    --dataset_name 'MCOD' \
    --save_root 'output/Prediction/CamoFormer_MCOD' \
    --ckpt 'checkpoint/CamoFormer-trained.pth'

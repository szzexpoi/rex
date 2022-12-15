# REX: Reasoning-aware and Grounded Explanation

This code implements the Reasoning-aware and Grounded EXplanation (REX) framework. It consists of:
- a new GQA-REX dataset with 1,040,830 multi-modal explanations for visual reasoning, and a functional program for automatically constructing the explanations based on reasoning process
- a novel explanation generation method that explicitly maps visual grounding results to explanations

### Reference
If you use our code or data, please cite our paper:
```
@InProceedings{rex2022,
author = {Chen, Shi and Zhao, Qi},
title = {REX: Reasoning-aware and Grounded Explanation},
booktitle = {CVPR},
year = {2022}
}
```

### Disclaimer
We adopt VisualBert implemented in the [Transformers](https://github.com/huggingface/transformers) library as the backbone visual reasoning model. We use the bottom-up features provided in [this repository](https://github.com/airsplay/lxmert). Please refer to these links for further README information.

### Requirements
1. Requirements for Pytorch. We use Pytorch 1.9.0 in our experiments.
2. Requirements for Tensorflow. We only use the tensorboard for visualization.
3. Requirements for Transformers
4. Requirements for [COCO Caption Evaluation](https://github.com/salaniz/pycocoevalcap), please clone the repo to `ROOT/model`, where `ROOT` is the root directory of our project, and install the corresponding dependencies.

### Data
1. Download our [GQA-REX dataset](https://drive.google.com/file/d/1tppPRVtiLTwc_oYQw6Q0OvSGFpI8ZCtg/view?usp=sharing). The file includes both the raw explanations and converted explanations for model training. The explanations correspond to balanced questions in the GQA dataset. We also provide the [explanations for all 14M GQA training questions](https://drive.google.com/file/d/1UupJyqbnlTx88Vtggex9MoVuxAw89xnU/view?usp=sharing).
2. Download the [GQA Dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html).
3. Download the [GQA-OOD Dataset](https://github.com/gqa-ood/GQA-OOD)
4. Download the [bottom-up features](https://github.com/airsplay/lxmert) and unzip it.
5. Extracting features from the raw tsv files (**Important**: You need to run the code with Python 2):
  ```
  python2 ./preprocessing/extract_tsv.py --input $TSV_FILE --output $FEATURE_DIR
  ```

### Constructing Explanations from Scratch
We also provide our functional program for constructing the explanations from scratch:
1. Generate our atomic operations abstracted from GQA annotations:
  ```
  cd ./pre_processing
  python process_semantics_exp.py --question $GQA_ROOT/question --mapping ./data --save ./data
  ```
2. Generate raw explanations:
  ```
  python exp_generator.py --question $GQA_ROOT/ --data ./data --save ./data
  python post_processing.py --question $GQA_ROOT/ --data ./data --save ./data
  ```
3. Converting the raw explanations for modeling
  ```
  python convert_explanation --question $GQA_ROOT/ --data ./data --bbox $FEATURE_DIR/box --save ./data
  python finalize_exp.py --data ./data --save $EXP_DIR
  ```

### Explanation Generation Experiments
We provide the code for experimenting with our explanation generation method under two different settings, including multi-task learning and transfer learning. Before training with our method, you need to first generate the dictionary for questions, answers, and explanations:
  ```
  cd ./model
  python generate_dictionary --question $GQA_ROOT/question --exp $EXP_DIR --save ./processed_data
  ```

For the multi-task learning experiments, the training process can be called as:
  ```
  python main.py --mode train --anno_dir $GQA_ROOT/question --ood_dir $OOD_ROOT/data --sg_dir $GQA_ROOT/scene_graph --lang_dir ./processed_data --img_dir $FEATURE_DIR/features --bbox_dir $FEATURE_DIR/box --checkpoint_dir $CHECKPOINT --use_structure 1
  ```
To evaluate on the GQA-testdev set or generating submission file for online evaluation on the test-standard set, call:
  ```
  python main.py --mode $MODE --anno_dir $GQA_ROOT/question --ood_dir $OOD_ROOT/data --lang_dir ./processed_data --img_dir $FEATURE_DIR/features --weights $CHECKPOINT/model_best.pth --use_structure 1
  ```
and set `$MODE` to `eval` or `submission` accordingly.

For the transfer learning experiment, you will first train the model on explanation generation alone:
  ```
  python main_transfer.py --mode train --anno_dir $GQA_ROOT/question --sg_dir $GQA_ROOT/scene_graph --lang_dir ./processed_data --img_dir $FEATURE_DIR/features --bbox_dir $FEATURE_DIR/box --checkpoint_dir $CHECKPOINT_EXP --use_structure 1 anno_type exp
  ```

Before training the model on the subsets for both question answering and explanation generation, you need to first create the question files for the subsets:
   ```
   python create_subset.py --question $GQA_ROOT/question
   ```

After that, you can start training with a specific percentage (`$PERCENTAGE`, e.g., 1, 5, 10) of annotations:
```
python main_transfer.py --mode train --anno_dir $GQA_ROOT/question --ood_dir $OOD_ROOT/data --sg_dir $GQA_ROOT/scene_graph --lang_dir ./processed_data --img_dir $FEATURE_DIR/features --bbox_dir $FEATURE_DIR/box --checkpoint_dir $CHECKPOINT_VQA --use_structure 1 anno_type vqa --percentage $PERCENTAGE --epoch 15
```

For evaluation on the test set:
```
python main_transfer.py --mode test --anno_dir $GQA_ROOT/question --ood_dir $OOD_ROOT/data --lang_dir ./processed_data --img_dir $FEATURE_DIR/features --weights $CHECKPOINT_VQA/model_best.pth --use_structure 1
```

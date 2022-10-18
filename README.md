# ARL (Action Representation Learning)

## Prerequisites ##
- Tested on Ubuntu 16.04
- Python 3
- NVIDIA GPU + CUDA 9.0
- Clone this repository\
<code>git clone https://github.com/shailaja183/ARL.git</code>
- Go to ARL directory\
<code>cd ARL</code>

## ImagetoScene ##

Once set up, this module can take any CLEVR-style image and obtain corresponding scene graph. 

- Create a conda environment and install necessary python packages\
<code>conda create --name im2scene_scene2qa -c conda-forge pytorch --file requirements.txt</code>
- Activate the conda environment\
<code>source activate im2scene_scene2qa</code>
- Go to ImagetoScene directory\
<code>cd ImagetoScene</code>
- Download data and pretrained Mask-RCNN backbone\
<code>sh download.sh</code>
- Compile CUDA for Mask-RCNN\
<code>cd mask_rcnn/lib</code>
<code>sh make.sh</code>
- Obtain object proposals\
<code>cd ..</code>\
<code>python tools/test_net.py --dataset clevr_original_val --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml --load_ckpt ../../data/pretrained/object_detector.pt --output_dir ../../data/mask_rcnn/results/clevr_val_pretrained</code>\
This will generate ARL/data/mask_rcnn/results/clevr_val_pretrained/detections.pkl file.
- Process detection results\
<code>cd ../attr_net</code>\
<code>python tools/process_proposals.py --dataset clevr --proposal_path ../../data/mask_rcnn/results/clevr_val_pretrained/detections.pkl --output_path ../../data/attr_net/objects/clevr_val_objs_pretrained.json</code>\
This will generate ARL/data/attr_net/objects/clevr_val_objs_pretrained.json file.
- Attribute extraction\
<code>python tools/run_test.py --run_dir ../data/attr_net/results --dataset clevr --load_checkpoint_path ../../data/pretrained/attribute_net.pt --clevr_val_ann_path ../../data/attr_net/objects/clevr_val_objs_pretrained.json --output_path ../../data/attr_net/results/clevr_val_scenes_parsed_pretrained.json</code>\
This will generate ARL/data/attr_net/results/clevr_val_scenes_parsed_pretrained.json file.

## ActionEncoder_EffectDecoder_NLtoAV ## 

- Create conda environment with Python 3 and install Pytorch 1.0\
<code>conda create --name en_de_nltoav</code>\
<code>source activate en_de_nltoav</code>\
<code>pip install pytorch=1.0.0</code>
- This module corresponds to learning in Stage 1+Stage 2 (Figure 5 of the paper):\
Stage-1:\
(i) ActionEncoder: Learn <difference_of_scenes> provided <initial_scene, updated_scene><br> 
(ii) EffectDecoder: Reconstruct <updated_scene> based on <initial_scene, learned_difference_of_scenes>\
Stage-2:\
NLtoAV: Learn <action_vector_representation> such that the loss with <learned_difference_of_scenes> is minimized.
- To run both Stage 1+Stage 2 together and create new encoder/decoder checkpoints\
<code>python evaluate.py</code>
- To run both Stage 1+Stage 2 together and load existing encoder/decoder checkpoints with a particular prefix\
<code>python evaluate.py --pretrain_prefix "best_"</code>
- To run Stage 2 without Stage 1 for ablations, run the following command\
<code>python3 evaluate.py --no_pretraining</code>

## ScenetoQA ##

Once set up, this module can take a scene graph in CLEVR-style, a question and obtain corresponding answer.
- Activate conda environment\
<code>source activate im2scene_scene2qa</code>
- Go to ScenetoQA directory\
<code>cd ScenetoQA</code>
- Obtain answers\
<code>python tools/run_test.py --run_dir ../data/reason/results --load_checkpoint_path ../data/pretrained/question_parser.pt --clevr_val_scene_path ../data/attr_net/results/clevr_val_scenes_parsed_pretrained.json --save_result_path ../data/reason/results/result_pretrained.json</code>\
This will generate ARL/data/reason/results/result_pretrained.json containing answers and accuracy of QA. 

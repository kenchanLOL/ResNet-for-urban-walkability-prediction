# ResNet-for-urban-walkability-prediction
A light-weighted ResNet modification for building reward model on image-based RLHF pipeline
![image](https://github.com/kenchanLOL/ResPrefNet/assets/55791584/1ab770bb-c965-4b2b-a9be-8ba8455e1e54)

## Usage

1. place your images in ``` images/ ``` folder and rename them as ```image_{index}.jpg ```
2. place your preference labels in ``` data/ ``` folder
3. download resnet50 pretrained weight and put it int ``` models/ ``` folder
4. train the model using the commend
```
python train.py --train data/train_judgements.csv --test data/test_judgements.csv --val data/validation_judgements.csv --resnet models/resnet50_best.pth --batch_size 8 --num_workers 4 --num_epoch 50 --lr 1e-3 --eval_ep 8 --grad_accum 8
```
5. run the visualization code
```
python visualize.py
```

## Sample Case : HK walkability 

## Data Visualization
Change of elo during sample runs
![trueskill_change_record](https://user-images.githubusercontent.com/55791584/205874429-cc5d422f-fabd-4c61-a776-c33de9d6cc96.png)
Distribution of overall and individual images
![overall_dist](https://user-images.githubusercontent.com/55791584/205874664-a9990eca-e896-4eb4-8178-1e1e2d47da1a.png)
![individual_dist](https://user-images.githubusercontent.com/55791584/205874685-b50f4c0a-c369-40e2-9acc-ade74322328e.png)

## Part 4: Results and Visualization with CAM
After undergoing fine-tuning for 50 epochs on a compact dataset with a low image-to-preference ratio, comprising 10,000 preferences across 3,000 images (averaging three comparisons per image), the model attained 95% of the desired ELO performance. GradCAM visualizations reveal that the model has adeptly internalized human perceptual patterns related to walkability, specifically honing in on key pedestrian infrastructures such as traffic lights, fences, and bridges, as well as identifying impediments like vehicles and unauthorized street blockages by goods.

| Model | Accuracy |
| ----------- | ----------- |
| Elo Score (Baseline) | 77.8% |
| ResNet50 | 73.8% |

### ROC
![image](https://github.com/kenchanLOL/ResPrefNet/assets/55791584/efb5abdf-0d90-4a76-af4e-bce0af608430)
### Prediction Distribution
![image](https://github.com/kenchanLOL/ResPrefNet/assets/55791584/7344a5f0-bc95-4bcb-a6b0-66c990dbddc2)
### Prediction Difference Distribution
![image](https://github.com/kenchanLOL/ResPrefNet/assets/55791584/fc59e93c-400f-4157-b080-2bbc516b2618)

### Resnet CAM
![resnet101_true_inverse_CAM_test](https://user-images.githubusercontent.com/55791584/205890213-f75d14c4-3da9-445d-a5d9-9e6c6c6ce7cb.jpg)

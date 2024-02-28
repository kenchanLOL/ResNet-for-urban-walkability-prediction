# ResPrefNet
A light-weighted ResNet modification for building reward model on image-based RLHF pipeline

## Sample Case : HK walkability 

## Data Visualization
Change of elo during sample runs
![trueskill_change_record](https://user-images.githubusercontent.com/55791584/205874429-cc5d422f-fabd-4c61-a776-c33de9d6cc96.png)
Distribution of overall and individual images
![overall_dist](https://user-images.githubusercontent.com/55791584/205874664-a9990eca-e896-4eb4-8178-1e1e2d47da1a.png)
![individual_dist](https://user-images.githubusercontent.com/55791584/205874685-b50f4c0a-c369-40e2-9acc-ade74322328e.png)

## Part 4: Results and Visualization with CAM
| Model | Accuracy |
| ----------- | ----------- |
| Elo Baseline | 77.8% |
| ResNet50 | 0.73.8% |

### ROC
![image](https://github.com/kenchanLOL/ResPrefNet/assets/55791584/efb5abdf-0d90-4a76-af4e-bce0af608430)
### Prediction Distribution
![image](https://github.com/kenchanLOL/ResPrefNet/assets/55791584/7344a5f0-bc95-4bcb-a6b0-66c990dbddc2)
### Prediction Difference Distribution
![image](https://github.com/kenchanLOL/ResPrefNet/assets/55791584/fc59e93c-400f-4157-b080-2bbc516b2618)

### Resnet CAM
![resnet101_true_inverse_CAM_test](https://user-images.githubusercontent.com/55791584/205890213-f75d14c4-3da9-445d-a5d9-9e6c6c6ce7cb.jpg)

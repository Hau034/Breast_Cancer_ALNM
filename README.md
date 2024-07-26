# Breast_Cancer_ALNM
# Computer-assisted Diagnosis for Axillary Lymph Node Metastasis of Early Breast Cancer based on Transformer with Dual-Modal Adaptive Mid-Term Fusion using ultrasound elastography
![model](https://github.com/user-attachments/assets/5df2b9ae-c5b3-4626-9c28-d801b8170ab8)
## Introction
 - **Objectives**：We propose a human-AI collaboration strategy to assist the radiologists for the noninvasive assessment and locational assessment of the ALNM, involving a novel transformer-based deep learning framework termed as DAMF-former to predict the metastatic status of the ALNs. 
 - **Method**: This study prospectively collected data from Jan. 2019 to Dec. 2023, involving a total of 526 female patients with 1298 ALNs. The DAMF-former uses the ViT as the backbone network with an adaptive mid-term fusion strategy to alternatively extract and adaptively fuse dual-modal features of UE images of ALNs in the axillary region, which attempts to mimic the radiologists’ observation for the assessment of the ALN status. The strategy of adaptive Youden index is designed to deal with the fully fused dual-modal image features to further improve the diagnosis outcome of the designed DAMF-former for ALNM. Accuracy, sensitivity, specificity, receiver operating characteristic (ROC) curves, and areas under the ROC curve (AUCs) were analyzed to evaluate our model.
 - **Results**: The DAMF-former on dual-modal images achieves better diagnostic performance than those on unimodal images, with AUCs of 0.933 (95% CI: 0.890, 0.976) vs. 0.856 (95% CI: 0.795-0.918) and 0.825 (95% CI: 0.756-0.894). Furthermore, ablation experiments show that the "Mid-term fusion," "Adaptive fusion," and "Adaptive Youden index" strategies we designed enabled the model to achieve the best diagnostic performance with an AUC of 0.933 (95% CI: 0.890, 0.976), 91.1% accuracy (95% CI: 0.890, 0.976), 82.5% sensitivity (95% CI: 0.869, 0.942), and 93.8% specificity (95% CI: 0.894, 0.968). Additionally, clinical experiments show that compared to independent diagnosis, junior and attending radiologists achieve better diagnostic outcomes when assisted by the model, with improvements in diagnostic AUC of 7.6% and 4.0%, respectively.
 - **Conclusion**: our study provides a promising CAD method for ALNM assessment, which has the potential to serve as an effective auxiliary tool of human-AI collaboration to improve the radiologists’ diagnostic performance for the patients with early breast cancer.


## Setup

 ### Environment
 Create environment and install dependencies.

    conda env create -f requirements.yaml

## Training
We have provided examples of the required data format for training in the dataset folder. If you wish to try more configurations, please refer to the details in `1_train_BmodeSwe.py`.

    cd Breast_cancer\code\BmodeSwe
    python 1_train_BmodeSwe
    


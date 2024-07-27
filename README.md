# Breast_Cancer_ALNM
# Computer-assisted Diagnosis for Axillary Lymph Node Metastasis of Early Breast Cancer based on Transformer with Dual-Modal Adaptive Mid-Term Fusion using ultrasound elastography
![model](https://github.com/user-attachments/assets/5df2b9ae-c5b3-4626-9c28-d801b8170ab8)
## Introction
 - **Objectives**：We propose a human-AI collaboration strategy to assist the radiologists for the noninvasive assessment and locational assessment of the ALNM, involving a novel transformer-based deep learning framework termed as DAMF-former to predict the metastatic status of the ALNs. 
 - **Method**: This study prospectively collected data from Jan. 2019 to Dec. 2023, involving a total of 519 female patients with 1280 ALNs. The DAMF-former uses the ViT as the backbone network with an adaptive mid-term fusion strategy to alternatively extract and adaptively fuse dual-modal features of UE images of ALNs in the axillary region, which attempts to mimic the radiologists’ observation for the assessment of the ALN status. The strategy of adaptive Youden index is designed to deal with the fully fused dual-modal image features to further improve the diagnosis outcome of the designed DAMF-former for ALNM. Accuracy, sensitivity, specificity, receiver operating characteristic (ROC) curves, and areas under the ROC curve (AUCs) were analyzed to evaluate our model.
 - **Results**: The DAMF-former on dual-modal images achieves better diagnostic performance than those on unimodal images, with AUCs of 0.933 (95% CI: 0.890, 0.976) vs. 0.856 (95% CI: 0.795-0.918) and 0.825 (95% CI: 0.756-0.894). Furthermore, ablation experiments show that the "Mid-term fusion," "Adaptive fusion," and "Adaptive Youden index" strategies we designed enabled the model to achieve the best diagnostic performance with an AUC of 0.933 (95% CI: 0.890, 0.976), 91.1% accuracy (95% CI: 0.890, 0.976), 82.5% sensitivity (95% CI: 0.869, 0.942), and 93.8% specificity (95% CI: 0.894, 0.968). Additionally, clinical experiments show that compared to independent diagnosis, junior and attending radiologists achieve better diagnostic outcomes when assisted by the model, with improvements in diagnostic AUC of 7.6% and 4.0%, respectively.
 - **Conclusion**: our study provides a promising CAD method for ALNM assessment, which has the potential to serve as an effective auxiliary tool of human-AI collaboration to improve the radiologists’ diagnostic performance for the patients with early breast cancer.

## Setup

 ### Environment
 Create environment and install dependencies.

    conda env create -f requirements.yaml
## Dataset
Due to ethical restrictions, patient images are for internal research use only, and we regret that we cannot disclose the entire dataset. To verify the authenticity of our research, we have provided sample patient files for testing and debugging. These samples can be used for model testing and to demonstrate our CAD demo software.

Please enter the `Sample` folder to view.
## Training
We have provided examples of the required data format for training in the dataset folder. If you wish to try more configurations, please refer to the details in `1_train_BmodeSwe.py`.

    cd code
    python 1_train_bmodeSwe
    
Furthermore, if you want to try other settings(e.g., network fusion terms), please see `1_train_BmodeSwe.py` for more details.
Please download the pre-trained weights and testing weights required for training and testing from this link:[here](https://drive.google.com/drive/folders/1IElSxuTPVTv_tv37_kk6pLqTL3gFDtNt?usp=sharing)
## Test
In`3_Test.py`，we present the model's predictions on the test dataset, including the predicted results for benign/malignant (metastatic/non-metastatic) categories and the probabilities for each category. Additionally, we have displayed the model's prediction results and generated a CSV file to save the prediction probabilities for future use in assisting doctors with the CAD demo software.

    cd code
    python 3_test
![test_result](https://github.com/user-attachments/assets/3aae7e20-6157-4c1d-a5e4-594aa5032ba2)
#### Note: Benign/Malignant (Metastatic/Non-metastatic)

### Confusion matrix
In `2_Confusion_matrix.py`, we demonstrate how to evaluate the model's performance, including metrics such as AUC, accuracy, sensitivity, specificity, false positive rate, false negative rate, and the confusion matrix. Additionally, you can use the model to compute the Youden index in `2_confusion_matrix.py` to obtain the optimal cutoff value.

    cd code
    python 2_confusion matrix

 If you want to try other settings, please refer to `2_confusion_matrix.py` for more details.
 ![test_roc](https://github.com/user-attachments/assets/47f0aaf4-11db-4893-a6de-c267bd9bf9fd)
#### Note: Benign/Malignant (Metastatic/Non-metastatic)
## CAD demo software
We also provide software to assist clinicians in the evaluation of ALNM.

Please download the software from  [here](https://drive.google.com/drive/folders/1IElSxuTPVTv_tv37_kk6pLqTL3gFDtNt?usp=sharing), and check the  `README.txt`  for usage.

After downloading, if the `diagnosis_app.exe` application is outside the folder, please move the `diagnosis_app.exe` file to the `\Diagnosis_APP\dist` directory to ensure proper execution.

Please note that this software is only used for demo, and it cannot be used for other purposes.
![cad demo software](https://github.com/user-attachments/assets/c5341069-d4c1-4e2b-9db1-77588b1a4003)

## Contact
If you encounter any problems, please open an issue without hesitation, and you can also contact us with the following:

-   email:  [cainian@gdut.edu.cn](mailto:cainian@gdut.edu.cn),  [2112203022@mail2.gdut.edu.cn](mailto:czhu@bupt.edu.cn)


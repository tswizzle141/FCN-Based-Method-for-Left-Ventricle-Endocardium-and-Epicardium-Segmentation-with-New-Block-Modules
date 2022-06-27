# FCN-Based-Method-for-Left-Ventricle-Endocardium-and-Epicardium-Segmentation-with-New-Block-Modules
#### (Specially thanks to @minhnhattrinh312 for our lovely cooperation)
## Introduction
Left ventricle cardiac segmentation on the [SunnyBrook](http://www.cardiacatlas.org/studies/sunnybrook-cardiac-data/) and [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).
## Our contributions
![Proposed Model 1](https://github.com/tswizzle141/FCN-Based-Method-for-Left-Ventricle-Endocardium-and-Epicardium-Segmentation-with-New-Block-Modules/blob/main/1.jpg)
![Proposed Model 2](https://github.com/tswizzle141/FCN-Based-Method-for-Left-Ventricle-Endocardium-and-Epicardium-Segmentation-with-New-Block-Modules/blob/main/2.jpg)
* One primary obstacle with the above modules is that even a modest number of $5 \times 5$ and $7 \times 7$ convolutions could be prohibitively expensive on top of a convolutional layer with a large number of filters. A $1 \times 1$ convolutional layer is judiciously exploited to overcome this dispute, as it could offer a channel-wise pooling, also called feature map pooling or a projection layer. Such simple technique with low dimensional embeddings could be utilized in dimensionality reduction whilst retaining salient features; also, generating a one-to-one projection of stack of feature maps to pool features across channels after conventional pooling layers.
* We adopt dilated convolutions at the asymmetric convolution layers as it supports RF exponential expansion without loss of resolution, while the parameters number increases linearly, to control the features' eccentricities. Between two consecutive layers, an activation function and a Mean-Variance normalization layer (MVN) are used for humbling the pixel distribution shifting right after a convolutional operation. Compare with Batch Normalization, which reduces internal covariate shift and accelerates the gradient flow through the network like MVN; MVN is still effective though being much simpler as it primarily centers and standardizes a single batch at a time. Instead of using ReLU activation; Swish is selected as the only activation function throughout the training. Overall, our proposed block modules could be briefly described as follows.
## Results
![table1](https://github.com/tswizzle141/FCN-Based-Method-for-Left-Ventricle-Endocardium-and-Epicardium-Segmentation-with-New-Block-Modules/blob/main/3.jpg)
![table2](https://github.com/tswizzle141/FCN-Based-Method-for-Left-Ventricle-Endocardium-and-Epicardium-Segmentation-with-New-Block-Modules/blob/main/4.jpg)
## Citation
If you find our work useful for your research, please cite at:
`@INPROCEEDINGS{left-ventricle-fcn,  
    author={Nham, Do-Hai-Ninh and Trinh, Minh-Nhat and Tran, Tien-Thanh and Pham, Van-Truong and Tran, Thi-Thao},  
    booktitle={2021 8th NAFOSTED Conference on Information and Computer Science (NICS)},   
    title={A modified FCN-based method for Left Ventricle endocardium and epicardium segmentation with new block modules},   
    year={2021}, 
    volume={},  number={},  
    pages={392-397},  
    doi={10.1109/NICS54270.2021.9701571}
}`


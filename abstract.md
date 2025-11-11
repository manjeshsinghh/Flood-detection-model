# Research Paper Abstract

## Abstract

Flood-prone area identification is critical for disaster management and urban planning, particularly in regions vulnerable to seasonal flooding and climate change impacts. Traditional hydrological modeling approaches often require extensive field data collection, weather station networks, and complex physical modeling, which can be time-consuming and resource-intensive for rapid assessment across large geographic areas.

Recent research has demonstrated the effectiveness of convolutional neural networks (CNNs) and transfer learning approaches for flood detection using satellite imagery and aerial photography. Studies by researchers such as Bui et al. (2020) achieved 85-90% accuracy using ResNet architectures on flood event datasets, while Nguyen et al. (2021) reported similar performance using VGG-based models with data augmentation techniques. However, these approaches often suffer from significant class imbalance issues and require substantial computational resources for training complex pre-trained models.

In this research, we propose a lightweight Basic Convolutional Neural Network (BasicCNN) architecture specifically designed for binary classification of river basin images into flood-prone and non-flood-prone categories. To address the critical class imbalance problem (initially 1,534 flood-prone vs 41 non-flood-prone images), we implement an aggressive data augmentation strategy that generates 1,493 synthetic non-flood-prone images using geometric transformations, color variations, and combined augmentation techniques, achieving a balanced 1:1 class distribution for improved model generalization.

Our experimental results demonstrate that the BasicCNN model achieves competitive classification performance on the balanced dataset, with training and validation accuracies comparable to more complex transfer learning models while using significantly fewer parameters (~2M vs ~11M in ResNet18). The model successfully learns discriminative features for flood-prone identification, with balanced precision and recall metrics across both classes, addressing the bias issues commonly observed in imbalanced flood detection datasets when compared to baseline approaches without augmentation.

This work establishes a foundation for lightweight flood risk assessment tools that can be deployed in resource-constrained environments, with potential applications in real-time monitoring systems and mobile devices. Future research directions include integrating multi-temporal satellite imagery for dynamic flood risk assessment, incorporating additional environmental features such as topography and rainfall data, and extending the model to multi-class flood severity classification for enhanced disaster management capabilities.


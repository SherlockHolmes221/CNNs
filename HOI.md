# HICO: A Benchmark for Recognizing Human-Object Interactions in Images

###  key features of HICO:
- a diverse set of interactions with common object categories, an average of 6.5 distinct interactions per object category 
- a list of well-defined, sense-based HOI categories
- an exhaustive labeling of co-occurring interactions with an object category in each image.

### What is HOI and Why it difficult
- distinguish a variety of different interactions with the same object category
- it is critical to understand what the person is doing to the object
- categories in the long tail that have very few training images

### HICO
- 47,774 images
- 600 categories of human-object interactions 
- 117 common actions
- 80 common objects 
- In our dataset all HOI categories have at least 6 images, so all categories have at least 1 training images. 

### Evaluation 
- Given an image, an approach being evaluated outputs a classification score for each of the HOI categories. 
  
  Then we compute the average precision (AP) for each HOI category by ranking the test images by the classification scores. 
  
  The average of AP for all HOI categories gives the mAP. 
- Known Object (KO) setting

  use the verified positives as positives, skip the ambiguous image and the “Unknown” images, and use the verified negatives as negatives

### HOCNN
- Given an input image, we first run object detectors and a pose estimator, which together generate a set of heatmaps, one per object category and one per human body part. 

###  Conclusions
- adding cooccurrence knowledge consistently improves performance. 
- Adding the V classifiers leads to moderate but consistent improvement for overall mAP as well as mAP for rare classes 
- DNNs enjoy a significant edge. 
- Semantic knowledge can significant improve HOI recognition, especially for uncommon categories.


# Learning to Detect Human-Object Interactions
predicting a human and an object bounding box with an interaction class label that connects them。
###  HICO-DET 
- offers more than 150K annotated instances of human-object pairs
- spanning the 600 HOI categories in HICO
- an average of 250 instances per HOI category
- multiple people interacting with one object
- one person interacting with multiple objects
- At least one training instance for each of the 600 HOI classes. 

### HO-RCNN

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/HO-RCNN.png)
 At the core of HO-RCNN is the Interaction Pattern, a novel DNN input that characterizes the spatial relations between two bounding boxes
- generate proposals of human-object region pairs using state-of-the-art human and object detectors
- each human-object proposal is passed into a ConvNet to generate HOI classification scores
- To extend to mulitple HOI classes, we train one binary classifier for each HOI class at the last layer of each stream
- The human stream extracts local features from the detected humans
- The object stream extracts local features from the detected objects
- The pairwise stream extracts features which encode pairwise spatial relations between the detected human and object
- The Interaction Patterns should be invariant to any joint translations of the bounding box pair
  
### Evaluation
- min(IOUh,IOUO) > 0.5. 
- (a) all 600 HOI categories in HICO(Full),
- (b) 138 HOI categories with less than 10 training instances (Rare), and 
- (c) 462 HOI categories with 10 or more training instances(Non-Rare). 
-  Known Object setting and Default setting

# Visual Semantic Role Labeling
 given an image we want to detect people doing actions and localize the objects of interaction
 
- annotate a dataset of 16K people instances in 10K images with actions they are doing and associate objects in the
 scene with different semantic roles for each action
- provide a set of baseline algorithms for this task and analyze error modes providing directions for future work

### V-COCO
- The VCOCO dataset contains a total of 10346 images containg 16199 people instances. Each annotated person has binary labels for 26 different actions.

### Methods
- Agent detection:

  Our model for agent detection starts by detecting people, and then classifies the detected people into different action categories
- Regression to bounding box for the role 

  Our first attempt to localize the object in semantic roles associated with an action involves training a regression model to regress to the location of the semantic role
- Using Object Detectors 
   
   Our second method for localizing these objects uses object detectors for the categories that can be a part of the semantic role.
   
   We start with the detected agent(using model A above)and for each detected agent attach the highest scoring box according to the following score function

# No-Frills Human-Object Interaction Detection: Factorization,Layout Encodings,and Training Techniques
factorization, direct encoding and scoring of layout,and improved training techniques
- In the first stage, object category specific bounding box candidates selected using a pre-trained object detector such as FasterRCNN 
- In the second stage, a factored model is used to score and rank candidate box-pairs (bh,bo) ∈ Bh × Bo for each HOI category

![](https://github.com/SherlockHolmes221/CNNs/raw/master/img/no-frills-p.png)

### Interaction Term
- Appearance
  The appearance of a box in an image is encoded using Faster-RCNN (Resnet-152 backbone) average pooled fc7 features extracted from the RoI
- Box Configuration
   We encode the absolute position and scale of both the human and object boxes using box width, height, center position, aspect ratio, and area. 
   
   We also encode relative conﬁguration of the human and object boxes using relative position of their centers, ratio of box areas and their intersection over union. 
- Human Pose
  We supplement the coarse layout encoded by bounding boxes with more ﬁne-grained layout information provided by human pose keypoints.
  
 
### training techniques 
- Eliminating train-inference mismatch
  fixes this mismatch by directly optimizing the combined scores using a multi-label HOI classification loss.
- Rejecting easy negatives using indicator terms
  If either b1 is not a “human” candidate(category h) or b2 is not an object candidate o, 
  then the factor model should predict a 0 probability of (b1,b2) belonging to HOI category (h,o,i) for any interactions i
- Training with large negative to positive ratio
  Higher ratios compared to object detector training are expected since the number of negative pairs is quadratic 
  in the number of object proposals as opposed to being linear for object detectors 

### Comparison to State-of-the-art 
- Appearance does not need to be relearned. 
  We only use ROI pooled features from Faster-RCNN pretrained on MS-COCO.
- Layout is directly encoded and scored
- Weight sharing for learning efficiency
- ROI pooling for context


### Questions
- How do they deal with human or object invisible? 

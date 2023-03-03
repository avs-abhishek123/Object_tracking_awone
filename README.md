# Problem Statement -

Create a pipeline to identify key points for a quadrilateral-shaped green card even if there is occlusion with other objects/cards of different color/ cards of green color.

<br>

## Articles & Research

The research papers that I will be going through is being added to this - [Link]()

## Implementations

| <h3>Methods implemented</h3> | <h3> Link </h3> |
| --- | --- |
| MeanShift | [MeanShift](https://github.com/avs-abhishek123/Object_tracking_awone/tree/main/code/mean_shift) |
|OpticalFlow| [OpticalFlow](https://github.com/avs-abhishek123/Object_tracking_awone/tree/main/code/OpticalFlow) |
|SIFTDetector + Optical Flow| [SIFT_Detector_OpticalFLow](https://github.com/avs-abhishek123/Object_tracking_awone/blob/main/code/OpticalFlow/OpticalFlow_4th_attempt.py) |
| FASTFeatureDetector + Optical Flow | [FASTFeatureDetector_OpticalFlow](https://github.com/avs-abhishek123/Object_tracking_awone/blob/main/code/OpticalFlow/OpticalFlow_6th_attempt_W.py) |
| FASTFeatureDetector + Farneback Optical Flow | [FASTFeatureDetector_Farneback_OpticalFlow](https://github.com/avs-abhishek123/Object_tracking_awone/blob/main/code/OpticalFlow/OpticalFlow_7th_attempt.py) |
| Kalman_Filter | [Kalman_Filter](https://github.com/avs-abhishek123/Object_tracking_awone/tree/main/code/Kalman_Filter) |
| tracking_by_detection |[tracking_by_detection](https://github.com/avs-abhishek123/Object_tracking_awone/tree/main/code/tracking_by_detection) |
| Haar Cascade Classifier + MOSSE tracker | [HAAR_MOSSE](https://github.com/avs-abhishek123/Object_tracking_awone/blob/main/code/tracking_by_detection/Attempt_1_HaarCascadeClassifier_MOSSE_tracker.py) |
| Haar Cascade Classifier + TrackerBoosting | [HAAR_TrackerBoosting]() |
| Haar Cascade Classifier + TrackerKCF | [HAAR_TrackerKCF]() |
| Haar Cascade Classifier + TrackerKCF + selectROI | [HAAR_TrackerKCF_selectROI]() |
| ROLO | [ROLO]() |
| YOLO with DeepSort | [YOLO_DeepSort](https://github.com/avs-abhishek123/Object_tracking_awone/tree/main/code/YOLO_DeepSort) |
|  | []() |
|  | []() |
|  | []() |

<br>
<hr>

## Challenges in Object Tracking

- #### Challenge-1: Occlusion problem
    - The occlusion of objects in videos is one of the most common obstacles to the seamless tracking of objects. In the below figure (left), the man in the background is detected, while the same guy goes undetected in the next frame (right).  Now, the challenge for the tracker lies in identifying the same guy when he is detected in a much later frame and associating his older track and features with his trajectory.
    - Remedy -
- #### Challenge-2: Variations in viewpoints

    - Often in tracking, the objective will be to track an object across different cameras. As a consequence of this, there will be significant changes in how we view the object. In such cases the features used to track an object become very important as we need to make sure they are invariant to the changes in views. 
    - Remedy -

- #### Challenge-3: Non-stationary camera 
    - When the camera used for tracking a particular object is also in motion with respect to the object, it can often lead to unintended consequences. Many trackers consider the features of an object to track them. Such a tracker might fail in scenarios where the object appears different because of the camera motion (appear bigger or smaller). A robust tracker for this problem can be very helpful in important applications like object tracking drones, and autonomous navigation.
    - Remedy -

- #### Challenge-4: Annotating training data
    - Getting good training data for a particular scenario is a challenge. Unlike building a dataset for an object detector, where randomly unconnected images where the object is seen can be annotated, we require video sequences where each instance of the object is identified throughout, for each frame.
    - Remedy - 


<hr>

In progress


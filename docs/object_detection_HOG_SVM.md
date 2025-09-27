 # Object Detection using HOG and SVM
 
 The core idea of object detector is simply applying the classifier to different 
 image patches of the image.
 
 A few things to keep in mind while thinking about object detection.
  - Location: The object being detected can appear at any location in an image.
  - Scale: The scale or the size of the object can be arbitrary.
 In the Dalal and Triggs paper, the pedestrian detector was trained for an
 image patch of size 64 x 128. However, a pedestrian can appear in an image
 in any size. It is also required to decide the smallest size of the object to be detected.
  - Rotation: The object may not be upright. It may be need to be rotated.
  - Non-Maximum Supression: Object detection often requires a post
 processing step called non-maximum supression. If an object is detected at a
 particular pixel location, there is a chance it will be detected again at a
 nearby pixel location. Both these bounding boxes point to the same object
 and therefore it is required to reject the overlapping bounding boxes.
 This is called non-maximum supression.

 ## Scale Space Search
 Detecting objects at multiple scales is equivalent to resizing the image and
 using the fixed sized detector on the resized image. So, object detectors
 create an image pyramid internally. 
 
 The image pyramid needs two parameters.
  - Levels: The number of levels in the pyramid. A typical number is 64.
  - Scale: By what percentage is the image resized for the next level.
 A typical number is 1.05.

 In most implementations of an object detector, the image is downscaled
 and never upscaled. If the object in expected to be tiny in the target image,
 the user should upscale the image before sending it to the object detector 
 and resize the rectangles obtained.

 ## Location search
 Perform a sliding window search for the location of the object in
 every image in the pyramid. The object is assumed to be the same size it was
 trained on, because the detector will find the object at the specified size
 in at least one level of the pyramid. Different sized objects will be detected
 in different scaled images, which means bigger objects in the images are
 detected at a lower resolution.

 The number of search locations is proportional to the square of the scale used.
 For example, if a user is searching for 10,000 locations at scale 1,
 they need to search for only 2500 locations in scale 0.5 and
 only 625 locations at scale 0.25.

 If the object is large, significant computational effort is spent searching
 at larger scales. Significant speedups can therefore be obtained by correctly
 resizing the image when you have an idea about the size of the object in the
 image and the size at which the object detector was trained.

 If user is building a webcam or selfie application that uses face detection,
 they can significantly improve speed by resizing the image to the appropriate size.

 ## Classifying a patch
 In the previous section, it is mentioned that many patches of the image
 at many locations and scale are evaluated to check if there is an object
 inside the patch. The evaluation is done using the classifier that is able to
 classify a patch into object vs. background.
 
 When an SVM is used as a classifier, the two classes are labeled as
 -1 (background) and 1 (object).
 When the response for the SVM classifier is greater than 0, the patch has a
 greater probability of belonging to the object class.
 
 In practice, if the SVM classifier response is greater than a threshold
 (called the hitTreshold in OpenCV) we say it is an object.
 A high hitTreshold would result in fewer detected objects.

 ## Grouping Rectangles (Non-maximum Suppression)
 As mentioned in the previous section, each bounding box has an associated SVM response.
 These responses are often called weights. The greater the weights,
 the more likely it is that the box contains an object.

 The goal of non-maximum suppression is to come up with a smaller list of
 rectangles and a weight associated with every rectangle in the final list.

 There are many different ways of doing non-maximum suppression.
 In OpenCV, the following steps are employed by the `HOG::groupRectangles` method internally.
  1. Clustering: Rectangles are clustered based on overlap between rectangles.
 The idea is to represent an entire cluster with just one rectangle.
  1. Pruning: If the number of rectangles in a cluster are less than a threshold,
 the cluster is removed.
  1. Cluster average: The average of all rectangles in every cluster is found.
 This rectangle is added to the final list.
  1. Pick maximum SVM response: The SVM response (weight) associated with this
 average rectangle is the maximum of the weights of all rectangles in the cluster.
  1. Filtering smaller rectangles: The final list is further pruned by removing
 smaller rectangles inside larger ones if the number of rectangles in
 the original cluster was less than 3.

 From the above description, it is evident that not all rectangles completely
 contained inside another rectangle are moved by OpenCV’s non-maximum suppression.

# Object Tracking Algorithms

Object tracking is an important and practical topic that has a very long history and has been applied to numerous fields. At a high level, tracking refers to estimating the state of an object (e.g., position and velocity) based on sensor measurements, and predicting its future location. For example, tracking the location of an aircraft based on radar measurements is a typical application of tracking. In the context of computer vision, tracking typically refers to processing video frames and predicting the location of an object (or multiple objects), in future video frames.

In tracking, the goal is to find an object in the current frame given the object has been tracked successfully in all (or nearly all) previous frames. Since it has been tracked up until the current frame, it is apparent that the object is moving and hence, the objeect has a motion model. The motion model will estimate the position and velocity of an object and use that information to predict the location of an object in future video frames.

Even if the motion model is available, that is not the only information available about the object, the appareance of the object is also available. The appearance model encodes what the object looks like and then searches the region around the predicted location from the motion model to then fine-tune the location of the object.

The motion model predicts the approximate location of the object. The appearance model fine tunes this estimate to provide a more accurate estimate based on appearance.

If the object was very simple and did not change it’s appearance much, a simple template as an appearance model can be used to track objects. However, real life is not that simple. The appearance of an object can change dramatically. To tackle this problem, in many modern trackers, this appearance model is a classifier that is trained in an online manner.

Classifier classifies a rectangular region of an image as either an object or background. The classifier takes in an image patch as input and returns a score between 0 and 1 to indicate the probability that the image patch contains the object. The score is 0 when it is absolutely sure the image patch is the background and 1 when it is absolutely sure the patch is the object.

It usually takes hours to train a classifier which is not available to the user during tracking. In machine learning, the word "online" refers to an algorithms that is trained on the fly at run time. An offline classifier may need thousands of examples to train a classifier, but an online classifier is typically trained using a very few examples at run time.

A classifier is trained by feeding it positive (object) and negative (background) examples. For example, to build a cat classifier, it is important to train it with thousands of images containing cats and thousands of images that do not contain cats. This way the classifier learns to differentiate what is a cat and what is not.

## BOOSTING Tracker

This tracker is based on an online version of AdaBoost — the algorithm that the HAAR cascade based face detector uses internally. This classifier needs to be trained at runtime with positive and negative examples of the object. The initial bounding box supplied by the user (or by another object detection algorithm) is taken as the positive example for the object, and many image patches outside the bounding box are treated as the background. Given a new frame, the classifier is run on every pixel in the neighborhood of the previous location and the score of the classifier is recorded. The new location of the object is the one where the score is maximum. As more frames come in, the classifier is updated with this additional data.

Pros : None. This algorithm is a decade old and works ok, but advances trackers (like MIL, KCF) based on similar principles are available.

Cons : Tracking performance is mediocre. It does not reliably know when tracking has failed.

## MIL Tracker

This tracker is similar in idea to the BOOSTING tracker described above. The big difference is that instead of considering only the current location of the object as a positive example, it looks in a small neighborhood around the current location to generate several potential positive examples.

It may seem like this is a bad idea because the object is not centered in most of these "positive" examples but this is where Multiple Instance Learning (MIL) comes to rescue. In MIL, you do not specify positive and negative examples, but positive and negative "bags". The bag is labeled as positive if any of the instance in the bag is labeled as positive by the classifier. Otherwise the bag is labeled as negative.

Pros : The performance is pretty good. It does not drift as much as the BOOSTING tracker and it does a reasonable job under partial occlusion.

Cons : Tracking failure is not reported reliably. Does not recover from full occlusion.

## KCF Tracker

KCF stands for Kernelized Correlation Filters. This tracker builds on the ideas presented in the previous two trackers. This tracker utilizes the fact that the multiple positive samples used in the MIL tracker have large overlapping regions. This overlapping data leads to some nice mathematical properties that is exploited by this tracker to make tracking faster and more accurate at the same time.

**NOTE :** KCF works well with a tight bounding box.

Pros: Accuracy and speed are both better than MIL and it reports tracking failure better than BOOSTING and MIL.

Cons : Does not recover from full occlusion.

## TLD Tracker

TLD stands for Tracking, learning and detection. As the name suggests, this tracker decomposes the long term tracking task into three components — (short term) tracking, learning, and detection.

From the author’s paper, "The tracker follows the object from frame to frame. The detector localizes all appearances that have been observed so far and corrects the tracker if necessary. The learning estimates detector’s errors and updates it to avoid these errors in the future." This output of this tracker tends to jump around a bit. For example, if you are tracking a pedestrian and there are other pedestrians in the scene, this tracker can sometimes temporarily track a different pedestrian than the one you intended to track. On the positive side, this track appears to track an object over a larger scale, motion, and occlusion. If you have a video sequence where the object is hidden behind another object, this tracker may be a good choice.

Pros : Works the best under occlusion over multiple frames. Also, tracks best over scale changes.

Cons : Lots of false positives making it almost unusable.

## MEDIANFLOW Tracker

Internally, this tracker tracks the object in both forward and backward directions in time and measures the discrepancies between these two trajectories. Minimizing this "ForwardBackward" error enables them to reliably detect tracking failures and select reliable trajectories in video sequences.

This tracker works best when the motion is predictable and small. Unlike, other trackers that keep going even when the tracking has clearly failed, this tracker knows when the tracking has failed.

Pros : Excellent tracking failure reporting. Works very well when the motion is predictable and there is no occlusion.

Cons : Fails under large motion.

## GOTURN tracker

Out of all the tracking algorithms in the tracker class, this is the only one based on Convolutional Neural Network (CNN). It is also the only one that uses an offline trained model, because of which it is faster that other trackers. OpenCV documentation mentions that it is "robust to viewpoint changes, lighting changes, and deformations", but it does not handle occlusion very well.

**NOTE :** GOTURN being a CNN based tracker, uses a caffe model for tracking. The Caffe model and the prototxt file must be present in the directory in which the code is present.

## MOSSE tracker

The idea of using correlation filters for tracking is very old. However, the concept of simply using an image patch around the detected object to try finding its location in the next frame using correlation does not produce good results. This is because the image patch appearance may change quite a bit.

Minimum Output Sum of Squared Error (MOSSE) uses discriminative correlation filter (DCF) for object tracking which produces stable correlation filters when initialized using a single frame. When the paper was published in 2010, it surprised the community because of it simplicity. It was an old idea that was modified slightly, and was able to outperform other algorithms that used heavy duty classifiers, complex appearance models, and stochastic search techniques. It was also substantially faster.

MOSSE tracker is robust to variations in lighting, scale, pose, and non-rigid deformations. It also detects occlusion based upon the peak-to-sidelobe ratio, which enables the tracker to pause and resume where it left off when the object reappears. MOSSE tracker also operates at a higher fps (450 fps and even more).

## CSRT tracker

The CRST tracker extends the Discriminative Correlation Filter (DCF) idea in MOSSE with what the authors call Channel and Spatial Reliability (DCF-CSR). In particular, they are able to extend the search region over while the search is performed. This ensures enlarging and localization of the selected region and improved tracking of the non-rectangular regions or objects. It uses only 2 standard features (HoGs and Colornames). It also operates at a comparatively lower fps (25 fps) but gives higher accuracy for object tracking.

## Observations

- MIL and Boosting are slow and suffer from occlusion. CSRT is a better tracking in this situation.
- Medianflow handles scale very well. It is also fast but fails under occlusion.
- MOSSE and KCF are fast but fail unexpectedly fail sometimes.
- CSRT is slow but stable and accurate.
- TLD: The detection is very hepful to recover from occlusion and hence the concept is very good but the false positives are very high and not very useful in real-life applications.
- If the objects moves upredictably, Correlation-based trackers(KCF, MOSSE, CSRT) might drift and eventually fail.
- Discriminative Trackers(MIL, BOOSTING, TLD) use machine learning based models to adapt and recover but they are still prone to failure in case of occlusion or fast motion.
- Kalman Filter-based trackers(e.g., used in conjunction with DeepSORT or SORT) have thw ability to estimate the new position of the object and recover when it reappears if the model is will tuned.

## Verdict

- Use CSRT when high accuracy and stability are required but the speed is sufficient for the application.
- Use KCF if the application needs speedy results and will not need to deal with large occlusion or scale changes.
- Use MOSSE if the application needs speedy results, and they don't have to be accurate.
- Use MedianFlow if the application will need to deal with linear motion and large variations in scale, without much occlusion.

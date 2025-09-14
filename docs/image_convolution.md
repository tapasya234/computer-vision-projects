# Image Filtering and Convolution
Filtering and convolution are usually the first steps in any CV applications. 
the basis of all linear filters. It is the very first step applied when we want to blur, sharpen or detect edges. 

### Signal Processing Jargon
While reading Computer Vision and Machine Learning literature, you will see signal processing jargon that can be intimidating at times. Let us demystify a few of those terms here.

Image Patch: An image patch is simply a small (3x3, 5x5 … ) region of the image centered around a pixel.

Low Frequency Information : An image patch is said to have low frequency information if it is smooth and does not have a lot of texture.

High Frequency Information : An image patch is said to have high frequency information if it has a lot of texture (edges, corners etc.).

Low Pass Filtering : This is essentially image blurring / smoothing. It you blur an image, you smooth out the texture. As the name suggests, low pass filtering lets lower frequency information pass and blocks higher frequency information.

High Pass Filtering : This is essentially a sharpening and edge enhancement type of operation. As the name suggests, low frequency information is suppressed and high frequency information is preserved in high pass filtering.

## Filtering
Image Filtering is a broad term applied to a variety of Image Processing techniques that enhance an image by eliminating unwanted characteristics (e.g. noise) and/or improving desired characteristics (e.g. better contrast). Blurring, edge detection, edge enhancement, and noise removal are all examples of image filtering.

Image filtering is a neighborhood (or local) operation. This means that the pixel value at location  (x,y)  in the output image depends on the Image patch. 

### Linear Filter
When the output pixel depends only on a linear combination of input pixels, we call the filter a Linear Filter. Linear filters are efficiently implemented using a convolution operation.

### Nonlinear Filter
When the output pixel depends only on nonlinear combination of input pixels, we call the filter a Nonlinear Filter. 

## Convolution
A convolution operation requires two inputs, the input image and a kernel. The input image can be either grayscale or color. And yes, it is the same convolution used in state-of-the-art neural networks.

For color images, convolution is performed independently on each channel. A convolution kernel is a 2D matrix that is used to filter images, also known as a convolution matrix. 
It is typically a square, MxN matrix, where both M and N are odd integers (e.g., 3×3, 5×5, 7×7, ...). The 3x3 kernel below is referd to as a Box kernel which is used for Box Bluring.

        [
            1  1  1
    1/9     1  1  1
            1  1  1
        ]

In most cases, we want the sum of the elements in a kernel to sum to one. This ensures that the output image has the same brightness level as the input image. 
If we do not do this, the output pixels will be approximately 9 times (3x3 = 9) brighter than the input pixels on average. 
There are no restrictions on entries of the kernel. They can be positive, negative, integers or floating-point numbers.

The output is a filtered image. Filtering of a source image is achieved by convolving the kernel with the image. 
In simple terms, convolution of an image with a kernel represents a simple mathematical operation, between the kernel and its corresponding elements in the image.

#### Convolution Workflow
1. Assume that the center of the kernel is positioned over a specific pixel (p), in an image.
1. Multiply the value of each element in the kernel with the corresponding pixel element (i.e., its pixel intensity) in the source image.
1. Add the result of those multiplications and compute the average.
1. Replace the value of pixel (p), with the calculated average value.

In OpenCV, convolution is performed using the function `filter2D()`

Because convoling involves placing the kernel over each image pixel, it begs the question, but what happens at the boundary? At the boundary, the convolution is not uniquely defined. There are a few options we can choose from.
1. Ignore the boundary pixels : If we discount the boundary pixels, the output image will be slightly smaller than the input image.
1. Zero padding : We can pad the input image with zeros at the boundary pixels to make it larger and then perform convolution.
1. Replicate border : The other option is to replicate the boundary pixels of the input image and then perform the convolution operation on this larger image.
1. Reflect border : The preferred option is to reflect the border about the boundary. Reflecting ensures a smooth intensity transition of pixels at the boundary.

By default OpenCV uses border type `BORDER_REFLECT_101` which is the same as option 4.

###### Note:
The convolution filtering described above is actually called correlation filtering. Correlation and convolution are exactly the same operation with one difference. 
In convolution, the kernel is rotated 180 degrees before doing the correlation operation. When the kernel is symmetric, correlation and convolution are the same.

## Sharpening
Convolution can be used for shapening an image. A 3x3 kerner for sharpening image is defined as below. We can sharpen an image with the following 2D-convolution kernel.

        [
            0  -1  0
            -1  5  -1
            0  -1  0
        ]

## Bluring
### Box Blur
Uses a box kernel where all the pixels are the same and the their sum equals to 1, a normalized kernel. The box kernel weights the contribution of all pixels in the neighborhood equally.
A larger kernel will blur the image more than a smaller kernel.

        [
            1  1  1
    1/9     1  1  1
            1  1  1
        ]

While a Box Blur can be done using `filter2D()` in OpenCV, there is a much simpler function to use, `blur()`.

### Gaussian Blur
Unlike the box kernel, the Gaussian kernel is not uniform. The middle pixel gets the maximum weight while the pixels farther away are given less weight. 
A Gaussian Blur kernel weights the contribution of a neighboring pixel based on a Gaussian distribution. 

The Gaussian 5x5 kernel shown below is an approximation to the 2D Gaussian distribution with  `σ = 1`. A large value of `σ` would add more weight to the edges of the kernel and therefore have a stronger blurring effect, 
while a smaller value would narrow the blurring effect. The size of the kernel also determines the amount of blurring. A larger kernel (with the same value of sigma) will blur the image more than a smaller kernel. 

            [
                1   4   7   4   1
                4   16  26  16  4
      1/273     7   26  41  26  7
                4   16  26  16  4
                1   4   7   4   1
            ]

An image blurred using the Gaussian kernel looks less blurry compared to a box kernel of the same size. 

A small amount of Gaussian blurring is frequently used 
to remove noise from an image. It is also applied to the image prior to noise-sensitive image filtering operations. 
For example, the Sobel kernel used for calculating the derivative of an image is a combination of a Gaussian kernel and a finite difference kernel.

In OpenCV, Gaussian Blur is performed using the function `GaussianBlur()`

### Median Blur
Median blur filtering is a nonlinear filtering technique that is most commonly used to remove salt-and-pepper noise from images. As the name suggests, salt-and-pepper noise shows up as randomly occurring white and black pixels that are sharply different from the surrounding.

For an image, the median blurring filter replaces the value of the central pixel with the median of all the pixels within the kernel area.

In OpenCV, Median Blur is performed using the function `medianBlur()`

### Bilateral Blur
Bilateral Filter is a non-linear, edge preserving and noise-reducing smoothing filter.
In edge-preserving filters, there are two competing objectives :
1. Smooth the image.
1. Don’t smooth the edges / color boundaries.

In other words, if we want to preserve edges, we cannot simply replace the color of a pixel by the weighted sum of its neighbors. 
It uses a gaussian filter but one more Gaussian filter which is a function of pixel difference.

Bilateral filter is done by perform both spatial and colour filtering, which is why it is a combination of Gaussian and Median filtering. 

Spatial filtering - It is a weighted filtering, where the weight of the pixel being considered is dependent on the distance between pixel and the central pixel.
Colour filter - It makes sure that only pixels with similar intensities to the central pixel are considered for the filter. Which is why, it is good at preserving edges.

In OpenCV, Bilateral Blur is performed using the function `bilateralFilter()`

### Which Blur should be used?
The following should be noted while deciding which filter to use for your problem :
- Median filtering is the best way to smooth images which have salt-pepper type of noise (sudden high / low values in the neighborhood of a pixel).
- Gaussian filtering can be used if there is low Gaussian noise.
- Bilateral Filtering should be used if there is high level of Gaussian noise, and you want the edges intact while blurring other areas.
- In terms of execution speed, Gaussian filtering is the fastest and Bilateral filtering is the slowest.

## Edge Detection
Edge detection is an image-processing technique used to identify the boundaries (edges) of objects, or regions within an image. Edges are among the most important features associated with images. We come to know of the underlying structure of an image through its edges. Computer vision processing pipelines therefore extensively use edge detection in applications.

Edge detection is typically performed on blurred, grayscale images for the following reasons:
1. GrayScale: The edge detection is done by calculating the intesity differences of neighbouring pixels and grayscale images are sufficient to do that. That doesn't mean, Edge Detection can't be performed on Colour images but it is definitely more expensive and requires a special form of Edge Detection. 
1. Blurring: Blurring is performed to reduce the noise in the image. In edge detection, numerical derivatives of the pixel intensities are computed, and this typically results in ‘noisy’ edges. In other words, the intensity of neighboring pixels in an image (especially near edges) can fluctuate quite a bit, giving rise to edges that don’t represent the predominant edge structure we are typically looking for. Blurring smooths the intensity variation near the edges, making it easier to identify the edge structure within the image.

### Sobel
Sobel Edge Detection is one of the most widely used algorithms for edge detection. The Sobel Operator detects edges that are marked by sudden changes in pixel intensity. A sudden change in the derivative of intensity function will reveal a change in the pixel intensity as well. With this in mind, we can approximate the derivative, using a 3×3 kernel. Sobel Filters perform Gaussian smoothing implicitly.
We use one kernel to detect sudden changes in pixel intensity in the X direction, and another in the Y direction.

X-direction Kernel:

        [
            -1  0  -1
            -2  0  -2
            -1  0  -1
        ]

Y-direction Kernel:

        [
            -1  -2  -1
             0   0   0
            -1  -2  -1
        ]

If we use only the Vertical Kernel, the convolution yields a Sobel image, with edges enhanced in the X-direction
Using the Horizontal Kernel yields a Sobel image, with edges enhanced in the Y-direction.

Sobel Edge Detection algorithm can be performed using the above kernels with `filter2D()` as well as `Sobel()` which does the same things with simplier parameters which makes it more convienent to use. 

### Canny
Canny Edge Detection is one of the most popular edge-detection methods in use today because it is so robust and flexible. It’s designed to detect clean, well-defined edges while reducing noise and avoiding false edges.

The algorithm itself follows a three-stage process for extracting edges from an image. Blurring is almost always added as a pre-processing step to reduce noise. This makes it a four-stage process, which includes:
1. Pre-processing step: Noise Reduction (blurring)
1. Calculating the intensity gradient of the image (using a Sobel kernel)
1. Non-maximum Suppression - Supresses pixles if the adjacent pixels have higher intensities.
1. Hysteresis thresholding - Uses the temporary edge map to figure out the valid edges.

In OpenCV, Canny Edge Detection can be performed using the `Canny()` which has three required parameters: 
1. src : Input image.
1. threshold1: First threshold for the hysteresis procedure.
1. threshold2: Second threshold for the hysteresis procedure.

The thresholds described above are interchangeable as arguments in the function call. Edges with intensity gradients greater than the higher threshold are considered "Sure Edges." Edges with intensity gardients less than the high threshold, but greater than the lower threshold are considered candidate edges. If the candidate edges are connected to a "Sure Edge" then they become a valid edge and are included in the final edge map. All other edges whose intensity gradients are less than the high threshold value are discarded.

### External Links
[Kernels for Image Processing - Wikipedia](https://en.wikipedia.org/wiki/Kernel_%28image_processing%29#)

[Visual explanation for Image Kernels](https://setosa.io/ev/image-kernels/)
# Colour Spaces
In simple terms, a colour space is a specific organization of colours that typically represents the space of all possible human-perceivable colours. 

A colour model is a mathematical construct for how to specify colours in the colour space with a unique tuple of numbers (typically as three or four values representing the relative contributions of colour components). A colour model can be thought of as a mathematical way to navigate a colour space. 

However, it is very common to use the term “colour space” to collectively define both a colour model along with a specific mapping of that model onto an absolute colour space.

Three bands of lights that are easily recognised by the retina:
 1. Red or Long wave-length: 564-580 nm
 1. Green or Medium wave-length: 534-545 nm
 1. Blue or Short wave-length: 420-440 nm

As an introduction to colour spaces we will consider two commonly used models: the RGB colour space (for Red, Green, Blue) and the HSV colour space (for Hue, Saturation, Value). Both colour spaces use a three-dimensional coordinate system to specify the component colours that represent a unique tuple, and therefore a unique colour. These components are also referred to as colour channels. Since colour images are typically represented by three colour channels as 8-bit unsigned integers for each channel, the individual colour components can take on values from [0,255]. So we can therefore represent 16.77 Million unique colours in either colour space (256 256 256).

### Jargon
Chominance - Colour component
Luminence - Lightness or brightness component 

## Additive and Subtractive colors

Additive color mixing: projecting primary color lights on a white surface shows secondary colors where two overlap; the combination of all three primaries in equal intensities makes white.
To form a color with RGB, three light beams (one red, one green, and one blue) must be superimposed (for example by emission from a black screen or by reflection from a white screen). Each of the three beams is called a component of that color, and each of them can have an arbitrary intensity, from fully off to fully on, in the mixture.

The RGB color model is additive in the sense that if light beams of differing color (frequency) are superposed in space their light spectra adds up, wavelength for wavelength, to make up a resulting, total spectrum.[5][6] This is in contrast to the subtractive color model, particularly the CMY Color Model, which applies to paints, inks, dyes and other substances whose color depends on reflecting certain components (frequencies) of the light under which they are seen.

In the additive model, if the resulting spectrum, e.g. of superposing three colors, is flat, white color is perceived by the human eye upon direct incidence on the retina. This is in stark contrast to the subtractive model, where the perceived resulting spectrum is what reflecting surfaces, such as dyed surfaces, emit. A dye filters out all colors but its own; two blended dyes filter out all colors but the common color component between them, e.g. green as the common component between yellow and cyan, red as the common component between magenta and yellow, and blue-violet as the common component between magenta and cyan. There is no common color component among magenta, cyan and yellow, thus rendering a spectrum of zero intensity: black.

## Why do we need different color spaces?
Different color spaces have been designed to cater to different applications like object segmentation, transmission, displaying, printing etc. Some properties of color spaces are :

 - Device Dependency - The color is dependent on the device producing it (camera) and the device displaying it (Monitor). Most color spaces are device dependent except for Lab color space, which is designed to be device independent. Please read this article to gain more insight into this.

 - Intuitiveness - It is more intuitive to be able to specify a color as “orange” (which we can do in HSV color space) instead of calling it a mixture of red and green (which we have to do in RGB color space).

 - Perceptual uniformity - A small change in the value of a particular attribute should result in a proportional change in perception of that attribute. The Lab color space is designed to have perceptual uniformity. The RGB color space is highly non-uniform in this aspect.

## RGB
In the RGB color space, all three channels contain information about the color as well as brightness. 

## HSV 

Similar to the BGR color space, HSV contains three channels. However, instead of these channels representing how much Blue, Green, and Red contribute to a single pixel, HSV instead defines the color of a single pixel stands in terms of Hue, Saturation, and Value. Both BGR and HSV take up the same number of channels, so we can convert from one to the other with minimal impact to our image (small rounding errors may take effect).

This is one of the most popular color spaces used in image processing after the RGB color space. Its three components are:
 - Hue: indicates the color / tint of the pixel
 - Saturation: indicates the purity (or richness) of the color
 - Value: indicates the amount of brightness of the pixel

The HSV color space converts the RGB color space from cartesian coordinates (x, y, z) to cylindrical coordinates (ρ, φ, z). It is more intuitive than the RGB color space because it separates the color and brightness into different axes. This makes it easier for us to describe any color directly.

The HSV color space is cylindrical. In OpenCV, the Hue component is measured in degrees and has a range of [0, 180]. Since the HSV color space is represented in a cylindrical coordinate system, the values for Hue wrap around 180. It is represented as an angle where a hue of 0 is red, green is 120 degrees ( 60 in OpenCV ), and blue is at 240 degrees( 120 in OpenCV ). In OpenCV, Hue is goes from 0 to 180 intensities values where one grayscale intensity change represents 2 degrees.

Saturation and Value both range from [0, 255]. 

Saturation refers to how pure the color is. Pure red has high saturation. Different shades of a color correspond to different saturation levels. With the Hue and Saturation channels known, we have a better idea about the color and tints or shades of color in our image.  If the color is faded, it is less saturated. When it is intense and deep, it is more saturated. When the saturation is 0, we lose all color information and the image looks grayscale.

Value refers to lightness or brightness. It indicates how dark or bright the color is. It also signifies the amount of light that might have fallen on the object. When the value is 0, the image is black and when it is close to 255, the image is white.

This is a very convenient color space to work with because the "color" is contained in a single component (H). The Saturation indicates how saturated the color is and the value indicates how bright or dark the color is.

Let’s enumerate some of its properties.

 - Best thing is that it uses only one channel to describe color (H), making it very intuitive to specify color.
 - Device dependent.

## Alpha Channel
The alpha channel determines the transparency of a color. It's the fourth channel of an image that has pixel intensities ranging from 0-255. 
0 represents full transparency, 255 represents full opacity and intermediate values provide translucency. 
Certain file types support an alpha channel and one common file type is the 'PNG' file type. 

## YCrCb color space
The YCrCb color space is derived from the RGB color space. Its three components are :
 - Y (Luma), derived from the RGB values of the image
 - Cr = R - Y (how far is the red component from the Luma, also known as Red Difference)
 - Cb = B - Y (how far is the blue component from the Luma, also known as Blue Difference)

This color space has the following properties.

 - Separates the luminance and chrominance components into different channels.
 - Mostly used in compression ( of Cr and Cb components ) for TV Transmission.
 - Device dependent.

## Lab color space
The Lab color space consists of :
 - Lightness
 - A (a color component ranging from Green to Magenta): Lower values indicate green color and higher values indicate magenta (or red). 
 - B (a color component ranging from Blue to Yellow): Lower values indicate blue color and higher values indicate yellow color.

It has the following properties.

 - Perceptually uniform color space which approximates how we perceive color.
 - Independent of device ( capturing or displaying ).
 - Used extensively in Adobe Photoshop.
 - Is related to the RGB color space by a complex transformation equation.
**Finding Lane Lines on the Road**

[image1/image2/image3/image4/image5/image6]: 

[image1]: test_images/solidWhiteCurve.jpg "White lane lines with curve"
[image2]: test_images/solidWhiteRight.jpg "Continues white lane lines on right"
[image3]: test_images/solidYellowCurve.jpg "Yellow lane lines with curve"
[image4]: test_images/solidYellowCurve2.jpg "Yellow lane lines with curve"
[image5]: test_images/solidYellowLeft.jpg "Continues yellow lane lines on left"  
[image6]: test_images/whiteCarLaneSwitch.jpg "Stright lane"

---

### Reflection

1. With RGB based thresholding script couldn't detect yellow lane lines properly. So i have used HSV based thresholding 

to detect yellow lane lines.

My pipeline consisted of 16 steps. I have gone through following steps to detect lane lines:
	1.Import required modules.
	2.Make a list of test videos.
	3.Make a loop to apply test on all videos in list.
	4.Load video.
	5.Set various parameters for RGB threshold, HSV threshold gaussian blur and hough transform etc.
	6.Read a frame from video and find properties of image.
	7.Apply region mask.
	8.Apply RGB threshold.
	9.Prepare image for line detection by converting image to gray and apply gaussian blur and canny edge detection.
	10.Find lines and draw it on blank canvas.
	11.Convert region masked image to HSV and apply HSV threshold.
	12.Prepare image for line detection by converting HSV image to RGB then to gray and apply gaussian blur and canny edge detection.
	13.Find straight lines and draw it on blank canvas.
	14.Draw lines on orignal image and display image.
	15.Repete above steps from step 6 and continue till last frame of video.
	16.Repete steps from step 4 till last video of frame.
	
	Same as above steps are applied on test images. Only diffrence is we don't need to read frames, directly load image and apply steps mentioned above.

2. Identify potential shortcomings with your current pipeline

	One potential shortcoming would be when luminosity will change script will not work properly.

	Another shortcoming could be car speed and steer will be fully dependent on dash board camera as it need to take decisons from lane lines.
	
	Dash board camera may not identify far away lines.So to take proper desision car may slow down. 

3. Suggest possible improvements to your pipeline

	To improve my lane line detection i could use machine learning approch i.e.

	I tried my code in my own car.

	In india it is monsoon session so that on some roads lane lines are not perfectly white due to mud. 
	
	Due to yellowish lane lines script can't identify lane lines at some points.
	
	After implementing machine learning script will automatically adjust HSV value and RGB value to detect lane lines. 

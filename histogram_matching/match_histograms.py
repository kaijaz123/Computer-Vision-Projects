from skimage import exposure
import matplotlib.pyplot as plt
import argparse
import cv2

# construct the argument parser and parse the arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--source", type=str, default = 'empire_state_cloudy.png', help="path to the input source image")
argparser.add_argument("--reference", type=str, default = 'empire_state_sunset.png', help="path to the input reference image")

def histogram_matching(src, ref):
	src = cv2.imread(src)
	ref = cv2.imread(ref)

	# determine if we are performing multichannel histogram matching
	# and then perform histogram matching itself
	multi = True if src.shape[-1] > 1 else False
	matched = exposure.match_histograms(src, ref, multichannel=multi)

	# visualize the histogram of each output
	src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
	ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
	match_rgb = cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)
	imgs = [src_rgb, ref_rgb, match_rgb]
	img_name = ['Original', 'Reference', 'Matched']
	plt.style.use("dark_background")
	plt.figure(figsize=(10,8))
	for index, img in enumerate(imgs):
		plt.subplot(2,3, index+1)
		plt.imshow(img)
		plt.subplot(2,3, index+4)
		plt.hist(img.ravel(), 255)
		plt.xlabel(img_name[index])
	plt.show()

if __name__ == '__main__':
	args = argparser.parse_args()
	histogram_matching(args.source, args.reference)

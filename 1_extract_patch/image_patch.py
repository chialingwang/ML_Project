from PIL import Image
import os,sys

def ImagePatchProcessing(path, patch_size):

	im = Image.open(path)
	im_array = list()
	for pixel in iter(im.getdata()):
		im_array.append(pixel)
	# end for

	M = im.size[0]
	N = im.size[1]
	half_patch_size = int(patch_size/2)
	feature_array = list()

	for i in range(0, M):
		for j in range(0, N):
			patch_array = list()
			for di in range(-half_patch_size, half_patch_size+1):
				for dj in range(-half_patch_size, half_patch_size+1):
					m = (i+M+di)%M
					n = (j+N+dj)%N
					feature_array.append(im_array[m*M+n])
				# end for
			# end for
			#print(patch_array)
			#feature_array.append(patch_array)
		# end for
	# end for	
	return feature_array


#ImagePatchProcessing("01-002.png", 3)
# ML_Project
In this project we present a approach to do texture image recognition 
by performing the vlad of all the texture images based on the \(CUReT\) texture database. 
The approach consists of following steps: 

1) Database prepossessing; 
https://github.com/chialingwang/ML_Project/tree/master/1_extract_patch/scrip_gen.csh

2) Apply Mini-batch kmeans to find the centroids; 
https://github.com/chialingwang/ML_Project/tree/master/2_extract_centroids_withscikit/scrip_gen.csh

3) Get Vlad of each image to find out the feature of image; 
https://github.com/chialingwang/ML_Project/blob/master/3_get_vlad_withscikit_new/scrip_gen.csh

4) Texture classification by using knn provided from scikit-learn. 
https://github.com/chialingwang/ML_Project/blob/master/4_classification_withscikit/scrip_gen.csh


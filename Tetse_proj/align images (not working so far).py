# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:22:21 2019

@author: marijnhazelbag
"""

import sys

sys.path.append('/usr/local/lib/python2.7/Lib/site-packages')

pip install opencv-python

import Python-2.7.x

import imp
imp.find_module("cv2")

import cv2
print(cv2.__version__)

from __future__ import print_function
import cv2
import numpy as np
 
 
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

img1 =mpimg.imread(" ".join(resultLeft.loc[222,[0,1,2]]))
img2 =mpimg.imread(" ".join(resultLeft.loc[223,[0,1,2]]))
plt.imshow(img1)
plt.imshow(img2)

def alignImages(im1, im2):
 
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
   
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
   
  return im1Reg, h
 
    
a1= alignImages(img1, img2)

imReg, h = alignImages(img1, img2)
 
warpPerspective(img1, img2, h, im2.size())
?warpPerspective

if __name__ == '__main__':
   
  # Read reference image
  refFilename = "form.jpg"
  print("Reading reference image : ", refFilename)
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
 
  # Read image to be aligned
  imFilename = "scanned-form.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
   
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
   
  # Write aligned image to disk. 
  outFilename = "aligned.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)
 
  # Print estimated homography
  print("Estimated homography : \n",  h)
  
  
  
  
  
  #####
  
  ####
  
const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;
 
 
alignImages(Mat &im1, Mat &im2, Mat &im1Reg, Mat &h)
{
   
  // Convert images to grayscale
  Mat im1Gray, im2Gray;
  cvtColor(im1, im1Gray, CV_BGR2GRAY);
  cvtColor(im2, im2Gray, CV_BGR2GRAY);
   
  // Variables to store keypoints and descriptors
  std::vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;
   
  // Detect ORB features and compute descriptors.
  Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
  orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
  orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);
   
  // Match features.
  std::vector<DMatch> matches;
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  matcher->match(descriptors1, descriptors2, matches, Mat());
   
  // Sort matches by score
  std::sort(matches.begin(), matches.end());
   
  // Remove not so good matches
  const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
  matches.erase(matches.begin()+numGoodMatches, matches.end());
   
   
  // Draw top matches
  Mat imMatches;
  drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
  imwrite("matches.jpg", imMatches);
   
   
  // Extract location of good matches
  std::vector<Point2f> points1, points2;
   
  for( size_t i = 0; i < matches.size(); i++ )
  {
    points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
    points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
  }
   
  // Find homography
  h = findHomography( points1, points2, RANSAC );
   
  // Use homography to warp image
  warpPerspective(im1, im1Reg, h, im2.size());
   
}
 
 
int main(int argc, char **argv)
{
  // Read reference image
  string refFilename("form.jpg"); 
  cout << "Reading reference image : " << refFilename << endl; 
  Mat imReference = imread(refFilename);
 
 
  // Read image to be aligned
  string imFilename("scanned-form.jpg");
  cout << "Reading image to align : " << imFilename << endl; 
  Mat im = imread(imFilename);
 
   
  // Registered image will be resotred in imReg. 
  // The estimated homography will be stored in h. 
  Mat imReg, h;
   
  // Align images
  cout << "Aligning images ..." << endl; 
  alignImages(im, imReference, imReg, h);
 
  // Write aligned image to disk. 
  string outFilename("aligned.jpg");
  cout << "Saving aligned image : " << outFilename << endl; 
  imwrite(outFilename, imReg);
 
  // Print estimated homography
  cout << "Estimated homography : \n" << h << endl; 
   
}
  
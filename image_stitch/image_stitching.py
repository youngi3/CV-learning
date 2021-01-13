from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

featureType = {'SIFT', 'SURF','FREAK', 'LATCH', 'BoostDesc'}
class ImageStitching():
    def __init__(self, ratio = 0.85, min_batch = 10, featureName = 'SIFT', smothing_wd_sz = 800):
        self.ratio = 0.85
        self.min_batch = 10
        if featureName in featureType:
            self.feature = eval(f'cv2.xfeatures2d.{featureName}_create()')
        else:
            print('featureName is not in candidates')
        self.smoothing_window_size = smothing_wd_sz
    
    def my_show(self, img1, img2, size = (10,10)):
        plt.figure(figsize=size)
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    #feature points match method    
    def bfMatcher(self, mode = cv2.NORM_L2):
        bfMatcher = cv2.BFMatcher(mode)
        return bfMatcher    
    #feature points match method    
    def flannMatcher(self, FLANN_INDEX_KDTREE = 0, trees = 5, checks = 50):
        FLANN_INDEX_KDTREE = FLANN_INDEX_KDTREE
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
        search_params = dict(checks = checks)
        flannMatcher = cv2.FlannBasedMatcher(index_params, search_params)
        return flannMatcher
    
    
    def registration(self, img1, img2, method = 'bf'):
        kp1, des1 = self.feature.detectAndCompute(img1, None)
        kp2, des2 = self.feature.detectAndCompute(img2, None)
        if method == 'bf':
            matcher = self.bfMatcher()
        elif method == 'flann':
            matcher = self.flannMatcher()
        raw_matcher = matcher.knnMatch(des1,des2, k = 2)        #filter wrong match points Knnmatch:k=2,返回最匹配的两个
        good_points = []
        good_matches = []
        for m1, m2 in raw_matcher:
            if m1.distance < m2.distance * self.ratio:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags = 2)
        cv2.imwrite('matching.jpg',img3)
        if len(good_points) > self.min_batch:
            img1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
            img2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
        H, status = cv2.findHomography(img2_kp, img1_kp, cv2.RANSAC, 5.0)  #RANSAC去噪，计算但应性矩阵
        return H
    
    #创建mask
    def create_mask(self, img1, img2, version):
        height_1 = img1.shape[0]
        width_1 = img1.shape[1]
        wigth_2 = img2.shape[1]
        height_panorama = height_1
        width_panorama = width_1 + wigth_2
        offset = int(self.smoothing_window_size/2)
        mask = np.zeros((height_panorama, width_panorama))
        if version == 'left':
            mask[:,width_1 - offset: width_1 + offset] = np.tile(np.linspace(1,0,2*offset),(height_panorama, 1))
            mask[:, :width_1 - offset] = 1
        else:
            mask[:,width_1 - offset: width_1 + offset] = np.tile(np.linspace(0,1,2*offset),(height_panorama, 1))
            mask[:, width_1 + offset:] = 1
        return cv2.merge([mask, mask, mask])
    
    def my_merge(self, H, img1, img2):
        height_1 = img1.shape[0]
        width_1 = img1.shape[1]
        width_2 = img2.shape[1]
        height_panorama = height_1
        width_panorama = width_1 + width_2
        
        panorama1 = np.zeros((height_panorama,width_panorama, 3))
        mask1 = self.create_mask(img1, img2, version = 'left')
        panorama1[:height_1, :width_1, :] = img1
        panorama1 *= mask1
        
        mask2 = self.create_mask(img1, img2, version = 'right')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))* mask2
        result = panorama1 + panorama2
        
        rows, cols = np.nonzero(result[:,:,0])
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1        
        result = result[min_row:max_row,min_col:max_col,:]
        return result
    
def main(argv1, argv2, argv3):
    img1 = cv2.imread(argv1,1)
    img2 = cv2.imread(argv2,1)
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    image_stitching = ImageStitching()
    H = image_stitching.registration(img1, img2, argv3)
    result = image_stitching.my_merge(H, img1, img2)
    cv2.imwrite('panorama.jpg', result)
    
if __name__ == '__main__':
    try:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    except:
        print("error!")
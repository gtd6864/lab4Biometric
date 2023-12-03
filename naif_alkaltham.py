import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Improved Image cleanup
def cleanup_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)
    # Adjust adaptive thresholding parameters if necessary
    return cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Enhanced Minutiae detection
def find_minutiae(path, disp=False):
    img = cleanup_img(path)
    # Adjust parameters for Harris corner detection based on your image characteristics
    dst = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # Refine corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.goodFeaturesToTrack(dst, maxCorners=150, qualityLevel=0.01, minDistance=10, blockSize=3, useHarrisDetector=True)

    # Draw corners for display
    if disp:
        img2 = img.copy()
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img2, (x, y), 3, 255, -1)
        plt.imshow(img2)
        plt.show()

    return corners

# Feature extraction with fixed size
def extract_features(minutiae, fixed_size=500):
    features = np.array([m.ravel() for m in minutiae if m is not None]).flatten()
    if len(features) > fixed_size:
        features = features[:fixed_size]
    elif len(features) < fixed_size:
        features = np.pad(features, (0, fixed_size - len(features)), 'constant')
    return features

# K-Nearest Neighbors Classifier
def ml_technique_one(train_features, train_labels):
    classifier = KNeighborsClassifier(n_neighbors=5)  # Adjust the number of neighbors
    classifier.fit(train_features, train_labels)
    return classifier

# Support Vector Machine Classifier
def ml_technique_two(train_features, train_labels):
    classifier = SVC(gamma='scale', kernel='rbf', probability=True)  # Experiment with different kernels
    classifier.fit(train_features, train_labels)
    return classifier

# Performance evaluation with additional metrics
def evaluate_performance(classifier, test_features, test_labels):
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, zero_division=0)
    cm = confusion_matrix(test_labels, predictions)
    
    # Additional Metrics
    max_fnr = 0  # Placeholder for maximum False Negative Rate
    min_fnr = 1  # Placeholder for minimum False Negative Rate
    sum_fnr = 0  # Sum of False Negative Rates
    max_fpr = 0  # Placeholder for maximum False Positive Rate
    min_fpr = 1  # Placeholder for minimum False Positive Rate
    sum_fpr = 0  # Sum of False Positive Rates
    
    for i in range(len(test_labels)):
        if test_labels[i] == 1 and predictions[i] == 0:  # False Negative
            fnr = 1
            sum_fnr += fnr
            if fnr > max_fnr:
                max_fnr = fnr
            if fnr < min_fnr:
                min_fnr = fnr
        if test_labels[i] == 0 and predictions[i] == 1:  # False Positive
            fpr = 1
            sum_fpr += fpr
            if fpr > max_fpr:
                max_fpr = fpr
            if fpr < min_fpr:
                min_fpr = fpr
    
    num_samples = len(test_labels)
    avg_fnr = sum_fnr / num_samples
    avg_fpr = sum_fpr / num_samples
    
    eer = (avg_fnr + avg_fpr) / 2  # Equal Error Rate
    
    return accuracy, report, max_fnr, min_fnr, avg_fnr, max_fpr, min_fpr, avg_fpr, eer

# Main function
# Main function
def main():
    image_dir = '/home/moon/Desktop/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/data'  # Replace with your image directory
    image_paths = glob.glob(f'{image_dir}/*.png')
    image_paths.sort()

    features = []
    labels = []
    for path in image_paths:
        minutiae = find_minutiae(path, disp=False)
        extracted_features = extract_features(minutiae)
        
        # Extract the label from the text and encode it as 0 or 1
        label_path = path.replace('.png', '.txt')
        with open(label_path, 'r') as f:
            label_text = f.read().strip()
            if "Gender: M" in label_text:  # Replace with the appropriate condition for labeling
                label = 1
            else:
                label = 0
        
        # Debugging: Print extracted features and label for each image
        #print("Extracted Features:", extracted_features)
        #print("Label:", label)
        
        features.append(extracted_features)
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels, dtype=int)  # Ensure labels are correctly encoded as integers

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)

    knn_classifier = ml_technique_one(train_features, train_labels)
    svm_classifier = ml_technique_two(train_features, train_labels)

    knn_accuracy, knn_report, max_fnr_knn, min_fnr_knn, avg_fnr_knn, max_fpr_knn, min_fpr_knn, avg_fpr_knn, eer_knn = evaluate_performance(knn_classifier, test_features, test_labels)
    svm_accuracy, svm_report, max_fnr_svm, min_fnr_svm, avg_fnr_svm, max_fpr_svm, min_fpr_svm, avg_fpr_svm, eer_svm = evaluate_performance(svm_classifier, test_features, test_labels)

    # Print results
    print("KNN Accuracy: ", knn_accuracy)
    print("KNN Report:\n", knn_report)
    print(f"KNN Max FNR: {max_fnr_knn:.4f}, Min FNR: {min_fnr_knn:.4f}, Avg FNR: {avg_fnr_knn:.4f}")
    print(f"KNN Max FPR: {max_fpr_knn:.4f}, Min FPR: {min_fpr_knn:.4f}, Avg FPR: {avg_fpr_knn:.4f}")
    print(f"KNN Equal Error Rate (EER): {eer_knn:.4f}")

    print("SVM Accuracy: ", svm_accuracy)
    print("SVM Report:\n", svm_report)
    print(f"SVM Max FNR: {max_fnr_svm:.4f}, Min FNR: {min_fnr_svm:.4f}, Avg FNR: {avg_fnr_svm:.4f}")
    print(f"SVM Max FPR: {max_fpr_svm:.4f}, Min FPR: {min_fnr_svm:.4f}, Avg FPR: {avg_fpr_svm:.4f}")
    print(f"SVM Equal Error Rate (EER): {eer_svm:.4f}")

if __name__ == '__main__':
    main()


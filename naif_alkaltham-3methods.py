import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
from sklearn.ensemble import RandomForestClassifier


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

# Random Forest Classifier
def ml_technique_three(train_features, train_labels):
    classifier = RandomForestClassifier(n_estimators=100, max_depth=200, min_samples_split=10, random_state=42)
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

def main():
    
    image_dir = 'NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/full-data'
    reference_image_paths = glob.glob(f'{image_dir}/f*.png')
    subject_image_paths = {os.path.basename(p).split('_')[0][1:]: p for p in glob.glob(f'{image_dir}/s*.png')}
    reference_image_paths.sort()

    paired_features = []
    labels = []

    # Load data and extract features for paired images
    for ref_path in reference_image_paths:
        file_id = os.path.basename(ref_path).split('_')[0][1:]
        subj_path = subject_image_paths.get(file_id)
        if subj_path:
            ref_minutiae = find_minutiae(ref_path, disp=False)
            subj_minutiae = find_minutiae(subj_path, disp=False)

            ref_features = extract_features(ref_minutiae)
            subj_features = extract_features(subj_minutiae)
            combined_features = np.concatenate((ref_features, subj_features))

            paired_features.append(combined_features)

            label_path = ref_path.replace('.png', '.txt')
            with open(label_path, 'r') as f:
                label_text = f.read().strip()
                label = 0 if "Gender: M" in label_text else 1
            labels.append(label)

            # Debugging: Print extracted features, label, and image names
            #print("Extracted Features:", combined_features)
            #print("Label:", label)
            #print("Image image pairs: ", os.path.basename(ref_path), os.path.basename(subj_path))
        

    paired_features = np.array(paired_features)
    labels = np.array(labels, dtype=int)

    train_features, test_features, train_labels, test_labels = train_test_split(paired_features, labels, test_size=0.25, random_state=42)
    
    # Print information about the dataset
    print("\n")
    total_samples = len(train_features) + len(test_features)
    train_samples = len(train_features)
    test_samples = len(test_features)
    print(f"Total Samples: {total_samples}")
    print(f"Training Samples: {train_samples}")
    print(f"Testing Samples: {test_samples}")
    print("\n")

    knn_classifier = ml_technique_one(train_features, train_labels)
    svm_classifier = ml_technique_two(train_features, train_labels)
    rf_classifier = ml_technique_three(train_features, train_labels)

    knn_accuracy, knn_report, max_fnr_knn, min_fnr_knn, avg_fnr_knn, max_fpr_knn, min_fpr_knn, avg_fpr_knn, eer_knn = evaluate_performance(knn_classifier, test_features, test_labels)
    svm_accuracy, svm_report, max_fnr_svm, min_fnr_svm, avg_fnr_svm, max_fpr_svm, min_fpr_svm, avg_fpr_svm, eer_svm = evaluate_performance(svm_classifier, test_features, test_labels)
    rf_accuracy, rf_report, max_fnr_rf, min_fnr_rf, avg_fnr_rf, max_fpr_rf, min_fpr_rf, avg_fpr_rf, eer_rf = evaluate_performance(rf_classifier, test_features, test_labels)

    # Print KNN and SVM results
    # Print results
    print("KNN Accuracy: ", knn_accuracy)
    print("KNN Report:\n", knn_report)
    print(f"KNN Max FNR: {max_fnr_knn:.4f}, Min FNR: {min_fnr_knn:.4f}, Avg FNR: {avg_fnr_knn:.4f}")
    print(f"KNN Max FPR: {max_fpr_knn:.4f}, Min FPR: {min_fpr_knn:.4f}, Avg FPR: {avg_fpr_knn:.4f}")
    print(f"KNN Equal Error Rate (EER): {eer_knn:.4f}")
    print("\n")

    print("SVM Accuracy: ", svm_accuracy)
    print("SVM Report:\n", svm_report)
    print(f"SVM Max FNR: {max_fnr_svm:.4f}, Min FNR: {min_fnr_svm:.4f}, Avg FNR: {avg_fnr_svm:.4f}")
    print(f"SVM Max FPR: {max_fpr_svm:.4f}, Min FPR: {min_fnr_svm:.4f}, Avg FPR: {avg_fpr_svm:.4f}")
    print(f"SVM Equal Error Rate (EER): {eer_svm:.4f}")
    print("\n")
    
    print("Random Forest Accuracy: ", rf_accuracy)
    print("Random Forest  Report:\n", rf_report)
    print(f"Random Forest  Max FNR: {max_fnr_rf:.4f}, Min FNR: {min_fnr_rf:.4f}, Avg FNR: {avg_fnr_rf:.4f}")
    print(f"Random Forest  Max FPR: {max_fpr_rf:.4f}, Min FPR: {min_fnr_rf:.4f}, Avg FPR: {avg_fpr_rf:.4f}")
    print(f"Random Forest  Equal Error Rate (EER): {eer_rf:.4f}")
    print("\n")

    # Create and display the summary table
    summary_table = [
        ["KNN", knn_accuracy, max_fnr_knn, min_fnr_knn, avg_fnr_knn, max_fpr_knn, min_fpr_knn, avg_fpr_knn, eer_knn],
        ["SVM", svm_accuracy, max_fnr_svm, min_fnr_svm, avg_fnr_svm, max_fpr_svm, min_fpr_svm, avg_fpr_svm, eer_svm],
        ["Random Forest", rf_accuracy, max_fnr_rf, min_fnr_rf, avg_fnr_rf, max_fpr_rf, min_fpr_rf, avg_fpr_rf, eer_rf]
    ]
    headers = ["Method", "Accuracy", "Max FNR", "Min FNR", "Avg FNR", "Max FPR", "Min FPR", "Avg FPR", "EER"]
    print(tabulate(summary_table, headers, tablefmt="grid"))

if __name__ == '__main__':
    main()


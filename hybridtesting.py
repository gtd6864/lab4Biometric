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
    corners = cv2.goodFeaturesToTrack(dst, maxCorners=150, qualityLevel=0.01, minDistance=10, blockSize=3,
                                      useHarrisDetector=True)

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
    classifier = KNeighborsClassifier(n_neighbors=50)  # Adjust the number of neighbors
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
def evaluate_performance(knn_classifier, svm_classifier, rf_classifier, test_features, test_labels):
    max_frr = 0  # Placeholder for maximum False Rejection Rate
    min_frr = 1  # Placeholder for minimum False Rejection Rate
    max_far = 0  # Placeholder for maximum False Acceptance Rate
    min_far = 1  # Placeholder for minimum False Acceptance Rate
    avg_frr = 0
    avg_far = 0
    eer = 1


    # base final prediction off of a threshold - test a variety of thresholds
    for i in range(1, 99):
        predictions = np.add(knn_classifier.predict_proba(test_features)[:, 1], svm_classifier.predict_proba(test_features)[:, 1])
        predictions = np.add(predictions, rf_classifier.predict_proba(test_features)[:, 1])
        predictions = ((predictions[:]/3) >= i/100).astype(int)

        # Additional Metrics
        sum_frr = 0  # Sum of False Rejects
        sum_far = 0  # Sum of False Accepts
        true_rejects = 0  # total number of true rejects in the tested data
        true_accepts = 0  # total number of true accepts in the tested data

        for j in range(len(test_labels)):
            if (test_labels[j] == 1):  # Count all true accepts
                true_accepts += 1
            else:  # Count all true rejects
                true_rejects += 1
            if test_labels[j] == 1 and predictions[j] == 0:  # False Rejection
                sum_frr += 1
            if test_labels[j] == 0 and predictions[j] == 1:  # False Acceptance
                sum_far += 1

        sub_avg_frr = sum_frr / true_accepts
        sub_avg_far = sum_far / true_rejects
        if sub_avg_frr > max_frr:
            max_frr = sub_avg_frr
        if sub_avg_frr < min_frr:
            min_frr = sub_avg_frr
        if sub_avg_far > max_far:
            max_far = sub_avg_far
        if sub_avg_far < min_far:
            min_far = sub_avg_far

        if (sub_avg_frr - .05) <= sub_avg_far and sub_avg_far <= (sub_avg_frr + .05):
            if (sub_avg_frr + sub_avg_far) / 2 < eer:
                eer = (sub_avg_frr + sub_avg_far) / 2  # Equal Error Rate
                accuracy = accuracy_score(test_labels, predictions)
                report = classification_report(test_labels, predictions, zero_division=0)

        avg_frr += sub_avg_frr
        avg_far += sub_avg_far

    avg_frr = avg_frr / 99
    avg_far = avg_far / 99

    return accuracy, report, max_frr, min_frr, avg_frr, max_far, min_far, avg_far, eer


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
            # print("Extracted Features:", combined_features)
            # print("Label:", label)
            # print("Image image pairs: ", os.path.basename(ref_path), os.path.basename(subj_path))

    paired_features = np.array(paired_features)
    labels = np.array(labels, dtype=int)

    # separate the data into first 1500 for training, last 500 for testing
    train_features, test_features, train_labels, test_labels = train_test_split(paired_features, labels, test_size=0.25,
                                                                                shuffle=False)

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

    hybrid_accuracy, hybrid_report, max_frr_hybrid, min_frr_hybrid, avg_frr_hybrid, max_far_hybrid, min_far_hybrid, avg_far_hybrid, eer_hybrid = evaluate_performance(
        knn_classifier, svm_classifier, rf_classifier, test_features, test_labels)


    # Print KNN and SVM results
    # Print results
    print("Hybrid Accuracy: ", hybrid_accuracy)
    print("Hybrid Report:\n", hybrid_report)
    print(f"Hybrid Max FRR: {max_frr_hybrid:.4f}, Min FRR: {min_frr_hybrid:.4f}, Avg FRR: {avg_frr_hybrid:.4f}")
    print(f"Hybrid Max FAR: {max_far_hybrid:.4f}, Min FAR: {min_far_hybrid:.4f}, Avg FAR: {avg_far_hybrid:.4f}")
    print(f"Hybrid Equal Error Rate (EER): {eer_hybrid:.4f}")
    print("\n")


    # Create and display the summary table
    summary_table = [
        ["hybrid", hybrid_accuracy, max_frr_hybrid, min_frr_hybrid, avg_frr_hybrid, max_far_hybrid, min_far_hybrid, avg_far_hybrid, eer_hybrid]
    ]
    headers = ["Method", "Accuracy", "Max FRR", "Min FRR", "Avg FRR", "Max FAR", "Min FAR", "Avg FAR", "EER"]
    print(tabulate(summary_table, headers, tablefmt="grid"))


if __name__ == '__main__':
    main()


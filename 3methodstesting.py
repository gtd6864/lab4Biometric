import numpy as np
import cv2
import glob

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
from sklearn.ensemble import RandomForestClassifier
import random


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
    classifier = KNeighborsClassifier(n_neighbors=100)  # Adjust the number of neighbors
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
    max_frr = 0  # Placeholder for maximum False Rejection Rate
    min_frr = 1  # Placeholder for minimum False Rejection Rate
    max_far = 0  # Placeholder for maximum False Acceptance Rate
    min_far = 1  # Placeholder for minimum False Acceptance Rate
    avg_frr = 0
    avg_far = 0
    eer = 1

    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, zero_division=0)
    # base final prediction off of a threshold - test a variety of thresholds
    for i in range(1, 99):
        predictions = (classifier.predict_proba(test_features)[:, 1] >= (i / 100)).astype(int)

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

        if (sub_avg_frr - .07) <= sub_avg_far and sub_avg_far <= (sub_avg_frr + .07):
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
    # Load data and extract features for the 1500 training images, 1500 matches and 3000 non-matches
    for i in range(len(reference_image_paths) - 500):
        ref_path = reference_image_paths[i]
        file_id = os.path.basename(ref_path).split('_')[0][1:]
        subj_path = subject_image_paths.get(file_id)
        if subj_path:
            ref_minutiae = find_minutiae(ref_path, disp=False)
            subj_minutiae = find_minutiae(subj_path, disp=False)

            ref_features = extract_features(ref_minutiae)
            subj_features = extract_features(subj_minutiae)
            combined_features = np.concatenate((ref_features, subj_features))

            paired_features.append(combined_features)
            # all feature pairs should be a match
            labels.append(1)
        # add 2 mismatch pairs for every fingerprint as well
        if i == 0:
            dif_path = reference_image_paths[len(reference_image_paths) - 501]
            dif_path_2 = reference_image_paths[len(reference_image_paths) - 502]
            dif_path_3 = reference_image_paths[len(reference_image_paths) - 503]
        elif i == 1:
            dif_path = reference_image_paths[0]
            dif_path_2 = reference_image_paths[len(reference_image_paths) - 501]
            dif_path_3 = reference_image_paths[len(reference_image_paths) - 502]
        elif i == 2:
            dif_path = reference_image_paths[1]
            dif_path_2 = reference_image_paths[0]
            dif_path_3 = reference_image_paths[len(reference_image_paths) - 501]
        else:
            dif_path = reference_image_paths[i - 1]
            dif_path_2 = reference_image_paths[i - 2]
            dif_path_3 = reference_image_paths[i - 3]
        file_id = os.path.basename(dif_path).split('_')[0][1:]
        subj_path = subject_image_paths.get(file_id)
        if subj_path:
            subj_minutiae = find_minutiae(subj_path, disp=False)
            subj_features = extract_features(subj_minutiae)
            combined_features = np.concatenate((ref_features, subj_features))

            paired_features.append(combined_features)
            # all shifted feature pairs should be a non-match
            labels.append(0)
        file_id = os.path.basename(dif_path_2).split('_')[0][1:]
        subj_path = subject_image_paths.get(file_id)
        if subj_path:
            subj_minutiae = find_minutiae(subj_path, disp=False)
            subj_features = extract_features(subj_minutiae)
            combined_features = np.concatenate((ref_features, subj_features))

            paired_features.append(combined_features)
            # all shifted feature pairs should be a non-match
            labels.append(0)
        file_id = os.path.basename(dif_path_3).split('_')[0][1:]
        subj_path = subject_image_paths.get(file_id)
        if subj_path:
            subj_minutiae = find_minutiae(subj_path, disp=False)
            subj_features = extract_features(subj_minutiae)
            combined_features = np.concatenate((ref_features, subj_features))

            paired_features.append(combined_features)
            # all shifted feature pairs should be a non-match
            labels.append(0)

    train_features = np.array(paired_features)
    train_labels = np.array(labels, dtype=int)

    # create our test features and labels
    test_features = []
    test_labels = []

    # compare each of our 500 test values to either itself or another random test value
    for i in range(500):
        ref_path = reference_image_paths[i + 1500]
        ref_minutiae = find_minutiae(ref_path, disp=False)
        ref_features = extract_features(ref_minutiae)
        k = random.randint(0, 1)
        # just do a normal matching pair
        if k == 0:
            file_id = os.path.basename(ref_path).split('_')[0][1:]
            subj_path = subject_image_paths.get(file_id)
            if subj_path:
                subj_minutiae = find_minutiae(subj_path, disp=False)
                subj_features = extract_features(subj_minutiae)
                combined_features = np.concatenate((ref_features, subj_features))
                test_features.append(combined_features)
                # all feature pairs should be a match
                test_labels.append(1)
        # otherwise, pair it with a random other value from the test fingerprints
        else:
            j = random.randint(1500, 1999)
            while j == i:
                j = random.randint(1500, 1999)
            dif_path = reference_image_paths[j]
            file_id = os.path.basename(dif_path).split('_')[0][1:]
            subj_path = subject_image_paths.get(file_id)
            if subj_path:
                subj_minutiae = find_minutiae(subj_path, disp=False)
                subj_features = extract_features(subj_minutiae)
                combined_features = np.concatenate((ref_features, subj_features))
                test_features.append(combined_features)
                # a mismatch pair should be always be a 0
                test_labels.append(0)

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

    knn_accuracy, knn_report, max_frr_knn, min_frr_knn, avg_frr_knn, max_far_knn, min_far_knn, avg_far_knn, eer_knn = evaluate_performance(
        knn_classifier, test_features, test_labels)
    svm_accuracy, svm_report, max_frr_svm, min_frr_svm, avg_frr_svm, max_far_svm, min_far_svm, avg_far_svm, eer_svm = evaluate_performance(
        svm_classifier, test_features, test_labels)
    rf_accuracy, rf_report, max_frr_rf, min_frr_rf, avg_frr_rf, max_far_rf, min_far_rf, avg_far_rf, eer_rf = evaluate_performance(
        rf_classifier, test_features, test_labels)

    # Print KNN and SVM results
    # Print results
    print("KNN Accuracy: ", knn_accuracy)
    print("KNN Report:\n", knn_report)
    print(f"KNN Max FRR: {max_frr_knn:.4f}, Min FRR: {min_frr_knn:.4f}, Avg FRR: {avg_frr_knn:.4f}")
    print(f"KNN Max FAR: {max_far_knn:.4f}, Min FAR: {min_far_knn:.4f}, Avg FAR: {avg_far_knn:.4f}")
    print(f"KNN Equal Error Rate (EER): {eer_knn:.4f}")
    print("\n")

    print("SVM Accuracy: ", svm_accuracy)
    print("SVM Report:\n", svm_report)
    print(f"SVM Max FRR: {max_frr_svm:.4f}, Min FRR: {min_frr_svm:.4f}, Avg FRR: {avg_frr_svm:.4f}")
    print(f"SVM Max FAR: {max_far_svm:.4f}, Min FAR: {min_frr_svm:.4f}, Avg FAR: {avg_far_svm:.4f}")
    print(f"SVM Equal Error Rate (EER): {eer_svm:.4f}")
    print("\n")

    print("Random Forest  Report:\n", rf_report)
    print(f"Random Forest  Max FRR: {max_frr_rf:.4f}, Min FRR: {min_frr_rf:.4f}, Avg FRR: {avg_frr_rf:.4f}")
    print(f"Random Forest  Max FAR: {max_far_rf:.4f}, Min FAR: {min_frr_rf:.4f}, Avg FAR: {avg_far_rf:.4f}")
    print(f"Random Forest  Equal Error Rate (EER): {eer_rf:.4f}")
    print("\n")

    # Create and display the summary table
    summary_table = [
        ["KNN", knn_accuracy, max_frr_knn, min_frr_knn, avg_frr_knn, max_far_knn, min_far_knn, avg_far_knn, eer_knn],
        ["SVM", svm_accuracy, max_frr_svm, min_frr_svm, avg_frr_svm, max_far_svm, min_far_svm, avg_far_svm, eer_svm],
        ["Random Forest", rf_accuracy, max_frr_rf, min_frr_rf, avg_frr_rf, max_far_rf, min_far_rf, avg_far_rf, eer_rf]
    ]
    headers = ["Method", "Accuracy", "Max FRR", "Min FRR", "Avg FRR", "Max FAR", "Min FAR", "Avg FAR", "EER"]
    print(tabulate(summary_table, headers, tablefmt="grid"))


if __name__ == '__main__':
    main()

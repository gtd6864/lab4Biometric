import numpy as np
import cv2
import glob

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
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


# Support Vector Machine Classifier
def svm_ml_technique(train_features, train_labels):
    classifier = SVC(gamma='scale', kernel='rbf', probability=True)  # Experiment with different kernels
    classifier.fit(train_features, train_labels)
    return classifier


# Performance evaluation with additional metrics
def evaluate_performance(classifier, test_features, test_labels):
    frr = 0
    far = 0

    r = random.sample(range(500), 100)  # choose a random set of indices to test
    random_test_features = []
    random_test_labels = []
    for i in r:
        random_test_features.append(test_features[i])
        random_test_labels.append(test_labels[i])
    predictions = (classifier.predict_proba(random_test_features)[:, 1] >= .13).astype(int)
    accuracy = accuracy_score(random_test_labels, predictions)
    report = classification_report(random_test_labels, predictions, zero_division=0)

    # Additional Metrics
    sum_frr = 0  # Sum of False Rejects
    sum_far = 0  # Sum of False Accepts
    true_rejects = 0  # total number of true rejects in the tested data
    true_accepts = 0  # total number of true accepts in the tested data

    for j in range(len(random_test_labels)):
        if (random_test_labels[j] == 1):  # Count all true accepts
            true_accepts += 1
        else:  # Count all true rejects
            true_rejects += 1
        if random_test_labels[j] == 1 and predictions[j] == 0:  # False Rejection
            sum_frr += 1
        if random_test_labels[j] == 0 and predictions[j] == 1:  # False Acceptance
            sum_far += 1

    frr = sum_frr / true_accepts
    far = sum_far / true_rejects

    return accuracy, report, frr, far


def compare_fingerprints(classifier, path_a, path_b, similarity_threshold=0.13, debug=False):
    # Detect minutiae points
    minutiae_a = find_minutiae(path_a, debug)
    minutiae_b = find_minutiae(path_b, debug)

    # Extract features
    features_a = extract_features(minutiae_a)
    features_b = extract_features(minutiae_b)

    combined_features = np.concatenate((features_a, features_b))
    # Calculate similarity
    similarity = (classifier.predict_proba(combined_features)[:, 1] >= similarity_threshold).astype(int)

    # Determine if the fingerprints are similar or not
    return True if similarity[0] == 1 else False


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

    svm_classifier = svm_ml_technique(train_features, train_labels)

    svm_accuracy, svm_report, frr, far = evaluate_performance(
        svm_classifier, test_features, test_labels)

    print("SVM Accuracy: ", svm_accuracy)
    print("SVM Report:\n", svm_report)
    print(f"SVM  FRR: {frr:.4f}")
    print(f"SVM FAR: {far:.4f}")
    print("\n")

    # Create and display the summary table
    summary_table = [
        ["SVM", svm_accuracy, frr, far]
    ]
    headers = ["Method", "Accuracy", "FRR", "FAR"]
    print(tabulate(summary_table, headers, tablefmt="grid"))


if __name__ == '__main__':
    main()

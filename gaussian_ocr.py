import math
from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix


"""
Implements a simple OCR model using gaussian class conditionals with pooled diagonal covariance
as part of an exercise in the statistical classification course @ RWTH.
"""


class BayesClassifier(object):

    def __init__(self, num_classes, num_dims, train):
        self.covariance_matrix, self.means, self.prior = self._estimate_parameters(num_classes, num_dims, train)

    def _estimate_diagonal_cov_matrix(self, means, num_dims, train):
        """Estimates diagonal pooled covariance matrix."""
        covariance_matrix = np.zeros(num_dims)

        for features, target in train:
            diff = features - means[target - 1]
            covariance_matrix += diff ** 2

        covariance_matrix /= len(train)

        return covariance_matrix

    def _estimate_means(self, num_classes, num_dims, prior, train):
        freqs = [int(class_prior * len(train)) for class_prior in prior]
        means = [np.zeros(num_dims) for _ in range(num_classes)]

        for feature, target in train:
            means[target - 1] += np.array(feature)

        means = [mean / freq for mean, freq in zip(means, freqs)]
        
        return means

    def _estimate_parameters(self, num_classes, num_dims, train):
        prior = self._estimate_prior(train)
        means = self._estimate_means(num_classes, num_dims, prior, train)
        covariance_matrix = self._estimate_diagonal_cov_matrix(means, num_dims, train)

        return covariance_matrix, means, prior

    def _estimate_prior(self, train):
        classes = [sample[1] for sample in train]
        freqs = Counter(classes)
        freqs = [(k, v) for k, v in dict(freqs).items()]
        normalization = len(train)

        prior = []

        for _, freq in sorted(freqs, key=lambda x: x[0]):
            prior.append(freq / normalization)

        return prior

    def empirical_error(self, test):
        predictions = np.array([self.predict(features) for features, _ in test])
        targets = np.array([target for _, target in test]).astype(np.int)

        return np.sum(predictions != targets) / len(test), confusion_matrix(predictions, targets)

    def gaussian(self, features, mean):
        # optimized for recognition using diagonal pooled covariances
        score = np.exp(-1/2 * ((features - mean) * (1 / self.covariance_matrix)).dot(features - mean))
        return score

    def predict(self, features):
        scores = []

        for mean, class_prior in zip(self.means, self.prior):

            scores.append(self.score(features, mean, class_prior))

        return np.argmax(scores) + 1

    def score(self, features, mean, class_prior):
        class_conditional = self.gaussian(features, mean)

        return class_conditional * class_prior


def read_file(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

        num_classes = int(lines.pop(0))
        num_dims = int(lines.pop(0))
        num_lines_per_sample = int(np.sqrt(num_dims)) + 1
        num_samples = int(math.ceil(len(lines) / num_lines_per_sample))

        dataset = []

        for sample_idx in range(num_samples):
            window = lines[sample_idx * num_lines_per_sample: (sample_idx + 1) * num_lines_per_sample]
            target = int(window.pop(0))
            # join feature lines
            features = " ".join(window)
            # reduce multiple whitespaces to one
            features = " ".join(features.split())
            # create list of features
            features = np.array([float(feature) for feature in features.split(" ")])
            dataset.append((features, target))

    return num_classes, num_dims, dataset

def write_params(covariance_matrix, means, file_name, num_classes, num_dims, prior, pooled=True):
    if pooled:
        with open(file_name, "w") as f:
            items = ["d", str(num_classes), str(num_dims)]
            f.writelines([item + "\n" for item in items])

            for class_idx, (mean, class_prior) in enumerate(zip(means, prior)):
                items = [str(class_idx + 1), str(class_prior), " ".join(mean.astype(str)), " ".join(covariance_matrix.astype(str))]
                f.writelines([item + "\n" for item in items])

if __name__ == '__main__':

    num_classes, num_dims, train = read_file("usps.train")
    _, _, test = read_file("usps.test")

    classifier = BayesClassifier(num_classes, num_dims, train)

    error_rate, confusion_matrix = classifier.empirical_error(test)

    print(f"Error rate: {error_rate}")

    with open("usps_d.error", "w") as f:
        f.write(str(error_rate))

    with open("usps_d.cm", "w") as f:
        for line in confusion_matrix:
            f.write("\t".join(line.astype(str)) + "\n")

    write_params(
        classifier.covariance_matrix, 
        classifier.means, 
        "usps_d.param", 
        num_classes, 
        num_dims, 
        classifier.prior
    )
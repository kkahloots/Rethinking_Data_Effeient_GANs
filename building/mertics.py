import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from tensorflow.keras.applications import VGG16
epsilon=1e-6

# prepare the inception v3 model
def prepare_inception(input_shape):
	return InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)

def prepare_VGG16_model(input_shape):
	 return VGG16(include_top=False, pooling='avg', input_shape=input_shape)


def calculate_is(inception_model, generated_images):
	prdictions = inception_model.predict(inception_preprocess_input(generated_images))

	print('calculating the inception_score mean ...')
	is_scores = calculate_is(prdictions)
	is_mean, is_sigma = np.mean(is_scores, axis=0), np.cov(is_scores, rowvar=False)
	return is_mean, is_sigma

# calculate frechet inception distance
def calculate_fid(inception_model, real_images, generated_images):
	# calculate activations
	act1 = inception_model.predict(inception_preprocess_input(real_images))
	act2 = inception_model.predict(inception_preprocess_input(generated_images))

	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


########################

def calculate_perceptual_path_length(VGG16_model, real_images, generated_images):


    def learned_perceptual_image_patch_similarity(images_a, images_b):
        """LPIPS metric using VGG-16 and Zhang weighting. (https://arxiv.org/abs/1801.03924)
        Takes reference images and corrupted images as an input and outputs the perceptual
        distance between the image pairs.
        """

        # Concatenate images.
        images = tf.concat([images_a, images_b], axis=0)

        # Extract features.
        vgg_features = VGG16_model(images)

        # Normalize each feature vector to unit length over channel dimension.
        normalized_features = []
        for x in vgg_features:
            x = tf.reshape(x, (len(x), 1))
            n = tf.reduce_sum(x ** 2, axis=1, keepdims=True) ** 0.5
            normalized_features.append(x / (n + 1e-10))

        # Split and compute distances.
        diff = [tf.subtract(*tf.split(x, 2, axis=0)) ** 2 for x in normalized_features]

        return np.array(diff)

    def filter_distances_fn(distances):
        # Reject outliers.
        lo = np.percentile(distances, 1, interpolation='lower')
        hi = np.percentile(distances, 99, interpolation='higher')
        filtered_distances = np.extract(np.logical_and(lo <= distances, distances <= hi), distances)
        return filtered_distances

    return filter_distances_fn(learned_perceptual_image_patch_similarity(real_images, generated_images) * (1 / epsilon ** 2))



########################

import time

def batch_pairwise_distances(U, V):
    """ Compute pairwise distances between two batches of feature vectors."""
    # Squared norms of each row in U and V.
    norm_u = np.sum(np.square(U), 1)
    norm_v = np.sum(np.square(V), 1)

    # norm_u as a row and norm_v as a column vectors.
    norm_u = np.reshape(norm_u, [-1, 1])
    norm_v = np.reshape(norm_v, [1, -1])

    # Pairwise squared Euclidean distances.
    D = np.maximum(norm_u - 2*np.matmul(U, V.numpy().T) + norm_v, 0.0)

    return D

class ManifoldEstimator():
    """Finds an estimate for the manifold of given feature vectors."""
    def __init__(self, features, row_batch_size, col_batch_size, nhood_sizes, clamp_to_percentile=None):
        """Find an estimate of the manifold of given feature vectors."""
        num_images = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features

        # Estimate manifold of features by calculating distances to kth nearest neighbor of each sample.
        self.D = np.zeros([num_images, self.num_nhoods], dtype=np.float16)
        distance_batch = np.zeros([row_batch_size, num_images], dtype=np.float16)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0:end1-begin1, begin2:end2] = batch_pairwise_distances(row_batch, col_batch)

            # Find the kth nearest neighbor from the current batch.
            self.D[begin1:end1, :] = np.partition(distance_batch[0:end1-begin1, :], seq, axis=1)[:, self.nhood_sizes]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0  #max_distances  # 0

    def evaluate(self, eval_features, return_realism=False, return_neighbors=False):
        """Evaluate if new feature vectors are in the estimated manifold."""
        num_eval_images = eval_features.shape[0]
        num_ref_images = self.D.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref_images], dtype=np.float16)
        batch_predictions = np.zeros([num_eval_images, self.num_nhoods], dtype=np.int32)

        realism_score = np.zeros([num_eval_images,], dtype=np.float32)
        nearest_indices = np.zeros([num_eval_images,], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0:end1-begin1, begin2:end2] = batch_pairwise_distances(feature_batch, ref_batch)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then the new sample lies on the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0:end1-begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(np.int32)

            nearest_indices[begin1:end1] = np.argmin(distance_batch[0:end1-begin1, :], axis=1)
            realism_score[begin1:end1] = self.D[nearest_indices[begin1:end1], 0] / np.min(distance_batch[0:end1-begin1, :], axis=1)

        if return_realism and return_neighbors:
            return batch_predictions, realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions


def knn_precision_recall_features(ref_features, eval_features, feature_net, nhood_sizes, row_batch_size, col_batch_size):
    """Calculates k-NN precision and recall for two sets of feature vectors."""
    state = dict()
    num_images = ref_features.shape[0]
    num_features = feature_net.output_shape[1]
    state['ref_features'] = ref_features
    state['eval_features'] = eval_features

    # Initialize DistanceBlock and ManifoldEstimators.
    state['ref_manifold'] = ManifoldEstimator(state['ref_features'], row_batch_size, col_batch_size, nhood_sizes)
    state['eval_manifold'] = ManifoldEstimator(state['eval_features'], row_batch_size, col_batch_size, nhood_sizes)

    # Evaluate precision and recall using k-nearest neighbors.
    print('Evaluating k-NN precision and recall with %i samples...' % num_images)
    start = time.time()

    # Precision: How many points from eval_features are in ref_features manifold.
    state['precision'], state['realism_scores'], state['nearest_neighbors'] = state['ref_manifold'].evaluate(state['eval_features'], return_realism=True, return_neighbors=True)
    state['knn_precision'] = state['precision'].mean(axis=0)

    # Recall: How many points from ref_features are in eval_features manifold.
    state['recall'] = state['eval_manifold'].evaluate(state['ref_features'])
    state['knn_recall'] = state['recall'].mean(axis=0)

    elapsed_time = time.time() - start
    print('Done evaluation in: %gs' % elapsed_time)

    return state

def precision_recall_score(VGG16_model, real_images, generated_images):
	ref_features = VGG16_model(real_images)
	eval_features = VGG16_model(generated_images)

	# Calculate precision and recall.
	state = knn_precision_recall_features(ref_features=ref_features, eval_features=eval_features,
										  feature_net=VGG16_model,
										  nhood_sizes=[5], row_batch_size=len(real_images),
										  col_batch_size=len(real_images))

	knn_precision = state['knn_precision'][0]
	knn_recall = state['knn_recall'][0]


	return knn_precision, knn_recall

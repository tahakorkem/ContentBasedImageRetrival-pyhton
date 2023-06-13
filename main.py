import os
from PIL import Image
from numpy import asarray, ndarray, array, delete, save, load
from typing import List, Dict


class Test:
    def __init__(self, pass_count: int, fail_count: int):
        self.pass_count = pass_count
        self.fail_count = fail_count


class Histogram:
    def __init__(self, histogram: ndarray, label: str, path: str):
        self.histogram = histogram
        self.label = label
        self.path = path


histograms: List[Histogram] = []


def main():
    global histograms
    global tests_by_label

    # take an input from user to set train_from_scratch
    train_from_scratch = input("Train from scratch? (y/n): ") == 'y'

    # if train_from_scratch is true, train the system
    # else load the training data from the file
    if train_from_scratch:
        labels = os.listdir('train')
        # iterate through the labels
        for label in labels:
            files = os.listdir(f'train/{label}')
            # iterate through the images in each label
            for file in files:
                print("Training >", label, file)
                train(label, f'train/{label}/{file}')

        # save the training data to the file
        with open('train_data.npy', 'wb') as f:
            save(f, array(histograms))
    else:
        with open('train_data.npy', 'rb') as f:
            histograms = load(f, allow_pickle=True)

    # initialize the tests_by_label dictionary with the labels as keys from histograms
    tests_by_label = {h.label: Test(0, 0) for h in histograms}

    # iterate through the test images
    for file in os.listdir  ('test'):
        test(f'test/{file}')

    # print the results
    print("\nTests by label:")

    # calculate success rate for each label
    for label, t in tests_by_label.items():
        success_rate = t.pass_count / (t.pass_count + t.fail_count) * 100
        print(f'{label} score: {success_rate:.2f}%')

    # calculate overall success rate
    overall_pass_count = sum(t.pass_count for t in tests_by_label.values())
    overall_fail_count = sum(t.fail_count for t in tests_by_label.values())
    overall_success_rate = overall_pass_count / (overall_pass_count + overall_fail_count) * 100

    print(f"\nOverall score: {overall_success_rate:.2f}%")


# get the lbp histogram from an grayscale form of an image given by path
def lbp_histogram_from_image(path: str) -> ndarray:
    # read the image
    image_arr = read_image(path)
    # convert to grayscale
    grayscale_image_arr = convert_to_grayscale(image_arr)
    # create lbp
    lbp_image_arr = create_lbp(grayscale_image_arr)
    # generate histogram
    histogram = generate_histogram(lbp_image_arr)
    return histogram


# train the system with a label and a path to an image
def train(label: str, path: str):
    h = lbp_histogram_from_image(path)
    # add the histogram to the training data
    histograms.append(Histogram(h, label, path))


# read an image from a path
def read_image(path: str) -> ndarray:
    image = Image.open(path)
    return asarray(image)


# convert an rgb image to grayscale
def convert_to_grayscale(arr: ndarray) -> ndarray:
    # convert to YCbCr
    # Y = 0.299 R + 0.587 G + 0.114 B

    Y = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).round().astype('uint8')

    return Y


# create a lbp image from a grayscale image
def create_lbp(arr):
    # create a new array with the same shape as the original array
    # and fill it with zeros
    lbp_arr = asarray([[0] * arr.shape[1]] * arr.shape[0])
    # iterate through the array
    for i in range(1, arr.shape[0] - 1):
        for j in range(1, arr.shape[1] - 1):
            center = arr[i, j]
            # create the 3x3 matrix flattened to a 1d array then remove the center pixel
            matrix = delete(array(arr[i - 1:i + 2, j - 1:j + 2]).flatten(), 4)  # 1 2 3 4 6 7 8 9
            # create the binary array
            binary_arr = asarray(['1' if x > center else '0' for x in matrix])
            # convert the binary array to decimal
            lbp_arr[i, j] = int(''.join(binary_arr), 2)
    # crop the array to remove the border
    lbp_arr = lbp_arr[1:-1, 1:-1].astype('uint8')
    return lbp_arr


# generate a histogram from a lbp image
def generate_histogram(arr):
    # create a histogram with 256 bins
    # iterate through the array
    # increment the bin that corresponds to the pixel value
    histogram = [0] * 256
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            histogram[arr[i, j]] += 1
    # normalize the histogram
    histogram = [x / (arr.shape[0] * arr.shape[1]) for x in histogram]
    return asarray(histogram)


tests_by_label: Dict[str, Test] = {}


# test the image given by path with the training data
def test(path: str):
    label = path.split('/')[1].split('_')[0]
    # get the histogram for the test image
    hist = lbp_histogram_from_image(path)
    # calculate manhattan distances for each histogram of histograms
    distances = map(lambda h: [h.label, h.path, manhattan_distance(hist, h.histogram)], histograms)
    # find the minimum three distances by sorting the distances by the second element (the distance)
    distances = sorted(distances, key=lambda x: x[2])[:3]
    # print the minimum three distances
    print("Testing >", path, "--the closest images: ", distances)
    # check if the label is in the minimum three distances
    if label in [x[0] for x in distances]:
        print(">>Test passed")
        tests_by_label[label].pass_count += 1
    else:
        print(">>Test failed")
        tests_by_label[label].fail_count += 1


# calculate the manhattan distance between two histograms
def manhattan_distance(h1: ndarray, h2: ndarray) -> float:
    return sum(abs(v1 - v2) for v1, v2 in zip(h1, h2))


if __name__ == '__main__':
    main()

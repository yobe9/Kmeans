import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy.linalg

# targil code
image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.loadtxt(centroids_fname)

orig_pixels = plt.imread(image_fname)
pixels = orig_pixels.astype(float) / 255.
pixels = pixels.reshape(-1, 3)


# function which receives a pixel, iterates the centroids and calculate
# the min distance and add the pixel to the centroid's cluster
# rely on there is a cluster
def dist_pixel_to_cetroid(pixel):
    min_dist = numpy.linalg.norm(centroids[0] - pixel)
    cent_num = 0
    min_cent = 0
    for cent in centroids:
        if numpy.linalg.norm(cent - pixel) < min_dist:
            min_dist = numpy.linalg.norm(cent - pixel)
            min_cent = cent_num
        cent_num += 1
    clusters[min_cent].append(pixel)


# return the average rounded
def avg_pixels(cluster):
    avg = np.mean(cluster, axis=0)
    n = 0
    for n in range(len(avg)):
        avg[n] = avg[n].round(4)
    return np.mean(cluster, axis=0).round(4)

# checks if the old centroids are equals to the new ones
def isCentoirdsEquals(old, new):
    return np.array_equal(old, new)

# write format of the centroids
def centroidPrinter(centPrint, ep):
    println = f"[iter {ep}]:"
    for cen in centPrint:
        println += np.array2string(cen)
        println += ","
    println = println[0:-1] + "\n"
    output.write(println)


if __name__ == '__main__':
    # checks there are centroids
    if len(z) <= 0:
        print("error")
        exit()

    # writing output file
    output = open(out_fname, "w")

    # opening txt file for the output
    centroids = []
    clusters = []
    # adding the centroids into the list
    for cent in z:
        centroids.append(cent)
        clusters.append([])

    # iterating 20 times or until convergence
    for epoch in range(20):
        for pix in pixels:
            dist_pixel_to_cetroid(pix)

        # craeting np array for each centorid's cluster
        for i in range(len(clusters)):
            clusters[i] = np.array(clusters[i])

        # saving the old centroids values
        oldCentroids = centroids.copy()
        # calculating the avg and set it to the centroid location
        for j in range(len(centroids)):
            if len(clusters[j]) != 0:
                centroids[j] = avg_pixels(clusters[j])

        centroidPrinter(centroids, epoch)

        # checking if centroids had change
        if isCentoirdsEquals(oldCentroids, centroids):
            # print("done")
            break

        # resetting the clusters
        for k in range(len(clusters)):
            clusters[k] = []

    # closing output file
    output.close()

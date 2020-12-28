import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
import keras
from minisom import MiniSom
import skfuzzy

s1 = np.loadtxt("s1.txt")
breast = np.loadtxt("breast.txt")

breast_copy = breast
'''
Najlepszy efekt podziału punktów daje algorytm k średnich, następnie sieć SOM, a na końcu sieć neuronowa.
'''

def c_mean(data, clusters):
    going = True
    counter = 0
    points = []
    indexes = None
    while going:
        counter += 1
        points = []
        centers = []
        closest_cluster = []
        for cluster in range(len(clusters)):
            the_closest_one = []
            for dimension in range(data.shape[1]):
                the_closest_one.append((data[:, dimension] - clusters[cluster, dimension]))
            odleglosci = np.zeros(len(the_closest_one[0]))
            the_closest_one = np.asarray(the_closest_one)
            the_closest_one = np.transpose(the_closest_one)
            for difference in range(the_closest_one.shape[1]):
                odleglosci += np.power(the_closest_one[:, difference], 2)
            odleglosci = np.sqrt(odleglosci)
            closest_cluster.append(odleglosci)
        closest_cluster = np.asarray(closest_cluster)
        closest_cluster = np.transpose(closest_cluster)
        indexes = np.argmin(closest_cluster, axis=1)
        for cluster in range(len(clusters)):
            centers.append(np.mean(data[indexes == cluster], axis=0))
            points.append(data[indexes == cluster, :])
        for center in range(len(centers)):
            for dimension in range(len(centers[0])):
                if np.abs(centers[center][dimension] - clusters[center, dimension]) > 1: counter = 0
        clusters = np.asarray(centers)
        if counter > 20: going = False
    return clusters, points, indexes

s1_clusters = np.asarray([[167041, 887004],
                          [451995, 848685],
                          [639238, 922500],
                          [869000, 755000],
                          [921000, 558000],
                          [785000, 310000],
                          [791000, 74000],
                          [614000, 695000],
                          [630000, 369000],
                          [378000, 602000],
                          [123000, 600000],
                          [369000, 449000],
                          [106000, 377000],
                          [244000, 189000],
                          [543000, 145000]])

breast_clusters = np.asarray([[ 1,  1,  1,  1,  1,  1,  1,  1,  1],
                              [10, 10, 10, 10, 10, 10, 10, 10, 10]])

fig1 = plt.figure()
fig1.suptitle("s1")

ax1 = fig1.add_subplot(221)
ax2 = fig1.add_subplot(222)
ax3 = fig1.add_subplot(223)
ax4 = fig1.add_subplot(224)

s1_centers, s1_points, s1_indexes = c_mean(s1, s1_clusters)
s1_indexes.sort()

colours = []
for cluster_points in range(len(s1_points)):
    colours.append((cluster_points/len(s1_points), np.random.rand(), np.random.rand()))
    points_to_draw = s1_points[cluster_points]
    ax1.plot(points_to_draw[:, 0], points_to_draw[:, 1], 'ok', color=colours[cluster_points])
ax1.plot(s1_centers[:,0], s1_centers[:,1], 'ok', color='red')
ax1.set_title("k-means")

breast_centers, breast_points, breast_indexes = c_mean(breast, breast_clusters)

breast_points0 = np.asarray(breast_points[0])
breast_points1 = np.asarray(breast_points[1])

fig2 = plt.figure()
fig2.suptitle("breast")

ax11 = fig2.add_subplot(431, projection='3d')
ax12 = fig2.add_subplot(432, projection='3d')
ax13 = fig2.add_subplot(433, projection='3d')
ax14 = fig2.add_subplot(434, projection='3d')
ax15 = fig2.add_subplot(435, projection='3d')
ax16 = fig2.add_subplot(436, projection='3d')
ax17 = fig2.add_subplot(437, projection='3d')
ax18 = fig2.add_subplot(438, projection='3d')
ax19 = fig2.add_subplot(439, projection='3d')
ax110 = fig2.add_subplot(4, 3, 10, projection='3d')
ax111 = fig2.add_subplot(4, 3, 11, projection='3d')
ax112 = fig2.add_subplot(4, 3, 12, projection='3d')


'''
Na czerwono są zaznaczone centra klastrów, a w okół nich na zielono punkty należące do jednego klastra,
a na niebiesko punkty należące do drugiego.
'''

ax11.scatter3D(breast_points0[:, 0], breast_points0[:, 1], breast_points0[:, 2], color='green')
ax11.scatter3D(breast_points1[:, 0], breast_points1[:, 1], breast_points1[:, 2], color='blue')
ax11.scatter3D(breast_centers[:, 0], breast_centers[:, 1], breast_centers[:, 2], color="red")
ax11.set_title("k-means (0, 1, 2)")

ax12.scatter3D(breast_points0[:, 3], breast_points0[:, 4], breast_points0[:, 5], color='green')
ax12.scatter3D(breast_points1[:, 3], breast_points1[:, 4], breast_points1[:, 5], color='blue')
ax12.scatter3D(breast_centers[:, 3], breast_centers[:, 4], breast_centers[:, 5], color="red")
ax12.set_title("k-means (3, 4, 5)")

ax13.scatter3D(breast_points0[:, 6], breast_points0[:, 7], breast_points0[:, 8], color='green')
ax13.scatter3D(breast_points1[:, 6], breast_points1[:, 7], breast_points1[:, 8], color='blue')
ax13.scatter3D(breast_centers[:, 6], breast_centers[:, 7], breast_centers[:, 8], color="red")
ax13.set_title("k-means (6, 7, 8)")

'''
Uczenie sieci i rysowanie przynależności punktów s1.txt zależnie od predykcji nauczonej sieci

Aby od nowa nauczyć sieć trzeba odkomentować kawałek kodu poniżej
'''

data1 = np.zeros((s1.shape[0], 17), dtype=float)

sum = 0
for i in range(len(s1_points)):
    data1[sum:sum + len(s1_points[i]), 0:2] = s1_points[i]
    sum += len(s1_points[i])

for rrr in range(s1.shape[0]):
    data1[rrr, s1_indexes[rrr]+2] = 1

for i in range(10):
    np.random.shuffle(data1)

x1 = data1[:, 0:2]

x1_train = data1[:int(0.8*data1.shape[0]), 0:2]
y1_train = data1[:int(0.8*data1.shape[0]), 2:]

x1_test = data1[int(0.8*data1.shape[0]):int(0.9*data1.shape[0]), 0:2]
y1_test = data1[int(0.8*data1.shape[0]):int(0.9*data1.shape[0]), 2:]

x1_verify = data1[int(0.9*data1.shape[0]):, 0:2]
y1_verify = data1[int(0.9*data1.shape[0]):, 2:]

model = Sequential()
model.add(layers.BatchNormalization())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(15, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x1_train, y1_train, validation_data=(x1_test, y1_test), epochs=1000, batch_size=30, verbose=1)

scores = model.evaluate(x1_verify, y1_verify, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# model.save("s1_network.h5")
#
# model = keras.models.load_model("s1_network.h5")

y1 = model.predict(x1)

s1_final_indexes = np.argmax(y1, axis=1)

final_s1_points = []

for i in range(15):
    final_s1_points.append(x1[s1_final_indexes == i, :])

for final_s1_cluster_points in range(len(final_s1_points)):
    points_to_draw = final_s1_points[final_s1_cluster_points]
    ax2.plot(points_to_draw[:, 0], points_to_draw[:, 1], 'ok', color=colours[final_s1_cluster_points])
ax2.set_title("keras")

'''
Uczenie sieci i rysowanie przynależności punktów breast.txt zależnie od predykcji nauczonej sieci

Aby od nowa nauczyć sieć trzeba odkomentować kawałek kodu poniżej
'''

data2 = np.zeros((breast_copy.shape[0], 11), dtype=float)

sum2 = 0
for i in range(len(breast_points)):
    data2[sum2:sum2 + len(breast_points[i]), 0:9] = breast_points[i]
    sum2 += len(breast_points[i])

for rrr in range(breast_copy.shape[0]):
    data2[rrr, breast_indexes[rrr]+9] = 1

for i in range(10):
    np.random.shuffle(data2)

x2 = data2[:, 0:9]

x2_train = data2[:int(0.8*data2.shape[0]), 0:9]
y2_train = data2[:int(0.8*data2.shape[0]), 9:]

x2_test = data2[int(0.8*data2.shape[0]):int(0.9*data2.shape[0]), 0:9]
y2_test = data2[int(0.8*data2.shape[0]):int(0.9*data2.shape[0]), 9:]

x2_verify = data2[int(0.9*data2.shape[0]):, 0:9]
y2_verify = data2[int(0.9*data2.shape[0]):, 9:]

model2 = Sequential()
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(32, activation='relu'))
model2.add(layers.Dense(16, activation='relu'))
model2.add(layers.Dense(2, activation='softmax'))

opt2 = keras.optimizers.Adam(learning_rate=0.001)
model2.compile(loss='categorical_crossentropy', optimizer=opt2, metrics=['accuracy'])

model2.fit(x2_train, y2_train, validation_data=(x2_test, y2_test), epochs=1000, batch_size=36, verbose=1)

scores2 = model2.evaluate(x2_verify, y2_verify, verbose=1)
print("%s: %.2f%%" % (model2.metrics_names[1], scores2[1]*100))

# model2.save("breast_network.h5")
#
# model2 = keras.models.load_model("breast_network.h5")

y2 = model2.predict(x2)

breast_final_indexes = np.argmax(y2, axis=1)

final_breast_points = []

for i in range(2):
    final_breast_points.append(x2[breast_final_indexes == i, :])

final_breast_points0 = np.asarray(final_breast_points[0])
final_breast_points1 = np.asarray(final_breast_points[1])

'''
Na zielono i niebiesko są punkty o dwóch różnych przynależnościach.
'''

ax14.scatter3D(final_breast_points0[:, 0], final_breast_points0[:, 1], final_breast_points0[:, 2], color='green')
ax14.scatter3D(final_breast_points1[:, 0], final_breast_points1[:, 1], final_breast_points1[:, 2], color='blue')
ax14.set_title("keras (0, 1, 2)")

ax15.scatter3D(final_breast_points0[:, 3], final_breast_points0[:, 4], final_breast_points0[:, 5], color='green')
ax15.scatter3D(final_breast_points1[:, 3], final_breast_points1[:, 4], final_breast_points1[:, 5], color='blue')
ax15.set_title("keras (3, 4, 5)")

ax16.scatter3D(final_breast_points0[:, 6], final_breast_points0[:, 7], final_breast_points0[:, 8], color='green')
ax16.scatter3D(final_breast_points1[:, 6], final_breast_points1[:, 7], final_breast_points1[:, 8], color='blue')
ax16.set_title("keras (6, 7, 8)")

'''
Algorytm SOM dla s1.txt
'''

np.random.shuffle(s1)
s1 = (s1 - np.mean(s1, axis=0)) / np.std(s1, axis=0)

som_shape = (1, 15)
som = MiniSom(som_shape[0], som_shape[1], s1.shape[1], sigma=0.3, learning_rate=0.9)

som.train_batch(s1, 100000, verbose=True)

coords = np.array([som.winner(x) for x in s1]).T

som_cluster_indexes = np.ravel_multi_index(coords, som_shape)

for c in np.unique(som_cluster_indexes):
    ax3.plot(s1[som_cluster_indexes == c, 0], s1[som_cluster_indexes == c, 1], 'ok', color=colours[c])
ax3.set_title("SOM")

'''
Algorytm SOM dla breast.txt
'''

np.random.shuffle(breast)
breast = (breast - np.mean(breast, axis=0)) / np.std(breast, axis=0)

som_shape2 = (1, 2)
som2 = MiniSom(som_shape2[0], som_shape2[1], breast.shape[1], sigma=0.3, learning_rate=0.15)

som2.train_batch(breast, 100000, verbose=True)

coords2 = np.array([som2.winner(xz) for xz in breast]).T

som_cluster_indexes2 = np.ravel_multi_index(coords2, som_shape2)

ax17.scatter3D(breast[som_cluster_indexes2 == 0, 0], breast[som_cluster_indexes2 == 0, 1],
               breast[som_cluster_indexes2 == 0, 2], color='green')
ax17.scatter3D(breast[som_cluster_indexes2 == 1, 0], breast[som_cluster_indexes2 == 1, 1],
               breast[som_cluster_indexes2 == 1, 2], color='blue')
ax17.set_title("SOM (0, 1, 2)")

ax18.scatter3D(breast[som_cluster_indexes2 == 0, 3], breast[som_cluster_indexes2 == 0, 4],
               breast[som_cluster_indexes2 == 0, 5], color='green')
ax18.scatter3D(breast[som_cluster_indexes2 == 1, 3], breast[som_cluster_indexes2 == 1, 4],
               breast[som_cluster_indexes2 == 1, 5], color='blue')
ax18.set_title("SOM (3, 4, 5)")

ax19.scatter3D(breast[som_cluster_indexes2 == 0, 6], breast[som_cluster_indexes2 == 0, 7],
               breast[som_cluster_indexes2 == 0, 8], color='green')
ax19.scatter3D(breast[som_cluster_indexes2 == 1, 6], breast[som_cluster_indexes2 == 1, 7],
               breast[som_cluster_indexes2 == 1, 8], color='blue')
ax19.set_title("SOM (6, 7, 8)")


'''
Logika rozmyta dla s1
'''

print("Fuzzy dla s1...")

rozmyte_s1_trening = skfuzzy.cmeans(x1_train.transpose(), s1_centers.shape[0], 2, 0.001, 10000)
rozmyte_s1 = skfuzzy.cmeans_predict(x1.transpose(), rozmyte_s1_trening[0], 2, 0.001, 10000)
rozmyte_indeksy_s1 = np.argmax(rozmyte_s1[0].transpose(), axis=1)

final_s1_points_rozmyte = []

for i in range(15):
    final_s1_points_rozmyte.append(x1[rozmyte_indeksy_s1 == i, :])

for final_s1_cluster_points in range(len(final_s1_points)):
    points_to_draw = final_s1_points[final_s1_cluster_points]
    ax4.plot(points_to_draw[:, 0], points_to_draw[:, 1], 'ok', color=colours[final_s1_cluster_points])
ax4.set_title("fuzzy")

'''
Logika rozmyta dla breast
'''

print("Fuzzy dla breast...")

rozmyte_breast_trening = skfuzzy.cmeans(x2_train.transpose(), breast_centers.shape[0], 2, 0.001, 10000)
rozmyte_breast = skfuzzy.cmeans_predict(x2.transpose(), rozmyte_breast_trening[0], 2, 0.001, 10000)
breast_final_fuz_indexes = np.argmax(rozmyte_breast[0].transpose(), axis=1)

final_breast_points_fuz = []

for i in range(2):
    final_breast_points_fuz.append(x2[breast_final_fuz_indexes == i, :])

final_breast_points0 = np.asarray(final_breast_points_fuz[0])
final_breast_points1 = np.asarray(final_breast_points_fuz[1])

'''
Na zielono i niebiesko są punkty o dwóch różnych przynależnościach.
'''

ax110.scatter3D(final_breast_points0[:, 0], final_breast_points0[:, 1], final_breast_points0[:, 2], color='green')
ax110.scatter3D(final_breast_points1[:, 0], final_breast_points1[:, 1], final_breast_points1[:, 2], color='blue')
ax110.set_title("fuzzy (0, 1, 2)")

ax111.scatter3D(final_breast_points0[:, 3], final_breast_points0[:, 4], final_breast_points0[:, 5], color='green')
ax111.scatter3D(final_breast_points1[:, 3], final_breast_points1[:, 4], final_breast_points1[:, 5], color='blue')
ax111.set_title("fuzzy (3, 4, 5)")

ax112.scatter3D(final_breast_points0[:, 6], final_breast_points0[:, 7], final_breast_points0[:, 8], color='green')
ax112.scatter3D(final_breast_points1[:, 6], final_breast_points1[:, 7], final_breast_points1[:, 8], color='blue')
ax112.set_title("fuzzy (6, 7, 8)")

plt.show()
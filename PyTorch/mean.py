import numpy as np
val = np.reshape(image[:,:,0], -1)

count = 0
mean = 0

for i, file in enumerate(tqdm(first)):
    image = cv2.imread(file)
        val = np.reshape(image[:,:,0], -1)
        img_mean = np.mean(val)
        img_std = np.std(val)

        count = count + 1

print('mean: ', img_mean)
print('std: ',img_std)

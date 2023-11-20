from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.metrics import CategoricalAccuracy

def train_val_test_split(X, y, random_state = 42):
    X_train, X1, y_train, y1 = train_test_split(X, y, test_size=0.4, random_state = random_state)
    X_val, X_test, y_val, y_test = train_test_split(X1, y1, test_size=0.5, random_state = random_state)
    return(X_train, X_val, X_test, y_train, y_val, y_test)

#----SHEAR----
def random_shear_image(input_image, shear_range=(-0.2, 0.2)):
    input_image = 255*input_image
    # Get the dimensions of the input image
    image_height, image_width = input_image.shape

    # Generate random shear factors for x and y directions
    shear_x = np.random.uniform(shear_range[0], shear_range[1])
    shear_y = np.random.uniform(shear_range[0], shear_range[1])

    # Create a grid of coordinates for the original image
    x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))

    # Apply shear transformation to the coordinates
    sheared_x = x + shear_x * y
    sheared_y = y + shear_y * x

    # Initialize the sheared image
    sheared_image = np.zeros((image_height, image_width), dtype=np.uint8)

    # Interpolate pixel values from the original image to sheared coordinates
    sheared_image = np.zeros_like(input_image)

    # Use scipy's map_coordinates for efficient interpolation
    map_coordinates(input_image, (sheared_y, sheared_x), output=sheared_image, order=1, mode='nearest')

    return (sheared_image/255)

def augment_dataset_with_shear(images, y_train, proportion = 0.2, shear_range=(-0.2, 0.2)):

    augmented_images = []

    for image in images:
        augmented_image = random_shear_image(image, shear_range)
        augmented_images.append(augmented_image)
    
    augmented_dataset = np.array(augmented_images)
    X_aug, _, y_aug, _ = train_test_split(augmented_dataset, y_train, test_size=1-proportion, random_state=42)

    return(X_aug,y_aug) 

#----GAUSSIAN BLUR----
def add_gaussian_blur(image, kernel_size=5, sigma=1.0):
    image = image*255
    # Create a Gaussian kernel
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (kernel_size-1)/2)**2 + (y - (kernel_size-1)/2)**2) / (2*sigma**2)),
        (kernel_size, kernel_size)
    )
    kernel /= np.sum(kernel)  # Normalize the kernel

    # Apply convolution using FFT (Fast Fourier Transform)
    blurred_image = np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(kernel, s=image.shape))

    return (np.real(blurred_image)/255)

def augment_dataset_with_blur(images, y_train, proportion = 0.2, kernel_size = 5, sigma = 1.0):

    augmented_images = []

    for image in images:
        augmented_image = add_gaussian_blur(image, kernel_size, sigma)
        augmented_images.append(augmented_image)

    augmented_dataset = np.array(augmented_images)
    X_aug, _, y_aug, _ = train_test_split(augmented_dataset, y_train, test_size=1-proportion, random_state=42)

    return(X_aug,y_aug) 

#----NOISE----
def add_gaussian_noise(image, mean=0, std_dev=25):
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = image*255 + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Clip values to the valid grayscale range (0-255)
    noisy_image = noisy_image.astype(np.uint8)/255  # Convert the result back to an 8-bit unsigned integer
    return noisy_image

def augment_dataset_with_noise(images, y_train, proportion = 0.2, mean = 0, std_dev=25):

    augmented_images = []

    for image in images:
        augmented_image = add_gaussian_noise(image)
        augmented_images.append(augmented_image)

    augmented_dataset = np.array(augmented_images)
    X_aug, _, y_aug, _ = train_test_split(augmented_dataset, y_train, test_size=1-proportion, random_state=42)

    return(X_aug,y_aug) 

#----INVERTION----
def invert(image):

    adjusted_image = 255 - image*255
    return (np.clip(adjusted_image, 0, 255).astype(np.uint8)/255)

def augment_dataset_with_invertion(images, y_train, proportion = 0.2, min_brightness=80, max_brightness=100):

    augmented_images = []

    for image in images:
        augmented_image = invert(image)
        augmented_images.append(augmented_image)

    augmented_dataset = np.array(augmented_images)
    X_aug, _, y_aug, _ = train_test_split(augmented_dataset, y_train, test_size=1-proportion, random_state=42)

    return(X_aug,y_aug)

#----CONTRAST----
def lower_contrast(image, contrast_factor=0.5):

    adjusted_image = contrast_factor * (image*255 - 128) + 128
    return (np.clip(adjusted_image, 0, 255).astype(np.uint8)/255)

def augment_dataset_with_contrast(images, y_train, proportion=0.2, contrast_factor=0.5):

    augmented_images = []

    for image in images:

        augmented_image = lower_contrast(image, contrast_factor * np.random.uniform(0.6, 1.4))
        augmented_images.append(augmented_image)

    augmented_dataset = np.array(augmented_images)
    X_aug, _, y_aug, _ = train_test_split(augmented_dataset, y_train, test_size=1-proportion, random_state=42)

    return(X_aug,y_aug)

#----BRIGHTNESSS----
def adjust_brightness(image, brightness_factor):
    
    adjusted_image = image*255 + brightness_factor

    return (np.clip(adjusted_image, 0, 255).astype(np.uint8)/255)

def augment_dataset_with_brightness(images, y_train, proportion = 0.2, min_brightness=80, max_brightness=100):
    
    augmented_images = []

    for image in images:
        # Randomly choose a brightness adjustment factor within the specified range
        brightness_factor = np.random.uniform(min_brightness, max_brightness)
        if (np.random.uniform(0, 1) < 0.5):
          brightness_factor *= -1
        augmented_image = adjust_brightness(image, brightness_factor)
        augmented_images.append(augmented_image)

    augmented_dataset = np.array(augmented_images)
    X_aug, _, y_aug, _ = train_test_split(augmented_dataset, y_train, test_size=1-proportion, random_state=42)

    return(X_aug,y_aug)

#----MERGE AND SHUFFLE----
def merge_and_shuffle(X_train, y_train, X_aug, y_aug):
    # Assuming X1_train, X2_train, y1_train, and y2_train are numpy arrays
    # If not, you can convert them using np.array(your_list)

    # Reshape the image data to be 1D arrays
    X1_train_flat = X_train.reshape(-1, 50 * 50)
    X2_train_flat = X_aug.reshape(-1, 50 * 50)

    # Merge the datasets horizontally (along columns)
    X_combined = np.vstack((X1_train_flat, X2_train_flat))
    y_combined = np.hstack((y_train, y_aug))

    # Combine X_combined and y_combined
    combined_data = np.column_stack((X_combined, y_combined))

    # Shuffle the combined dataset
    np.random.shuffle(combined_data)

    # Separate the shuffled data back into X and y
    X_shuffled = combined_data[:, :-1].reshape(-1, 50, 50)  # Reshape back to 3D image data
    y_shuffled = combined_data[:, -1]

    # Now, X_shuffled and y_shuffled contain your merged and shuffled image data.
    return(X_shuffled, y_shuffled)


def merge_and_shuffle_for_one_hot(X_train, y_train, X_aug, y_aug):
    # Assuming X1_train, X2_train, y1_train, and y2_train are numpy arrays
    # If not, you can convert them using np.array(your_list)

    # Reshape the image data to be 1D arrays
    X1_train_flat = X_train.reshape(-1, 50 * 50)
    X2_train_flat = X_aug.reshape(-1, 50 * 50)

    # Merge the datasets horizontally (along columns)
    X_combined = np.vstack((X1_train_flat, X2_train_flat))

    y_combined = np.vstack((y_train, y_aug))

    # Combine X_combined and y_combined
    combined_data = np.column_stack((X_combined, y_combined))

    # Shuffle the combined dataset
    np.random.shuffle(combined_data)
    
    print(combined_data[:,:-5].shape)
    
    X_shuffled = combined_data[:, :-5].reshape(-1, 50, 50)  # Reshape back to 3D image data
    y_shuffled = combined_data[:, -5:]

    # Now, X_shuffled and y_shuffled contain your merged and shuffled image data.
    return X_shuffled, y_shuffled

#----PLOTTING----
def plot_image(image, label):
    """
    Plots a grayscale image with a size of 50x50.
    parameters: i - i-th image
    """
    plt.imshow(image, cmap='gray',vmin=0, vmax=255)
    plt.axis('off')
    plt.title(label)
    plt.show()
    
# Define your custom plot_image() function
def plot_images(image, label, ax):
    ax.imshow(image, cmap='gray',vmin=0, vmax=1)
    ax.axis('off')
    ax.set_title(str(label))

def plot10(X, y, offset=0):
    # Create a 2x5 grid of subplots
    fig, axs = plt.subplots(2, 5, figsize=(12, 6))

    for i in range(2):
        for j in range(5):
            index = i * 5 + j
            if index < 10:
                plot_images(X[index+offset], y[index+offset], axs[i, j])

    # Adjust subplot spacing and display the figure
    plt.tight_layout()
    plt.show()
#----AUGMENTATION----
def augment(X_train, y_train, datagen):
    # Reshape the image to (1, height, width, channels) for ImageDataGenerator
    X_train_reshaped = X_train.reshape((X_train.shape[0], 50, 50, 1))

    # Generate augmented images
    augmented_images = datagen.flow(X_train_reshaped, y_train)

    # Initialize lists for X and y batches
    X_batch_list = []
    y_batch_list = []

    # Collect batches into X and y lists
    for _ in range(len(augmented_images)):
        X_batch, y_batch = augmented_images.next()
        X_batch_list.append(X_batch)
        y_batch_list.append(y_batch)

    # Concatenate X and y batches into single NumPy arrays
    X_batches = np.concatenate(X_batch_list, axis=0)
    y_batches = np.concatenate(y_batch_list, axis=0)

    X_aug = X_batches.reshape(X_train.shape[0], 50, 50)
    y_aug = y_batches
    return (X_aug, y_aug)

#----MIXUP----
def mixup_data(X, y, mixup_ratio, size=0.2):
    assert len(X) == len(y), "X and y must have the same length"

    # Create a dictionary that groups indices by their corresponding labels
    label_indices = {}
    for idx, label in enumerate(y):
        label = tuple(label)  # Convert the label to a hashable tuple
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(idx)

    X_mixed = []
    y_mixed = []

    for _ in range(int(size * len(X))):
        # Randomly select two different label groups
        l1, l2 = np.random.choice((0,1,2,3,4), size=2, replace=False)
        label_group_1 = np.zeros(5, dtype=int)
        label_group_2 = np.zeros(5, dtype=int)
        label_group_1[l1] = 1
        label_group_2[l2] = 1

        # Randomly select indices from the two selected label groups
        i = np.random.choice(label_indices[tuple(label_group_1)])
        j = np.random.choice(label_indices[tuple(label_group_2)])

        # Randomly sample the mixup ratio λ from a Beta distribution
        lambda_value = np.random.beta(mixup_ratio, mixup_ratio)

        # Perform mixup for the features
        mixed_features = lambda_value * X[i] + (1 - lambda_value) * X[j]
        X_mixed.append(mixed_features)

        # Perform mixup for the labels (assuming one-hot encoding)
        mixed_labels = lambda_value * y[i] + (1 - lambda_value) * y[j]
        y_mixed.append(mixed_labels)

    return np.array(X_mixed), np.array(y_mixed)

#----TRAIN AND EVAL----
def train_and_eval_pipe(model, X_train, y_train, X_val, y_val, epochs_number = 10):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=3e-4*(0.2**2))
    #Train the model
    history = model.fit(X_train, y_train, batch_size=256,
                      validation_data = (X_val, y_val),epochs=epochs_number,
                      verbose = 0, callbacks=[reduce_lr])

     # Print the Train and val accuracies
    m = CategoricalAccuracy()
    y_pred_train = model.predict(X_train, verbose = 0)
    m.update_state(y_train,y_pred_train)
    print("Train set accuracy:", m.result().numpy())
    m.reset_state()
    y_pred_val = model.predict(X_val, verbose = 0)
    m.update_state(y_val, y_pred_val)
    print("Validation set accuracy:", m.result().numpy())

    # Plot training history
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def test_set_accuracy(model, X_test, y_test):
    m = CategoricalAccuracy()
    y_pred_test = model.predict(X_test, verbose = 0)
    m.update_state(y_test, y_pred_test)
    print("Test set accuracy:", m.result().numpy())
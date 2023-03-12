import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


from google.colab import drive
drive.mount('/content/gdrive')


# Load the annotations file
# annotations_file = '/content/gdrive/MyDrive/Awone/dataset/annotations/annotations_frames_bbox_rebound_2K.xlsx'
annotations_file = '/content/gdrive/MyDrive/awone/dataset/annotations/annotations_frames_bbox_rebound_2K.xlsx'
df = pd.read_excel(annotations_file)


# Load the images and store them in a numpy array
image_dir = '/content/gdrive/MyDrive/awone/dataset/image_frames_rebound_2K/'
num_frames = df['frame'].nunique()
print(num_frames)
image_size = (256, 256)
images = np.zeros((num_frames,) + image_size + (3,), dtype=np.uint8)

# Enhancing the code to make it load and resize images faster
import multiprocessing as mp
import time

start_time = time.time()

def load_and_resize_image(frame):
    image_file = image_dir + "frame_" + str(frame).zfill(4) + '.png'
    image = cv2.imread(image_file)
    if image is None or image.shape == ():
        print(f"Error loading image {image_file}")
        return None
    image = cv2.resize(image, image_size)
    return image

# Create a pool of worker processes
pool = mp.Pool(processes=mp.cpu_count())

# Load and resize all images in parallel
results = pool.map(load_and_resize_image, range(num_frames))

# Close the pool to release resources
pool.close()

# Combine the results into a numpy array
images = np.array(results)

end_time = time.time()

print("Time taken: ", end_time - start_time, "seconds")


# Normalize the bounding box coordinates to the range [0, 1]
df[['x1', 'y1', 'x2', 'y2']] /= image_size[0]

# Split the data into train, validation, and test sets
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
train_size = int(num_frames * train_ratio)
val_size = int(num_frames * val_ratio)
test_size = num_frames - train_size - val_size

train_images = images[:train_size]
train_bbox = df[df['frame'] < train_size][['x1', 'y1', 'x2', 'y2']].values

val_images = images[train_size:train_size+val_size]
val_bbox = df[(df['frame'] >= train_size) & (df['frame'] < train_size+val_size)][['x1', 'y1', 'x2', 'y2']].values

test_images = images[train_size+val_size:]
test_bbox = df[df['frame'] >= train_size+val_size][['x1', 'y1', 'x2', 'y2']].values



print("train_images",train_images[0])
print("train_bbox",train_bbox[0])
print("val_images",val_images[0])
print("val_bbox",val_bbox[0])
print("test_images",test_images[0])
print("test_bbox",test_bbox[0])


train_df = df[df['frame'] < train_size]
val_df = df[(df['frame'] >= train_size) & (df['frame'] < train_size+val_size)]
test_df = df[df['frame'] >= train_size+val_size]
train_df

print(train_df['frame'].dtype)
print(val_df['frame'].dtype)
print(test_df['frame'].dtype)

train_df.loc[:, 'frame'] = train_df['frame'].astype(str).str.zfill(4)
val_df.loc[:, 'frame'] = val_df['frame'].astype(str).str.zfill(4)
test_df.loc[:, 'frame'] = test_df['frame'].astype(str).str.zfill(4)

print(train_df['frame'].dtype)
print(val_df['frame'].dtype)
print(test_df['frame'].dtype)

train_df


# add a string before and after each element in the 'frame' column
train_df['frame'] = train_df['frame'].apply(lambda x: 'frame_' + x + '.png')
val_df['frame'] = val_df['frame'].apply(lambda x: 'frame_' + x + '.png')
test_df['frame'] = test_df['frame'].apply(lambda x: 'frame_' + x + '.png')

train_df
val_df
test_df


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_dir,
    x_col='frame',
    y_col=['x1', 'y1', 'x2', 'y2'],
    target_size=image_size,
    batch_size=32,
    class_mode='raw'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=image_dir,
    x_col='frame',
    y_col=['x1', 'y1', 'x2', 'y2'],
    target_size=image_size,
    batch_size=32,
    class_mode='raw'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=image_dir,
    x_col='frame',
    y_col=['x1', 'y1', 'x2', 'y2'],
    target_size=image_size,
    batch_size=1,
    class_mode='raw'
)

def create_openpose_model(input_shape=(256, 256, 3)):
    # Define the OpenPose model architecture
    model = Sequential([
        # Stage 1
        Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        MaxPooling2D((2,2), strides=(2,2)),
        # Stage 2
        Conv2D(128, (3,3), padding='same', activation='relu'),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        MaxPooling2D((2,2), strides=(2,2)),
        # Stage 3
        Conv2D(256, (3,3), padding='same', activation='relu'),
        Conv2D(256, (3,3), padding='same', activation='relu'),
        Conv2D(256, (3,3), padding='same', activation='relu'),
        MaxPooling2D((2,2), strides=(2,2)),
        # Stage 4
        Conv2D(512, (3,3), padding='same', activation='relu'),
        Conv2D(512, (3,3), padding='same', activation='relu'),
        Conv2D(512, (3,3), padding='same', activation='relu'),
        MaxPooling2D((2,2), strides=(2,2)),
        # Stage 5
        Conv2D(512, (3,3), padding='same', activation='relu'),
        Conv2D(512, (3,3), padding='same', activation='relu'),
        Conv2D(512, (3,3), padding='same', activation='relu'),
        MaxPooling2D((2,2), strides=(2,2)),
        # Output
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='linear')
    ])
    return model


# Build the model
model = create_openpose_model((256, 256, 3))


# Define the loss function and optimizer
def openpose_loss(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)

model.compile(optimizer='adam', loss=openpose_loss)

# Train the model
batch_size = 32
epochs = 25

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

history = model.fit(train_images, train_bbox, 
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    callbacks=[early_stopping, model_checkpoint])

# Evaluate the model on the test set
model.evaluate(test_images, test_bbox)


from google.colab.patches import cv2_imshow

# Make predictions on the test set and visualize the results
predictions = model.predict(test_images)
for i in range(len(predictions)):
    image = test_images[i]
    bbox_true = tuple((test_bbox[i] * image_size[0]).astype(int))
    bbox_pred = tuple((predictions[i] * image_size[0]).astype(int))
    image_with_bbox = cv2.rectangle(image, bbox_true[:2], bbox_true[2:], (0, 255, 0), 2)
    image_with_bbox = cv2.rectangle(image_with_bbox, bbox_pred[:2], bbox_pred[2:], (255, 0, 0), 2)
    cv2_imshow(image_with_bbox)
    cv2.waitKey(0)
cv2.destroyAllWindows()


# Train the model
model.fit(train_generator, epochs=10, validation_data=val_generator)


# Evaluate the model on the test set
# test_generator = DataGenerator(test_df, batch_size=1, shuffle=False)
results = model.evaluate(test_generator)


results


# Print the test set loss and accuracy
# print(f'Test loss: {results[0]}, Test accuracy: {results[1]}')
print(f'Test loss:', {results})


# Save the model
model.save('/content/gdrive/MyDrive/awone/openpose_model.h5')


"""TableNet Extractor
   This module is built from this ICDAR 2019 paper:
   
   TableNet: Deep Learning model for end-to-end Table detection and Tabular 
   data extraction from Scanned Document Images

   This module should be loadable and able to accept images for extraction 
   from any source as long as they are JPEG
"""
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, Concatenate, UpSampling2D

class prepMarmot:

    def __init__(self,marmot_dir, col_mask_out, tbl_mask_out):
        self.marmot_dir = marmot_dir
        self.col_mask_out = col_mask_out
        self.tbl_mask_out = tbl_mask_out

    @staticmethod
    def sameTable(ymin_1, ymin_2, ymax_1, ymax_2):
        # Returns if columns belong to same table or not
        min_diff = abs(ymin_1 - ymin_2)
        max_diff = abs(ymax_1 - ymax_2)

        if min_diff <= 5 and max_diff <=5:
            return True
        elif min_diff <= 4 and max_diff <=7:
            return True
        elif min_diff <= 7 and max_diff <=4:
            return True
        return False
    
    def prepData(self):
        for file in tqdm(os.listdir(self.marmot_dir)):
            filename_full = os.fsdecode(file)
            if filename_full.endswith(".xml"):
                filename = filename_full[:-4]

                # read in the xml file
                tree = ET.parse(self.marmot_dir + filename_full)
                root = tree.getroot()

                # Get size
                size    = root.find('size')
                width   = int(size.find('width').text)
                height  = int(size.find('height').text)

                # Create grayscale array
                col_mask = np.zeros((height, width), dtype = np.int32)
                tbl_mask = np.zeros((height, width), dtype = np.int32)

                got_first_column = False
                i=0
                tbl_xmin = 10000
                tbl_xmax = 0

                tbl_ymin = 10000
                tbl_ymax = 0
                for column in root.findall('object'):
                    bndbox = column.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)

                    col_mask[ymin:ymax, xmin:xmax] = 255

                    if got_first_column:
                        if self.sameTable(prev_ymin, ymin, prev_ymax, ymax) == False:
                            i+=1
                            got_first_column = False
                            tbl_mask[tbl_ymin:tbl_ymax, tbl_xmin:tbl_xmax] = 255
                            
                            tbl_xmin = 10000
                            tbl_xmax = 0

                            tbl_ymin = 10000
                            tbl_ymax = 0

                    if got_first_column == False:
                        got_first_column = True
                        first_xmin = xmin
                        
                    prev_ymin = ymin
                    prev_ymax = ymax
                    
                    tbl_xmin = min(xmin, tbl_xmin)
                    tbl_xmax = max(xmax, tbl_xmax)
                    
                    tbl_ymin = min(ymin, tbl_ymin)
                    tbl_ymax = max(ymax, tbl_ymax)

                tbl_mask[tbl_ymin:tbl_ymax, tbl_xmin:tbl_xmax] = 255
                im = Image.fromarray(col_mask.astype(np.uint8),'L')
                im.save(self.col_mask_out + filename + ".jpeg")

                im = Image.fromarray(tbl_mask.astype(np.uint8),'L')
                im.save(self.tbl_mask_out + filename + ".jpeg")

class tblNet:
    img_height, img_width = 768, 768

    def __init__(self,marmot_dir=None,col_mask_dir=None,tbl_mask_dir=None):
        self.marmot_dir     = marmot_dir
        self.col_mask_dir   = col_mask_dir
        self.tbl_mask_dir   = tbl_mask_dir

    @staticmethod
    def normalize(input_image):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        return input_image

    def decode_img(self,img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img,channels=3)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = tf.cast(img,tf.uint8)
        # resize the image to the desired size
        return img

    def decode_mask_img(self,img):
        # convert the compressed string to a 2D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=1)
        # resize the image to the desired size
        return tf.image.resize(img, [self.img_height, self.img_width])

    def process_path(self,file_path):
        file_path = tf.strings.regex_replace(file_path, '.xml', '.jpg')

        mask_file_path = tf.strings.regex_replace(file_path, '.jpg', '.jpeg')

        table_mask_file_path = tf.strings.regex_replace(mask_file_path, self.marmot_dir, self.tbl_mask_dir)
        column_mask_file_path = tf.strings.regex_replace(mask_file_path, self.marmot_dir, self.col_mask_dir)

        img = self.decode_img(tf.io.read_file(file_path))
        tbl_mask = self.decode_mask_img(tf.io.read_file(table_mask_file_path))
        col_mask = self.decode_mask_img(tf.io.read_file(column_mask_file_path))
        img = self.normalize(img)
        tbl_mask = self.normalize(tbl_mask)
        col_mask = self.normalize(col_mask)

        return img, {"table_output" : tbl_mask, "column_output" : col_mask }

    def prep_image_from_path(self,path):
        """prep_image_from_path
           Take an image path and prep it for classification
        """
        img = tf.io.read_file(path)
        img = self.decode_img(img)
        img = self.normalize(img)
        img = tf.expand_dims(img, axis=0)
        return img

    @staticmethod
    def build_table_decoder(inputs, pool3, pool4):
        x = Conv2D(512, (1, 1), activation = 'relu', name='conv7_table')(inputs)
        x = UpSampling2D(size=(2, 2))(x)

        concatenated = Concatenate()([x, pool4])

        x = UpSampling2D(size=(2,2))(concatenated)
    
        concatenated = Concatenate()([x, pool3])

        x = UpSampling2D(size=(2,2))(concatenated)
        x = UpSampling2D(size=(2,2))(x)

        last = tf.keras.layers.Conv2DTranspose(3, 3, strides=2,padding='same', name='table_output') 
    
        x = last(x)

        return x

    @staticmethod
    def build_column_decoder(inputs, pool3, pool4):
    
        x = Conv2D(512, (1, 1), activation = 'relu', name='block7_conv1_column')(inputs)
        x = Dropout(0.8, name='block7_dropout_column')(x)

        x = Conv2D(512, (1, 1), activation = 'relu', name='block8_conv1_column')(x)
        x = UpSampling2D(size=(2, 2))(x)

        concatenated = Concatenate()([x, pool4])

        x = UpSampling2D(size=(2,2))(concatenated)
        
        concatenated = Concatenate()([x, pool3])

        x = UpSampling2D(size=(2,2))(concatenated)
        x = UpSampling2D(size=(2,2))(x)

        last = tf.keras.layers.Conv2DTranspose(3, 3, strides=2,padding='same', name='column_output') 
        
        x = last(x)

        return x  

    def vgg_base(self,inputs):
        base_model = tf.keras.applications.vgg19.VGG19(input_shape=[self.img_width, self.img_height, 3], include_top=False, weights='imagenet')
        
        layer_names = ['block3_pool', 'block4_pool', 'block5_pool']
        layers = [base_model.get_layer(name).output for name in layer_names]

        pool_layers_model = Model(inputs=base_model.input, outputs=layers, name='VGG-19')
        pool_layers_model.trainable = False

        return pool_layers_model(inputs)
    
    @staticmethod
    def create_mask(pred_mask1, pred_mask2):
        pred_mask1 = tf.argmax(pred_mask1, axis=-1)
        pred_mask1 = pred_mask1[..., tf.newaxis]
        pred_mask2 = tf.argmax(pred_mask2, axis=-1)
        pred_mask2 = pred_mask2[..., tf.newaxis]

        return pred_mask1[0], pred_mask2[0]
    
    def build(self):
        inputShape = (self.img_width, self.img_height, 3)

        inputs = Input(shape=inputShape, name='input')

        pool_layers = self.vgg_base(inputs)

        x = Conv2D(512, (1, 1), activation = 'relu', name='block6_conv1')(pool_layers[2])
        x = Dropout(0.8, name='block6_dropout1')(x)
        x = Conv2D(512, (1, 1), activation = 'relu', name='block6_conv2')(x)
        x = Dropout(0.8, name = 'block6_dropout2')(x)
        
        table_mask = self.build_table_decoder(x, pool_layers[0], pool_layers[1])
        column_mask = self.build_column_decoder(x, pool_layers[0], pool_layers[1])

        self.model = Model(inputs=inputs,outputs=[table_mask, column_mask],name="tablenet")

    def compile(self):
        losses = {
            "table_output": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "column_output": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        }

        lossWeights = {"table_output": 1.0, "column_output": 1.0}

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-08),
            loss=losses,
            metrics=['accuracy'],
            loss_weights=lossWeights
        )

        self.model.summary()

    
    def train(self):
        list_dataset = tf.data.Dataset.list_files(f'{self.marmot_dir}*.xml')

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath="./tblNet_log/mymodel_{epoch}.h5",
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        )
    
        DATASET_SIZE = len(list(list_dataset))
        train_size = int(0.9 * DATASET_SIZE)
        test_size = int(0.1 * DATASET_SIZE)

        EPOCHS = 1000
        VAL_SUBSPLITS = 5
        BATCH_SIZE = 2
        BUFFER_SIZE = 1000
        VALIDATION_STEPS = test_size//BATCH_SIZE//VAL_SUBSPLITS

        train = list_dataset.take(train_size)
        test = list_dataset.skip(train_size)

        TRAIN_LENGTH = len(list(train))
        
        STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

        train = train.shuffle(BUFFER_SIZE)

        train = train.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        test = test.map(self.process_path)

        train_dataset = train.batch(BATCH_SIZE).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        test_dataset = test.batch(BATCH_SIZE)

        model_history = self.model.fit(
            train_dataset, 
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS,
            validation_data=test_dataset,
            callbacks=[model_checkpoint,]
        )

    def classify(self, image_path, get_rows=False):
        """classify
           Method to call on an image file to identify table structures.
        """
        preppedImg = self.prep_image_from_path(image_path)

        # Build the model and then load its trained weights in.
        self.build()
        self.model.summary()
        self.model.load_weights("tblNet.h5")

        tblMask, colMask = self.model.predict(preppedImg)

        # Drop the first column (the batch dimension) out of the array
        colMask = tf.cast(colMask, tf.uint8)
        colMask = np.squeeze(colMask, axis=0)
        colMaskImg = Image.fromarray(colMask, 'RGB')
        colMaskImg.save("col_mask.jpg")
        
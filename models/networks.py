import tensorflow as tf
import tf_keras.layers as layers
import tf_keras.models as models
import tf_keras.initializers as initializers
import tf_keras.regularizers as regularizers


class Pnet(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.conv1 = layers.Conv2D(10, (3, 3), kernel_regularizer=regularizers.l2(0.0005))
        self.prelu1 = layers.PReLU(alpha_initializer=initializers.Constant(0.25), shared_axes=[1, 2])
        self.maxpool = layers.MaxPool2D((2, 2))

        self.conv2 = layers.Conv2D(16, (3, 3), kernel_regularizer=regularizers.l2(0.0005))
        self.prelu2 = layers.PReLU(alpha_initializer=initializers.Constant(0.25), shared_axes=[1, 2])

        self.conv3 = layers.Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.0005))
        self.prelu3 = layers.PReLU(alpha_initializer=initializers.Constant(0.25), shared_axes=[1, 2])

        self.classifier = layers.Conv2D(2, (1, 1), activation="softmax")
        self.regressor = layers.Conv2D(4, (1, 1))     

    def call(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.prelu2(x)

        x = self.conv3(x)
        x = self.prelu3(x)

        face_cls = self.classifier(x)
        box_reg = self.regressor(x)

        return face_cls, box_reg


class Rnet(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = layers.Conv2D(28, (3, 3))
        self.prelu1 = layers.PReLU(alpha_initializer=initializers.Constant(0.25), shared_axes=[1, 2])
        self.maxpool1 = layers.MaxPool2D((3, 3), (2, 2), padding="same")

        self.conv2 = layers.Conv2D(48, (3, 3))
        self.prelu2 = layers.PReLU(alpha_initializer=initializers.Constant(0.25), shared_axes=[1, 2])
        self.maxpool2 = layers.MaxPool2D((3, 3), (2, 2))

        self.conv3 = layers.Conv2D(64, (2, 2))
        self.prelu3 = layers.PReLU(alpha_initializer=initializers.Constant(0.25), shared_axes=[1, 2])

        self.permute = layers.Permute((3, 2, 1))
        self.flatten = layers.Flatten()

        self.linear = layers.Dense(128)
        self.prelu4 = layers.PReLU(alpha_initializer=initializers.Constant(0.25))

        self.classifier = layers.Dense(2, activation="softmax")
        self.regressor = layers.Dense(4)

    def call(self, x):

        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.prelu3(x)

        x = self.permute(x)
        x = self.flatten(x)

        x = self.linear(x)
        x = self.prelu4(x)

        face_cls = self.classifier(x)
        box_reg = self.regressor(x)

        return face_cls, box_reg


class Onet(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = layers.Conv2D(32, (3, 3))
        self.prelu1 = layers.PReLU(alpha_initializer=initializers.Constant(0.25), shared_axes=[1, 2])
        self.maxpool1 = layers.MaxPool2D((3, 3), (2, 2), padding="same")

        self.conv2 = layers.Conv2D(64, (3, 3))
        self.prelu2 = layers.PReLU(alpha_initializer=initializers.Constant(0.25), shared_axes=[1, 2])
        self.maxpool2 = layers.MaxPool2D((3, 3), (2, 2))

        self.conv3 = layers.Conv2D(64, (3, 3))
        self.prelu3 = layers.PReLU(alpha_initializer=initializers.Constant(0.25), shared_axes=[1, 2])
        self.maxpool3 = layers.MaxPool2D((2, 2))

        self.conv4 = layers.Conv2D(128, (2, 2))
        self.prelu4 = layers.PReLU(alpha_initializer=initializers.Constant(0.25), shared_axes=[1, 2])

        self.permute = layers.Permute((3, 2, 1))
        self.flatten = layers.Flatten()

        self.linear = layers.Dense(256)
        self.prelu5 = layers.PReLU(alpha_initializer=initializers.Constant(0.25))
         
        self.classifier = layers.Dense(2, activation="softmax")
        self.box_regressor = layers.Dense(4)
        self.landmarks_regressor = layers.Dense(10)

    def call(self, x):

        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.prelu4(x)

        x = self.permute(x)
        x = self.flatten(x)

        x = self.linear(x)
        x = self.prelu5(x)

        face_cls = self.classifier(x)
        box_reg = self.box_regressor(x)
        landmarks_reg = self.landmarks_regressor(x)

        return face_cls, box_reg, landmarks_reg



if __name__ == "__main__":
    
    pnet = Pnet()
  
    rnet = Rnet()
    
    onet = Onet()
 
    dum1 = tf.random.uniform((2, 12, 12, 3), 0, 1)
    print(pnet(dum1))

    dum2 = tf.random.uniform((2, 24, 24, 3), 0, 1)
    print(rnet(dum2))

    dum3 = tf.random.uniform((2, 48, 48, 3), 0, 1)
    print(onet(dum3))

    pnet.summary()
    rnet.summary()
    onet.summary()

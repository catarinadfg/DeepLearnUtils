# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file

import tensorflow as tf
import tensorflow.nn as nn
import tensorflow.keras as tfk
import numpy as np
import unittest
from trainutils import samePadding, calculateOutShape, createTestImage

def predictSegmentation(logits):
    '''
    Given the logits from a network, computing the segmentation by thresholding all values above 0 if `logits' has one
    channel, or computing the argmax along the channel axis otherwise.
    '''
    # generate prediction outputs, logits has shape BHW[D]C
    if logits.shape[-1] == 1:
        return tf.cast(logits[..., 0] >= 0, tf.int32)  # for binary segmentation threshold on channel 0
    else:
        return tf.argmax(logits, logits.ndim - 1)  # take the index of the max value along dimension 1


class DiceLoss(tfk.losses.Loss):
    smooth = 1e-5

    def call(self, y_true, y_pred):
        """
        Evaluate the dice loss, using the binary formulation if y_pred.shape[-1]==1 and n-class formulation otherwise.
        
        Args:
            y_true: Ground truth, BHW1
            y_pred: Prediction, BHWC
        """
        batchsize = y_true.shape[0]

        if y_pred.shape[-1] == 1:
            psum = tf.sigmoid(y_pred)
            tsum = y_true
        else:
            psum = nn.softmax(y_pred)
            tsum = tf.one_hot(y_true, y_pred.shape[-1])
            tsum = tf.transpose(tsum[..., 0, :], perm=[0, 3, 1, 2])

        tsum = tf.reshape(tsum, (batchsize, -1))
        psum = tf.reshape(psum, (batchsize, -1))

        intersection = psum * tsum
        sums = psum + tsum

        intersection = tf.reduce_sum(intersection, 1) + self.smooth
        sums = tf.reduce_sum(sums, 1) + self.smooth

        score = 2.0 * intersection / sums

        return 1.0 - tf.reduce_mean(score)
    
    
class Identity(tfk.Model):
    def __init__(self, *_, **__):
        super().__init__()

    def call(self, x):
        return x    


class Convolution2D(tfk.Sequential):
    def __init__(self, inChannels, outChannels, strides=1, kernelSize=3, instanceNorm=True,
                 dropout=0, dilation=1, bias=True, convOnly=False, isTransposed=False):
        super().__init__()
        self.inChannels=inChannels
        self.outChannels = outChannels
        self.isTransposed = isTransposed

        # TODO: instance norm should be a choice here
        normalizeFunc = tfk.layers.BatchNormalization

        if isTransposed:
            conv = tfk.layers.Conv2DTranspose(outChannels, kernelSize, strides, 'same',
                                              dilation_rate=dilation, use_bias=bias)
        else:
            conv = tfk.layers.Conv2D(outChannels, kernelSize, strides, 'same',
                                     dilation_rate=dilation, use_bias=bias)

        self.add(conv)

        if not convOnly:
            self.add(normalizeFunc())

            if dropout > 0:
                self.add(tfk.layers.Dropout(dropout))

            self.add(tfk.layers.PReLU(shared_axes=(1,2,3)))


class ResidualUnit2D(tfk.Model):
    def __init__(self, inChannels, outChannels, strides=1, kernelSize=3, subunits=2,
                 instanceNorm=True, dropout=0, dilation=1, bias=True, lastConvOnly=False):
        super().__init__()
        self.outChannels = outChannels
        self.conv = tfk.Sequential()
        self.residual = None

        sstrides = strides
        subunits = max(1, subunits)

        for su in range(subunits):
            convOnly = lastConvOnly and su == (subunits - 1)
            unit = Convolution2D(inChannels, outChannels, sstrides, kernelSize, 
                                 instanceNorm, dropout, dilation, bias, convOnly)
            self.conv.add(unit)
            sstrides = 1

        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides) != 1 or inChannels != outChannels:
            rkernelSize = kernelSize

            if np.prod(strides) == 1:  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernelSize = 1

            self.residual = tfk.layers.Conv2D(outChannels, rkernelSize, strides, 'same', use_bias=bias)

    def call(self, x):
        res = x if self.residual is None else self.residual(x)  # create the additive residual from x
        cx = self.conv(x)  # apply x to sequence of operations
        return cx + res  # add the residual to the output
    
    
class AutoEncoder(tfk.Model):
    def __init__(self, inChannels, outChannels, channels, strides, kernelSize=3, upKernelSize=3,
                 numResUnits=0, interChannels=[], interDilations=[], numInterUnits=2, instanceNorm=True, dropout=0):
        super().__init__()
        assert len(channels) == len(strides)
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernelSize = kernelSize
        self.upKernelSize = upKernelSize
        self.numResUnits = numResUnits
        self.instanceNorm = instanceNorm
        self.dropout = dropout
        self.numInterUnits = numInterUnits
        self.interChannels = list(interChannels)
        self.interDilations = list(interDilations or [1] * len(interChannels))

        self.encodedChannels = inChannels
        decodeChannelList=list(channels[-2::-1]) + [outChannels]
        
        self.encode, self.encodedChannels = self._getEncodeModule(self.encodedChannels, channels, strides)
        self.intermediate, self.encodedChannels = self._getIntermediateModule(self.encodedChannels, numInterUnits)
        self.decode, _ = self._getDecodeModule(self.encodedChannels, decodeChannelList,strides[::-1] or [1])

    def _getEncodeModule(self, inChannels, channels, strides):
        encode = tfk.Sequential()
        layerChannels = inChannels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._getEncodeLayer(layerChannels, c, s, False)
            encode.add(layer)
            layerChannels = c

        return encode, layerChannels

    def _getIntermediateModule(self, inChannels, numInterUnits):
        intermediate = Identity()
        layerChannels = inChannels

        if self.interChannels:
            intermediate = tfk.Sequential()

            for i, (dc, di) in enumerate(zip(self.interChannels, self.interDilations)):

                if self.numInterUnits > 0:
                    unit = ResidualUnit2D(layerChannels, dc, 1, self.kernelSize,
                                          self.numInterUnits, self.instanceNorm, self.dropout, di)
                else:
                    unit = Convolution2D(layerChannels, dc, 1, self.kernelSize, self.instanceNorm, self.dropout, di)

                intermediate.add(unit)
                layerChannels = dc

        return intermediate, layerChannels

    def _getDecodeModule(self, inChannels, channels, strides):
        decode = tfk.Sequential()
        layerChannels = inChannels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._getDecodeLayer(layerChannels, c, s, i == (len(strides) - 1))
            decode.add(layer)
            layerChannels = c

        return decode, layerChannels

    def _getEncodeLayer(self, inChannels, outChannels, strides, isLast):
        if self.numResUnits > 0:
            return ResidualUnit2D(inChannels, outChannels, strides, self.kernelSize,
                                  self.numResUnits, self.instanceNorm, self.dropout, lastConvOnly=isLast)
        else:
            return Convolution2D(inChannels, outChannels, strides, self.kernelSize, self.instanceNorm, self.dropout,
                                 convOnly=isLast)

    def _getDecodeLayer(self, inChannels, outChannels, strides, isLast):
        conv = Convolution2D(inChannels, outChannels, strides, self.upKernelSize,
                             self.instanceNorm, self.dropout, convOnly=isLast and self.numResUnits == 0,
                             isTransposed=True)

        if self.numResUnits > 0:
            return tfk.Sequential([conv,
                                 ResidualUnit2D(outChannels, outChannels, 1, self.kernelSize, 1, self.instanceNorm,
                                                self.dropout, lastConvOnly=isLast)
                                 ])
        else:
            return conv

    def call(self, x):
        x = self.encode(x)
        x = self.intermediate(x)
        x = self.decode(x)
        return (x,)


class VarAutoEncoder(AutoEncoder):
    def __init__(self, inShape, outChannels, latentSize, channels, strides, kernelSize=3, upKernelSize=3,
                 numResUnits=0, interChannels=[], interDilations=[], numInterUnits=2, instanceNorm=True, dropout=0):

        self.inHeight, self.inWidth, inChannels = inShape
        self.latentSize = latentSize
        self.finalSize = np.asarray([self.inHeight, self.inWidth], np.int)

        super().__init__(inChannels, outChannels, channels, strides, kernelSize, upKernelSize, numResUnits,
                         interChannels, interDilations, numInterUnits, instanceNorm, dropout)

        for s in strides:
            self.finalSize = calculateOutShape(self.finalSize, self.kernelSize, s, samePadding(self.kernelSize))

        linearSize = int(np.product(self.finalSize)) * self.encodedChannels
        self.mu = tfk.layers.Dense(self.latentSize)
        self.logvar = tfk.layers.Dense(self.latentSize)
        self.decodeL = tfk.layers.Dense(linearSize)

    def encodeForward(self, x):
        x = self.encode(x)
        if self.intermediate is not None:
            x = self.intermediate(x)
        x = x.view(x.shape[0], -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def decodeForward(self, z):
        x = nn.relu(self.decodeL(z))
        x = x.view(x.shape[0], self.channels[-1], self.finalSize[0], self.finalSize[1])
        x = self.decode(x)
        x = nn.sigmoid(x)
        return x

    def reparameterize(self, mu, logvar):
        std = tf.exp(0.5 * logvar)

        if self.training:  # multiply random noise with std only during training
            std = tf.random.uniform(std.shape).mul(std)

        return std.add_(mu)

    def forward(self, x):
        mu, logvar = self.encodeForward(x)
        z = self.reparameterize(mu, logvar)
        return self.decodeForward(z), mu, logvar, z    


class UnetBlock(tfk.Model):
    def __init__(self, encode, decode, subblock):
        super().__init__()
        self.encode = encode
        self.decode = decode
        self.subblock = subblock

    def call(self, x):
        enc = self.encode(x)
        sub = self.subblock(enc)
        dec = tf.concat([enc, sub], len(enc.shape) - 1)
        return self.decode(dec)


class Unet(tfk.Model):
    def __init__(self, inChannels, numClasses, channels, strides, kernelSize=3,
                 upKernelSize=3, numResUnits=0, instanceNorm=True, dropout=0):
        super().__init__()
        assert len(channels) == (len(strides) + 1)
        self.inChannels = inChannels
        self.numClasses = numClasses
        self.channels = channels
        self.strides = strides
        self.kernelSize = kernelSize
        self.upKernelSize = upKernelSize
        self.numResUnits = numResUnits
        self.instanceNorm = instanceNorm
        self.dropout = dropout

        def _createBlock(inc, outc, channels, strides, isTop):
            c = channels[0]
            s = strides[0]

            if len(channels) > 2:
                subblock = _createBlock(c, c, channels[1:], strides[1:], False)
                upc = c * 2
            else:
                subblock = self._getBottomLayer(c, channels[1])
                upc = c + channels[1]

            down = self._getDownLayer(inc, c, s, isTop)
            up = self._getUpLayer(upc, outc, s, isTop)

            return UnetBlock(down, up, subblock)

        self.model = _createBlock(inChannels, numClasses, self.channels, self.strides, True)

    def _getDownLayer(self, inChannels, outChannels, strides, isTop):
        if self.numResUnits > 0:
            return ResidualUnit2D(inChannels, outChannels, strides, self.kernelSize,
                                  self.numResUnits, self.instanceNorm, self.dropout)
        else:
            return Convolution2D(inChannels, outChannels, strides, self.kernelSize, self.instanceNorm, self.dropout)

    def _getBottomLayer(self, inChannels, outChannels):
        return self._getDownLayer(inChannels, outChannels, 1, False)

    def _getUpLayer(self, inChannels, outChannels, strides, isTop):
        conv = Convolution2D(inChannels, outChannels, strides, self.upKernelSize, self.instanceNorm,
                             self.dropout, convOnly=isTop and self.numResUnits == 0, isTransposed=True)

        if self.numResUnits > 0:
            return tfk.Sequential([conv, ResidualUnit2D(outChannels, outChannels, 1, self.kernelSize, 1,
                                                       self.instanceNorm, self.dropout, lastConvOnly=isTop)])
        else:
            return conv

    def call(self, x):
        x = self.model(x)
        return x, predictSegmentation(x)


########################################################################################################################
### Tests
########################################################################################################################


class ImageTestCase(unittest.TestCase):
    def setUp(self):
        self.inShape = (128, 128)
        self.inputChannels = 1
        self.outputChannels = 4
        self.numClasses = 3

        im, msk = createTestImage(self.inShape[0], self.inShape[1], 4, 20, 0, self.numClasses)

        self.imT = tf.convert_to_tensor(im[None, ..., None])

        self.seg1 = tf.convert_to_tensor((msk[None, ..., None] > 0).astype(np.float32))
        self.segn = tf.convert_to_tensor(msk[None, ..., None])
        self.seg1hot = tf.cast(tf.one_hot(tf.convert_to_tensor(msk[None]), self.numClasses + 1), tf.float32)


class TestDiceLoss(ImageTestCase):
    def setUp(self):
        super().setUp()
        self.loss = DiceLoss()

    def test_shapes(self):
        self.assertEqual(self.seg1.shape, (1, self.inShape[0], self.inShape[1], 1))
        self.assertEqual(self.segn.shape, (1, self.inShape[0], self.inShape[1], 1))
        self.assertEqual(self.seg1hot.shape, (1, self.inShape[0], self.inShape[1], self.numClasses + 1))

    def test_binary1(self):
        l = self.loss(
            y_true=self.seg1,
            y_pred=self.seg1
        )
        self.assertTrue(l.numpy() > 0)

    def test_nclass1(self):
        l = self.loss(
            y_true=self.segn,
            y_pred=self.seg1hot
        )
        self.assertTrue(l.numpy() > 0)


class TestConvolution2D(ImageTestCase):
    def test_conv1(self):
        conv = Convolution2D(self.inputChannels,self.outputChannels)
        out = conv(self.imT)
        expectedShape = (1, self.inShape[0], self.inShape[1], self.outputChannels)
        self.assertEqual(out.shape, expectedShape)

    def test_convOnly1(self):
        conv = Convolution2D(self.inputChannels,self.outputChannels, convOnly=True)
        out = conv(self.imT)
        expectedShape = (1, self.inShape[0], self.inShape[1], self.outputChannels)
        self.assertEqual(out.shape, expectedShape)

    def test_stride1(self):
        conv = Convolution2D(self.inputChannels,self.outputChannels, strides=2)
        out = conv(self.imT)
        expectedShape = (1, self.inShape[0] // 2, self.inShape[1] // 2, self.outputChannels)
        self.assertEqual(out.shape, expectedShape)

    def test_dilation1(self):
        conv = Convolution2D(self.inputChannels,self.outputChannels, dilation=3)
        out = conv(self.imT)
        expectedShape = (1, self.inShape[0], self.inShape[1], self.outputChannels)
        self.assertEqual(out.shape, expectedShape)

    def test_dropout1(self):
        conv = Convolution2D(self.inputChannels,self.outputChannels, dropout=0.15)
        out = conv(self.imT)
        expectedShape = (1, self.inShape[0], self.inShape[1], self.outputChannels)
        self.assertEqual(out.shape, expectedShape)

    def test_transpose1(self):
        conv = Convolution2D(self.inputChannels,self.outputChannels, isTransposed=True)
        out = conv(self.imT)
        expectedShape = (1, self.inShape[0], self.inShape[1], self.outputChannels)
        self.assertEqual(out.shape, expectedShape)

    def test_transpose2(self):
        conv = Convolution2D(self.inputChannels,self.outputChannels, strides=2, isTransposed=True)
        out = conv(self.imT)
        expectedShape = (1, self.inShape[0] * 2, self.inShape[1] * 2, self.outputChannels)
        self.assertEqual(out.shape, expectedShape)


class TestResidualUnit2D(ImageTestCase):
    def test_convOnly1(self):
        conv = ResidualUnit2D(1, self.outputChannels)
        out = conv(self.imT)
        expectedShape = (1, self.inShape[0], self.inShape[1], self.outputChannels)
        self.assertEqual(out.shape, expectedShape)

    def test_stride1(self):
        conv = ResidualUnit2D(1, self.outputChannels, strides=2)
        out = conv(self.imT)
        expectedShape = (1, self.inShape[0] // 2, self.inShape[1] // 2, self.outputChannels)
        self.assertEqual(out.shape, expectedShape)

    def test_dilation1(self):
        conv = ResidualUnit2D(1, self.outputChannels, dilation=3)
        out = conv(self.imT)
        expectedShape = (1, self.inShape[0], self.inShape[1], self.outputChannels)
        self.assertEqual(out.shape, expectedShape)

    def test_dropout1(self):
        conv = ResidualUnit2D(1, self.outputChannels, dropout=0.15)
        out = conv(self.imT)
        expectedShape = (1, self.inShape[0], self.inShape[1], self.outputChannels)
        self.assertEqual(out.shape, expectedShape)


class TestAutoEncoder(ImageTestCase):
    def test_1channel1(self):
        net = AutoEncoder(1, 1, [4, 8, 16], [2, 2, 2])
        out = net(self.imT)
        self.assertEqual(out[0].shape, self.imT.shape)

    def test_nchannel1(self):
        net = AutoEncoder(1, self.numClasses + 1, [4, 8, 16], [2, 2, 2])
        out = net(self.imT)
        self.assertEqual(out[0].shape, self.seg1hot.shape)
    
    
class TestVarAutoEncoder(ImageTestCase):
    def test_1channel1(self):
        net = VarAutoEncoder(self.imT.shape[1:], 1, 64, [4, 8, 16], [2, 2, 2])
        out = net(self.imT)
        self.assertEqual(out[0].shape, self.imT.shape)

    def test_nchannel1(self):
        unet = VarAutoEncoder(self.imT.shape[1:], self.numClasses + 1, 64, [4, 8, 16], [2, 2, 2])
        out = unet(self.imT)
        self.assertEqual(out[0].shape, self.seg1hot.shape)
        

class TestUnet(ImageTestCase):
    def test_1class1(self):
        net = Unet(1, 1, [4, 8, 16], [2, 2])
        out = net(self.imT)
        self.assertEqual(out[0].shape, self.imT.shape)
        self.assertEqual(out[1].shape, self.imT.shape[:-1])

    def test_nclass1(self):
        net = Unet(1, self.numClasses + 1, [4, 8, 16], [2, 2])
        out = net(self.imT)
        self.assertEqual(out[0].shape, self.seg1hot.shape)
        self.assertEqual(out[1].shape, self.imT.shape[:-1])
        
    def test_residual1(self):
        net = Unet(1, self.numClasses + 1, [4, 8, 16], [2, 2],numResUnits=2)
        out = net(self.imT)
        self.assertEqual(out[0].shape, self.seg1hot.shape)
        self.assertEqual(out[1].shape, self.imT.shape[:-1])


if __name__ == '__main__':
    unittest.main()

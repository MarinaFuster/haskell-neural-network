module Optimization (
    Optimizer (..),
    AParameters (..),
    train
) where

import Numeric.LinearAlgebra
import NeuralNetwork ( NeuralNetwork(..), Layer(..), Gradients(..), backprop )

type LearningRate = Double
type Beta = Double
type Epsilon = Double

data AParameters = AParameters Beta Beta Epsilon Double
data Optimizer = GradientDescent LearningRate | Adam AParameters

train :: NeuralNetwork               -- network to train
    -> Optimizer                        -- optimizer to use
    -> (Matrix Double, Matrix Double)   -- (samples, targets)
    -> Int                              -- epochs
    -> NeuralNetwork                    -- resulting neural network

-- | Optimizers' train definitions

train net0 (GradientDescent lr) dataset epochs = last $ take epochs (iterate step net0)
    where
        step net = zipWith (update lr) net gradients
          where
            (_, gradients) = backprop net dataset

train net0 (Adam params) dataset epochs = net
  where
    s0 = initializeAtZero net0
    v0 = initializeAtZero net0
    (net, _, _) = _adam params epochs (net0, s0, v0) dataset


-- | Auxiliary functions for optimizers

update :: Double -> Layer -> Gradients -> Layer
update lr = \(Layer w b a) (Gradients dW dB) -> 
    Layer (w - lr `scale` dW) (b - lr `scale` dB) a

initializeAtZero :: NeuralNetwork -> [(Matrix Double, Matrix Double)]
initializeAtZero net = 
    let zerosLike m = matrix (cols m) (replicate ((rows m)*(cols m)) 0.0) 
        in let zf (Layer a b _) = (zerosLike a, zerosLike b)
            in map zf net

_adam :: AParameters
    -> Int
    -> ([Layer], [(Matrix Double, Matrix Double)], [(Matrix Double, Matrix Double)])
    -> (Matrix Double, Matrix Double)
    -> ([Layer], [(Matrix Double, Matrix Double)], [(Matrix Double, Matrix Double)])
_adam (AParameters beta1 beta2 epsilon lr) iterN (w0, s0, v0) dataSet = last $ take iterN (iterate step (w0, s0, v0))
  where
    step (w, s, v) = (wN, sN, vN)
      where
        (_, dW) = backprop w dataSet -- we compute gradients at each iteration with mini batch

        sN = zipWith f2 s dW
        vN = zipWith f3 v dW
        wN = zipWith3 f w vN sN

        f :: Layer
          -> (Matrix Double, Matrix Double)
          -> (Matrix Double, Matrix Double)
          -> Layer
        f (Layer w_ b_ sf) (vW, vB) (sW, sB) =
           Layer (w_ - lr `scale` vW / ((sqrt sW) `addC` epsilon))
                 (b_ - lr `scale` vB / ((sqrt sB) `addC` epsilon))
                 sf

        addC m c = cmap (+ c) m

        f2 :: (Matrix Double, Matrix Double)
           -> Gradients
           -> (Matrix Double, Matrix Double)
        f2 (sW, sB) (Gradients dW dB) =
          ( beta2 `scale` sW + (1 - beta2) `scale` (dW^2)
          , beta2 `scale` sB + (1 - beta2) `scale` (dB^2))

        f3 :: (Matrix Double, Matrix Double)
           -> Gradients
           -> (Matrix Double, Matrix Double)
        f3 (vW, vB) (Gradients dW dB) =
          ( beta1 `scale` vW + (1 - beta1) `scale` dW
          , beta1 `scale` vB + (1 - beta1) `scale` dB)
module Optimization (
    Optimizer (..),
    train
) where

import Numeric.LinearAlgebra
import NeuralNetwork ( NeuralNetwork(..), Layer(..), Gradients(..), backprop )

type LearningRate = Double
data Optimizer = GradientDescent LearningRate

train :: NeuralNetwork               -- network to train
    -> Optimizer                        -- optimizer to use
    -> (Matrix Double, Matrix Double)   -- (samples, targets)
    -> Int                              -- epochs
    -> NeuralNetwork                    -- resulting neural network
update :: Optimizer -> Layer -> Gradients -> Layer

-- | Gradient descent definitions

train net0 (GradientDescent lr) dataset epochs = last $ take epochs (iterate step net0)
    where
        step net = zipWith (update (GradientDescent lr)) net gradients
          where
            (_, gradients) = backprop net dataset
update (GradientDescent lr) = \(Layer w b a) (Gradients dW dB) -> 
    Layer (w - lr `scale` dW) (b - lr `scale` dB) a

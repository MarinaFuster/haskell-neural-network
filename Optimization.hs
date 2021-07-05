module Optimization (
    Optimizer (..),
    train
) where

import Numeric.LinearAlgebra
import NeuralNetwork ( NeuralNetwork(..), Layer(..), Gradients(..), backprop )

type LearningRate = Double
-- data GDParameters = GDParameters LearningRate
data Optimizer = GradientDescent LearningRate

train :: NeuralNetwork               -- network to train
    -> Optimizer                        -- optimizer to use
    -> (Matrix Double, Matrix Double)   -- (samples, targets)
    -> Int                              -- epochs
    -> NeuralNetwork                    -- resulting neural network

train net0 optimizer dataset epochs = last $ take epochs (iterate step net0)
    where
        step net = zipWith (update optimizer) net gradients
          where
            (_, gradients) = backprop net dataset

update :: Optimizer -> Layer -> Gradients -> Layer
update (GradientDescent lr) = \(Layer w b a) (Gradients dW dB) -> 
    Layer (w - lr `scale` dW) (b - lr `scale` dB) a
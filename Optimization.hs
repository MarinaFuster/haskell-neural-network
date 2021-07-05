module Optimization (
    GDParameters (..),
    Optimizer (..),
    optimize
) where

import Numeric.LinearAlgebra
import NeuralNetwork ( NeuralNetwork(..) )

type LearningRate = Double
data GDParameters = GDParameters LearningRate
data Optimizer = GradientDescent GDParameters

optimize :: NeuralNetwork               -- network to train
    -> Optimizer                        -- optimizer to use
    -> (Matrix Double, Matrix Double)   -- (samples, targets)
    -> Int                              -- epochs
    -> NeuralNetwork                    -- resulting neural network

optimize = undefined
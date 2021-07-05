module Evaluation ( hits ) where

import Numeric.LinearAlgebra ( Matrix (..), sumElements, rows )
import NeuralNetwork ( NeuralNetwork (..), forward )

hits :: NeuralNetwork -> (Matrix Double, Matrix Double) -> Double
hits net (samples, targets) = 100 * (1 - errors / (fromIntegral $ rows targets))
    where
        errors = sumElements $ abs (targets - (forward net samples))

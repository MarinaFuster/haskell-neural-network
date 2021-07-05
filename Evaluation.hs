module Evaluation ( hits, classify ) where

import Numeric.LinearAlgebra ( Matrix (..), sumElements, rows, cmap )
import NeuralNetwork ( NeuralNetwork (..), forward )

classify :: NeuralNetwork -> Matrix Double -> Matrix Double
classify net samples = cmap (\a -> if a < 0.5 then 0 else 1) (forward net samples)

-- | TODO: this hits should be HOW MANY ARE OK
hits :: NeuralNetwork -> (Matrix Double, Matrix Double) -> Double
hits = undefined
--hits net (samples, targets) = 100 * (1 - errors / (fromIntegral $ rows targets))
--    where
--        errors = sumElements $ abs (targets - (classify net samples))

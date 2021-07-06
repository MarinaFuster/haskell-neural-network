module Evaluation ( binaryClassify, binaryHits, binaryAccuracy ) where

import Numeric.LinearAlgebra ( Matrix (..), sumElements, rows, cmap )
import NeuralNetwork ( NeuralNetwork (..), feedforward )

-- | Transforms output layer with 1 neuron into binary classification

binaryClassify :: NeuralNetwork -> Matrix Double -> Matrix Double
binaryClassify net samples = cmap (\a -> if a < 0.5 then 0 else 1) (feedforward net samples)

-- | Returns how many were hits and how many were errors

binaryHits :: NeuralNetwork -> (Matrix Double, Matrix Double) -> (Double, Double)
binaryHits net (samples, targets) =
   let n = fromIntegral $ rows targets
    in let errors = sumElements $ abs (targets - (binaryClassify net samples))
     in (n - errors, errors)

-- | Returns accuracy for binary classification problems

binaryAccuracy :: NeuralNetwork -> (Matrix Double, Matrix Double) -> Double
binaryAccuracy net (samples, targets) = 100 * (1 - e / m)
  where
    predictions = net `binaryClassify` samples
    e = sumElements $ abs (targets - predictions)
    m = fromIntegral $ rows targets

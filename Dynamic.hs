module Dynamic (
    buildNetwork
) where

import Numeric.LinearAlgebra
import NeuralNetwork

-- | Neural Network dynamic building functions

_genValues :: (Int, Int) -> IO (Matrix Double)
_genValues (dimensionIn, dimensionOut) = do            
    let k = sqrt (1.0 / fromIntegral dimensionIn)
    w <- randn dimensionIn dimensionOut
    return (k `scale` w)

_generateLayerConnections :: (Int, Int) -> IO (Weights, Biases)
_generateLayerConnections (inconnections, neurons) = do
    weights <- _genValues (inconnections, neurons)
    biases <- _genValues (1, neurons)
    return (weights, biases)

buildNetwork :: Int -> [Int] -> [Activation] -> IO NeuralNetwork
buildNetwork features neurons activations = do
    connections <- mapM _generateLayerConnections dimforlayers
    return (zipWith (\(w, b) a -> Layer w b a) connections activations)
    where
        dimforlayers = zip (features:neurons) neurons

module NeuralNetwork (
    Activation (..),
    buildNetwork
) where      -- here everything is exported

import Numeric.LinearAlgebra as LA
import Numeric.Morpheus.Activation ( 
    relu,
    reluGradient,
    sigmoid,
    sigmoidGradient,
    tanh_,
    tanhGradient
    )

-- | Neural Network data types

data Activation = Relu | Tanh | Sigmoid     -- Activation :: matrix double -> matrix double
data Error = MSE                            -- Error :: samples -> targets -> matrix double

type Weights = Matrix Double    -- layer weights
type Biases = Matrix Double     -- layer biases
type DW = Matrix Double         -- delta for weights
type DB = Matrix Double         -- delta for bias

data Layer = Layer Weights Biases Activation
data Gradients = Gradients DW DB

type NeuralNetowrk = [Layer]

-- | Neural Network building functions

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

buildNetwork :: Int -> [Int] -> [Activation] -> IO NeuralNetowrk
buildNetwork features neurons activations = do
    connections <- mapM _generateLayerConnections dimforlayers
    return (zipWith (\(w, b) a -> Layer w b a) connections activations)
    where
        dimforlayers = zip (features:neurons) neurons

-- | Functions and derivatives

apply :: Activation -> Matrix Double -> Matrix Double
apply Relu = relu
apply Tanh = tanh_
apply Sigmoid = sigmoid

applyDerivative :: Activation -> Matrix Double -> Matrix Double -> Matrix Double
applyDerivative Sigmoid = \x dY -> sigmoidGradient x * dY
applyDerivative Relu = \x dY -> reluGradient x * dY
applyDerivative Tanh = \x dY -> tanhGradient x * dY

compute :: Error -> Matrix Double -> Matrix Double -> Matrix Double
compute MSE = \x y -> 0.5 `scale` (cmap (^ 2) (x - y))  

computeDerivative :: Error -> Matrix Double -> Matrix Double -> Matrix Double
computeDerivative MSE = \x y -> x - y

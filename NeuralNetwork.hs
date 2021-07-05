module NeuralNetwork (
    Activation (..),
    Error (..),
    Weights (..),
    Biases (..),
    DW (..),
    DB (..),
    Layer (..),
    Gradients (..),
    NeuralNetwork (..)
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

type NeuralNetwork = [Layer]

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
-- | TODO: test
-- sumElements will return a single value instead of MSE for each sample
compute MSE = \x y -> sumElements $ 0.5 `scale` (cmap (^ 2) (x - y))  

computeDerivative :: Error -> Matrix Double -> Matrix Double -> Matrix Double
-- | TODO: test
computeDerivative MSE = \x y -> x - y

-- | TODO: recursive function for backprop and feedforward
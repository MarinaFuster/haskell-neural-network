module NeuralNetwork (
    Activation (..),
    Error (..),
    Weights (..),
    Biases (..),
    DW (..),
    DB (..),
    Layer (..),
    Gradients (..),
    NeuralNetwork (..),
    backprop,
    feedforward
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

data Activation = Relu | Tanh | Sigmoid
data Error = MSE | CrossEntropy

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

compute :: Error -> Matrix Double -> Matrix Double -> Double
compute MSE = \x y -> sumElements $ 0.5 `scale` (cmap (^ 2) (x - y))  

computeDerivative :: Error -> Matrix Double -> Matrix Double -> Matrix Double
computeDerivative MSE = \x y -> x - y

-- | Backpropagation and Feed Forward algorithms

propagate ::
  Error ->                                    -- error/cost function
  Matrix Double ->                            -- targets
  Matrix Double ->                            -- inputs
  NeuralNetwork ->                            -- layers
  (Matrix Double, Matrix Double, [Gradients]) -- (dX, predictions, list of gradients)

propagate err t inp [] = computeLoss err t inp
propagate err t inp (Layer w b a:xs) = (dX, preds, gradient:gradients)
  where
    h = getH inp w b
    (dZ, preds, gradients) = propagate err t (apply a h) xs
    (dX, gradient) = computeGradients inp h dZ (Layer w b a)

backprop ::
  Error ->                            -- error function
  NeuralNetwork ->                    -- nn
  (Matrix Double, Matrix Double) ->   -- (samples, targets)
  (Matrix Double, [Gradients])        -- (predictions, list of gradients)

backprop err net (samples, targets) = dropFirst $ propagate err targets samples net 

feedforward ::
  NeuralNetwork ->      -- nn
  Matrix Double ->      -- samples
  Matrix Double         -- predictions

-- we use undefined for both targets and error function (we won't need them)
feedforward net samples = fst $ backprop undefined net (samples, undefined) 

-- | Auxiliary functions

computeLoss :: Error -> Matrix Double -> Matrix Double -> (Matrix Double, Matrix Double, [Gradients])
computeLoss err t inp = (computeDerivative err inp t, inp, [])

getH :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
getH inp w b = (inp LA.<> w) + b

computeGradients :: 
  Matrix Double ->                  -- inputs 
  Matrix Double ->                  -- excitation of layer
  Matrix Double ->                  -- dZ from next layer
  Layer ->                          -- current layer
  (Matrix Double, Gradients)
computeGradients inp h dZ (Layer w b a) = (linearX' w dY, Gradients dW dB)
  where
    dY = applyDerivative a h dZ
    (dW, dB) = (linearW' inp dY, bias' dY)

bias' :: Matrix Double -> Matrix Double
bias' dY = cmap (/ m) r
  where
    r = matrix (cols dY) $ map sumElements (toColumns dY)
    m = fromIntegral $ rows dY

linearW' :: Matrix Double -> Matrix Double -> Matrix Double
linearW' x dY = cmap (/ m) (tr' x LA.<> dY)
  where
    m = fromIntegral $ rows x

linearX' :: Matrix Double -> Matrix Double -> Matrix Double
linearX' w dY = dY LA.<> tr' w

dropFirst :: (a, b, c) -> (b, c)
dropFirst (x, y, z) = (y, z)

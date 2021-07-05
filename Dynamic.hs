module Dynamic (
    buildNetwork,
    makeCircles,
    makeSpirals
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

-- | Circles dataset
makeCircles :: Int -> Double -> Double -> IO (Matrix Double, Matrix Double)
makeCircles m factor noise = do
  let rand' n = (scale (2 * pi)) <$> rand n 1
      m1 = m `div` 2
      m2 = m - (m `div` 2)

  r1 <- rand' m1
  r2 <- rand' m2
  ns <- scale noise <$> randn m 2

  let outerX = cos r1
      outerY = sin r1
      innerX = scale factor $ cos r2
      innerY = scale factor $ sin r2
      -- Merge them all
      x = fromBlocks [[outerX, outerY], [innerX, innerY]]

      -- Labels
      y1 = m1 >< 1 $ repeat 0
      y2 = m2 >< 1 $ repeat 1
      y = y1 === y2

  return (x + ns, y)

-- | Spirals dataset.
-- Note, produces twice more points than m.
makeSpirals :: Int -> Double -> IO (Matrix Double, Matrix Double)
makeSpirals m noise = do
  r0 <- (scale (780 * 2*pi / 360). sqrt) <$> rand m 1
  d1x0 <- scale noise <$> rand m 1
  d1y0 <- scale noise <$> rand m 1

  let d1x = d1x0 - cos(r0) * r0
  let d1y = d1y0 + sin(r0) * r0

  let x = (fromBlocks [[d1x, d1y], [-d1x, -d1y]]) / 10.0
  let y1 = m >< 1 $ repeat 0
  let y2 = m >< 1 $ repeat 1
  let y = y1 === y2
  return (x, y)
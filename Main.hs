import Numeric.LinearAlgebra ( loadMatrix )
import Text.Printf ( printf )
import NeuralNetwork ( Activation(..) )
import Optimization ( Optimizer(..), train )
import Dynamic ( buildNetwork )

-- | Experiment example
experiment :: IO ()
experiment = do

    -- provision of dataset
    samples <- loadMatrix "datasets/iris_x.dat"
    targets <- loadMatrix "datasets/iris_y.dat"

    -- provision of initial net
    let inputFeatures = 4
    let layers = [128, 3]
    let activations = [Relu, Sigmoid]
    net <- buildNetwork inputFeatures layers activations

    -- provision of optmizer
    let learningRate = 0.001
    let optimizer = GradientDescent learningRate

    -- provision of training parameters
    let epochs = 7000

    putStrLn "Finished experiment"


main = experiment

-- | TODO: figure out a way to compute time properly and compare it
-- | to python way of doing it ??
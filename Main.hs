import Numeric.LinearAlgebra ( loadMatrix )
import Text.Printf ( printf )
import NeuralNetwork ( Activation(..) )
import Optimization ( Optimizer(..), train )
import Dynamic ( buildNetwork )
import Evaluation ( hits )

-- | Experiment example

experiment :: IO ()
experiment = do

    -- provision of dataset
    samples <- loadMatrix "datasets/iris_x.dat"
    targets <- loadMatrix "datasets/iris_y.dat"

    -- provision of initial net
    let inputFeatures = 4
        layers = [128, 3]
        activations = [Relu, Sigmoid]
    
    net <- buildNetwork inputFeatures layers activations

    -- provision of train parameters
    let learningRate = 0.001
        optimizer = GradientDescent learningRate
        epochs = 7000
        
        trainedNet = train net optimizer (samples, targets) epochs

    let correct = hits trainedNet (samples, targets)

    putStrLn "Finished experiment"
    putStrLn $ printf "Hits for net (gradient descent) %.2f" (correct)

main = experiment

-- | TODO: figure out a way to compute time properly and compare it
-- | to python way of doing it ??
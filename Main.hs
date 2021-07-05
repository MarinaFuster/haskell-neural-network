import Text.Printf ( printf )
import NeuralNetwork ( Activation(..) )
import Optimization ( GDParameters(..), Optimizer(..) )
import Dynamic ( makeCircles, buildNetwork )

-- | Experiment example
experiment :: IO ()
experiment = do

    -- provision of dataset
    -- | TODO: this should be dataset for reproducible results
    (samples, targets) <- makeCircles 3 0.6 0.1

    -- provision of initial net
    let inputFeatures = 2
    let layers = [2, 1]
    let activations = [Relu, Sigmoid]
    net <- buildNetwork inputFeatures layers activations

    -- provision of optmizer
    let learningRate = 0.001
    let gdParameters = GDParameters learningRate
    let optimizer = GradientDescent gdParameters

    -- provision of training parameters
    let epochs = 2000

    putStrLn "Finished experiment"


main = experiment


-- | TODO: figure out a way to compute time properly and compare it
-- | to python way of doing it ??
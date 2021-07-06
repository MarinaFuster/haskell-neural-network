import Numeric.LinearAlgebra ( loadMatrix, takeRows )
import Text.Printf ( printf )
import NeuralNetwork ( NeuralNetwork (..), Activation(..), Error (..), feedforward )
import Optimization ( Optimizer(..), train )
import Dynamic ( buildNetwork )
import Formatting
import Formatting.Clock
import System.Clock
import Control.Exception

main = do
    samples <- loadMatrix "datasets/spiral_x.dat"
    targets <- loadMatrix "datasets/spiral_y.dat"

    net0 <- buildNetwork 2 [5, 1] [Relu, Sigmoid]

    let optimizer = GradientDescent 0.001
        errFunction = MSE
        epochs = 1000
    
    start <- getTime Monotonic
    
    let trainedNet0 = train net0 errFunction (samples, targets) epochs optimizer
    print $ takeRows 1 (feedforward trainedNet0 samples)
    
    end <- getTime Monotonic
    
    fprint (timeSpecs) start end

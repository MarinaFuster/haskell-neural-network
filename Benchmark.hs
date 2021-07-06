import Numeric.LinearAlgebra ( Matrix (..), loadMatrix )
import Text.Printf ( printf )
import NeuralNetwork ( NeuralNetwork (..), Activation(..), Error (..) )
import Optimization ( Optimizer(..), train )
import Dynamic ( buildNetwork )
import Criterion.Main

optimizer = GradientDescent 0.001
errFunction = MSE
epochs = 3000

trainNet :: Matrix Double -> Matrix Double -> Int -> NeuralNetwork -> NeuralNetwork
trainNet samples targets epochs net =  train net errFunction (samples, targets) epochs optimizer

benchmarkNet = trainNet
benchmarkEpochs samples targets = flip (trainNet samples targets)

main = do
    samples <- loadMatrix "datasets/spiral_x.dat"
    targets <- loadMatrix "datasets/spiral_y.dat"

    net0 <- buildNetwork 2 [5, 1] [Relu, Sigmoid]
    net1 <- buildNetwork 2 [5, 10, 15, 1] [Relu, Relu, Relu, Sigmoid]
    net2 <- buildNetwork 2 [16, 32, 64, 128, 64, 32, 16, 1] 
        [Relu, Relu, Relu, Relu, Relu, Relu, Relu, Sigmoid]

    let optimizer = GradientDescent 0.001
        errFunction = MSE
        epochs = 1000

    defaultMain [ 
        bgroup "benchmark NN" [ 
            bench "net_0"  $ whnf (benchmarkNet samples targets epochs) net0,
            bench "net_1"  $ whnf (benchmarkNet samples targets epochs) net1,
            bench "net_2"  $ whnf (benchmarkNet samples targets epochs) net2,
            bench "epochs_5e3"  $ whnf (benchmarkEpochs samples targets net1) 5000,
            bench "epochs_5e4"  $ whnf (benchmarkEpochs samples targets net1) 50000,
            bench "epochs_1e5"  $ whnf (benchmarkEpochs samples targets net1) 100000
            ]
        ]

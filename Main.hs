import Numeric.LinearAlgebra
import Text.Printf ( printf )
import NeuralNetwork

program = do

    net <- buildNetwork 2 [128, 1] [Relu, Sigmoid]
    putStrLn "Network build"


main = program
import Numeric.LinearAlgebra ( loadMatrix, takeRows )
import Text.Printf ( printf )
import NeuralNetwork ( Activation(..), Error (..), feedforward )
import Optimization ( AParameters (..), Optimizer(..), train )
import Dynamic ( buildNetwork )
import Evaluation ( binaryClassify, binaryHits, binaryAccuracy )

-- | Experiment example

experimentTemplate :: IO ()
experimentTemplate = do

    -- provision of dataset

    samples <- loadMatrix "datasets/iris_x.dat"  -- PARAM: change datasets/iris_x.dat for another dataset
    targets <- loadMatrix "datasets/iris_y.dat"  -- PARAM: change datasets/iris_y.dat for true values

    -- provision of initial net

    let inputFeatures = 4                        -- PARAM: quantity of columns of datasets/iris_x.dat
        layers = [128, 3]                        -- PARAM: quantity of neurons for each layer of your nn
        activations = [Relu, Sigmoid]            -- PARAM: activation function for each layer of your nn
    
    net <- buildNetwork inputFeatures layers activations

    -- provision of train parameters

    let learningRate = 0.001                     -- PARAM: learning rate for your optimizer
        optimizer = GradientDescent learningRate -- PARAM: optimizer for your nn
        errFunc = MSE                            -- PARAM: error function to minimize
        epochs = 7000                            -- PARAM: for how many epochs will the nn train
        
        trainedNet = train net errFunc (samples, targets) epochs optimizer

    print $ takeRows 10 (feedforward trainedNet samples)

adamVsGradientDescent :: IO ()
adamVsGradientDescent = do

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
        gdOptimizer = GradientDescent learningRate -- gradient descent optimizer
        
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        lr = 0.001
        adamParams = AParameters beta1 beta2 epsilon lr
        adamOptimizer = Adam adamParams            -- adam optimizer
        
        epochs = 800
        errFunc = MSE          
        
        trainable = train net errFunc (samples, targets) epochs

        gradientNet = trainable gdOptimizer
        adamNet = trainable adamOptimizer

    print $ takeRows 10 (feedforward gradientNet samples)
    print $ takeRows 10 (feedforward adamNet samples)

spiralExperiment = do
    -- provision of dataset

    samples <- loadMatrix "datasets/spiral_x.dat"  
    targets <- loadMatrix "datasets/spiral_y.dat"

    net <- buildNetwork 2 [5, 10, 15, 1] [Relu, Relu, Relu, Sigmoid]

    let optimizer = GradientDescent 0.001
        errFunc = MSE                          
        epochs = 100000
        
        trainedNet = train net errFunc (samples, targets) epochs optimizer

    print $ takeRows 10 (feedforward trainedNet samples)

main = experimentTemplate

data Parameters a = GDParameters a 
    | BGDParameters a
    | AParameters a

data Optimizer = GradientDescent Parameters
    | BatchGradientDescent Parameters 
    | Adam Parameters

optimize :: NeuralNetowrk               -- network to train
    -> Optimizer                        -- optimizer to use
    -> Parameters                       -- parameters for that optimizer
    -> (Matrix Double, Matrix Double)   -- (samples, targets)
    -> Int                              -- epochs
    -> NeuralNetowrk                    -- resulting neural network

using ScikitLearn
using Flux
using Flux.Losses
using Flux: onehotbatch, onecold
using JLD2, FileIO
using Statistics: mean
using Flux: params
using XLSX: readdata;
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

function modelCrossValidation2(modelHyperparameters::Dict, inputs::Matrix{Int64}, dataset::Matrix{Float32}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})

    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==length(targets));
    classes = unique(targets);

    targets = oneHotEncoding(targets, classes);

    # Creamos los vectores para las metricas que se vayan a usar
    # En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);
    testconfMatrix = Array{Array{Float64, 2}, 1}(undef, numFolds);

    # Para cada fold, entrenamos
    for numFold in 1:numFolds
            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold,:];
            testTargets       = targets[crossValidationIndices.==numFold,:];
            # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que entrenar varias veces, y
            #  se crean vectores adicionales para almacenar las metricas para cada entrenamiento
            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testF1EachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testconfMatrixEachRepetition = Array{Array{Float64, 2}, 1}(undef, modelHyperparameters["numExecutions"]);
            # Se entrena las veces que se haya indicado
            for numTraining in 1:modelHyperparameters["numExecutions"]
                if modelHyperparameters["validationRatio"]>0
                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
                    # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA

                    # Entrenamos la RNA, teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassConv(modelHyperparameters["topology"],(trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                    validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                    testDataset =       (testInputs,                          testTargets);
                    maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"], 
                    maxEpochsVal=modelHyperparameters["maxEpochsVal"], dataset);
                else

                    # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test,
                    #  teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassConv(modelHyperparameters["topology"], (trainingInputs, trainingTargets),
                        testDataset = (testInputs,     testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"],dataset);

                end;
                # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
                (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], testconfMatrixEachRepetition[numTraining]) = confusionMatrix(collect(ann(Float32.(getInstancia(vec(testInputs),dataset,20)))'), testTargets);
            end;

            acc = mean(testAccuraciesEachRepetition);
            F1  = mean(testF1EachRepetition);
            confMatrix = mean(testconfMatrixEachRepetition);
        # Almacenamos las 2 metricas que usamos en este problema
        testAccuracies[numFold] = acc;
        testF1[numFold]         = F1;
        testconfMatrix[numFold] = confMatrix;

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

    end; # for numFold in 1:numFolds

    println(": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println(": Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1), 10*mean(testconfMatrix));
end;


function trainClassConv(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, kFoldIndices::     Array{Int64,1}; 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, numRepetitionsANNTraining::Int=1, validationRatio::Real=0.0,
    maxEpochsVal::Int=20,dataset::Matrix{Float32})

    numFolds = maximum(kFoldIndices);

    # Creamos los vectores para las metricas que se vayan a usar
    # En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Dividimos los datos en entrenamiento y test
        trainingInputs    = inputs[kFoldIndices.!=numFold,:];
        testInputs        = inputs[kFoldIndices.==numFold,:];
        trainingTargets   = targets[kFoldIndices.!=numFold,:];
        testTargets       = targets[kFoldIndices.==numFold,:];

        # En el caso de entrenar una RNA, este proceso es no determinístico, por lo que es necesario repetirlo para cada fold
        # Para ello, se crean vectores adicionales para almacenar las metricas para cada entrenamiento
        testAccuraciesEachRepetition = Array{Float64,1}(undef, numRepetitionsANNTraining);
        testF1EachRepetition         = Array{Float64,1}(undef, numRepetitionsANNTraining);

        for numTraining in 1:numRepetitionsANNTraining
            if validationRatio>0
                (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), validationRatio*size(trainingInputs,1)/size(inputs,1));
                # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA

                # Entrenamos la RNA
                ann, = trainClassConv((trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                    validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                    testDataset =       (testInputs,                          testTargets);
                    maxEpochs=numMaxEpochs, learningRate=learningRate, maxEpochsVal=maxEpochsVal,dataset);

            else

                # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test
                ann, = trainClassConv((trainingInputs, trainingTargets),
                    testDataset = (testInputs,     testTargets);
                    maxEpochs=numMaxEpochs, learningRate=learningRate,dataset);

            end;

            # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(ann(testInputs')', testTargets);

            # Almacenamos las metricas de este entrenamiento
            testAccuraciesEachRepetition[numTraining] = acc;
            testF1EachRepetition[numTraining]         = F1;

        end;

        # Almacenamos las 2 metricas que usamos en este problema
        testAccuracies[numFold] = mean(testAccuraciesEachRepetition);
        testF1[numFold]         = mean(testF1EachRepetition);

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

    end;

    println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println("Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

end;


function trainClassConv(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
    kFoldIndices::     Array{Int64,1};
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    numRepetitionsANNTraining::Int=1, validationRatio::Real=0.0,
    maxEpochsVal::Int=20,dataset::Matrix{Float32})

    (trainingInputs,   trainingTargets)   = trainingDataset;

    return trainClassConv((trainingInputs, reshape(trainingTargets, length(trainingTargets), 1)), kFoldIndices; maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal,dataset);

end;

function trainClassConv(topology::AbstractArray{<:Int,1},trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20, dataset::Matrix{Float32}, showText::Bool=false)

    (trainingInputs,   trainingTargets)   = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs,       testTargets)       = testDataset;
    
    if (!isempty(validationInputs)) 
        validationInputs = Float32.(getInstancia(vec(validationInputs),dataset,20));
    end;

    trainingInputs = Float32.(getInstancia(vec(trainingInputs),dataset,20));
    testInputs = Float32.(getInstancia(vec(testInputs),dataset,20));

    train_labels = trainingTargets;
    test_labels = testTargets;

    # Creamos la RNA
    # Creamos los indices: partimos el vector 1:N en grupos de batch_si
    # Creamos un batch similar, pero con todas las imagenes de test

    funcionTransferenciaCapasConvolucionales = relu;
    ann = Chain(
        Conv((3,1), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3,1), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3,1), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(160, topology[1]),
        Dense(topology[1], topology[2]),
        Dense(topology[2], 5),
        softmax
    )

    entradaCapa = trainingInputs;
    numCapas = length(topology);
    for numCapa in 1:numCapas
        capa = ann[numCapa];
        salidaCapa = capa(entradaCapa);
        entradaCapa = salidaCapa;
    end

    ann(trainingInputs);
    # Definimos la funcion de loss
    loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    accuracy(inputs,targets) = mean(onecold(ann(inputs)) .== onecold(targets));

    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses   = Float32[];
    validationLosses = Float32[];
    testLosses       = Float32[];

    # Empezamos en el ciclo 0
    opt = ADAM(0.001);
    mejorPrecision = -Inf;
    criterioFin = false;
    numCiclo = 0;
    numCicloUltimaMejora = 0;
    mejorModelo = nothing;

    function calculateLossValues()
        # Calculamos el loss en entrenamiento, validacion y test. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        trainingLoss = loss(trainingInputs, trainingTargets');
        showText && print("Epoch ", numCiclo, ": Training loss: ", trainingLoss);
        push!(trainingLosses, trainingLoss);
        if !isempty(validationInputs)
            validationLoss = loss(validationInputs, validationTargets');
            showText && print(" - validation loss: ", validationLoss);
            push!(validationLosses, validationLoss);
        else
            validationLoss = NaN;
        end;
        if !isempty(testInputs)
            testLoss = loss(testInputs, testTargets');
            showText && print(" - test loss: ", testLoss);
            push!(testLosses, testLoss);
        else
            testLoss = NaN;
        end;
        showText && println("");
        return (trainingLoss, validationLoss, testLoss);
    end;

    (trainingLoss, validationLoss, _) = calculateLossValues();
    numEpochsValidation = 0; bestValidationLoss = validationLoss;
    while (!criterioFin && (numCiclo<maxEpochs) && (numEpochsValidation<maxEpochsVal) && (trainingLoss>minLoss) )

        # Se entrena un ciclo
        Flux.train!(loss, params(ann), [(trainingInputs, trainingTargets')], opt);
        numCiclo += 1;
        (trainingLoss, validationLoss, _) = calculateLossValues();
        # Se calcula la precision en el conjunto de entrenamiento:+
        precisionEntrenamiento = mean(accuracy(trainingInputs,train_labels'));
        # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
        if (precisionEntrenamiento >= mejorPrecision)
            mejorPrecision = precisionEntrenamiento;
            precisionTest = accuracy(testInputs,test_labels');
            mejorModelo = deepcopy(ann);
            numCicloUltimaMejora = numCiclo;
        end

        # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
        if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
            opt.eta /= 10.0
            numCicloUltimaMejora = numCiclo;
        end

        # Criterios de parada:

        # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
        if (precisionEntrenamiento >= 0.999)
            criterioFin = true;
        end

        # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
        if (numCiclo - numCicloUltimaMejora >= 10)
            criterioFin = true;
        end

        # Calculamos los valores de loss para este ciclo
        (trainingLoss, validationLoss, _) = calculateLossValues();
        # Aplicamos la parada temprana si hay conjunto de validacion
        if (!isempty(validationInputs))
            if (validationLoss<bestValidationLoss)
                bestValidationLoss = validationLoss;
                mejorModelo = deepcopy(ann);
            else
                numEpochsValidation += 1;
            end;
        end; 
    end;   
        # Si no hubo conjunto de validacion, la mejor RNA será siempre la del último ciclo
        return (mejorModelo, trainingLosses, validationLosses, testLosses);
    end;
    
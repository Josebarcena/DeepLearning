import Pkg;
using Random
using Random:seed!
using Flux;
using Flux.Losses;
using Flux: params
using XLSX: readdata;
using Statistics;
seed!(1);
include("../../aa/Funciones/p1_bole2_AA.jl");
include("../../aa/Funciones/p1_bole3_AA.jl");
include("../../aa/Funciones/p1_bole4_AA.jl");
include("../../aa/Funciones/p1_bole5_AA.jl");
include("../../aa/Funciones/p1_bole6_AA.jl");

inputs106  = readdata("../../e0106/e0106.xlsx","e0106","A1:A10000");
targets106 = readdata("../../e0106/e0106.xlsx","e0106","B1:B10000");
inputs208= readdata("../../e0208-2(1).xlsx","e0208","A1:E100");
targets208= readdata("../../e0208-2(1).xlsx","e0208","F1:F100");

targets208 = reshape(targets208,size(targets208,1));

posicionesInst=[471,727,1258,2075,2619,3667,4422,4673,6193,6443,7189,8660,9713];
booleanosInst=[1,1,1,1,1,1,1,1,1,1,1,1,1];
inputs106=getInstancia2(posicionesInst,booleanosInst,Float64.(inputs106),20);

targets=[booleanosInst;Int64.(targets208)];
inputs=[Float64.(inputs106);Float64.(inputs208)];
normalizeZeroMean!(Float64.(inputs));
targetsEncod = oneHotEncoding(targets);
numFolds = 2;
crossValidationIndices = crossvalidation(targets, numFolds);

##########################################################################################################################################    
############################################ Entrenamos las RR.NN.AA. ####################################################################
##########################################################################################################################################

    learningRate=0.01;
    numMaxEpochs=1000;
    validationRatio= 0.2;
    testRatio= 0.2;
    maxEpochsVal= 6;
    topology=[35,40];
    numRepetitionsANNTraining = 50;
        
    modelHyperparameters = Dict();
    modelHyperparameters["topology"] = topology;
    modelHyperparameters["learningRate"] = learningRate;
    modelHyperparameters["validationRatio"] = validationRatio;
    modelHyperparameters["numExecutions"] = numRepetitionsANNTraining;
    modelHyperparameters["maxEpochs"] = numMaxEpochs;
    modelHyperparameters["maxEpochsVal"] = maxEpochsVal;

    cross = modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

    open("../../results/RNA crossvalidation.txt","a") do io
       println(io, "\nparametros", modelHyperparameters);
       println(io,"validacion sin %=",cross[1:4]);
    end;
    
    open("../../results/RNA confusion.txt","a") do io
        println(io, "parametros", modelHyperparameters);
        println(io, "matriz confusion=",cross[5]);
     end;

##########################################################################################################################################
######################################################## Entrenamos las SVM ##############################################################
##########################################################################################################################################
    kernel = "linear";
    kernelDegree = 8;
    kernelGamma = 5;
    C=2;
    
    modelHyperparameters = Dict();
    modelHyperparameters["kernel"] = kernel;
    modelHyperparameters["kernelDegree"] = kernelDegree;
    modelHyperparameters["kernelGamma"] = kernelGamma;
    modelHyperparameters["C"] = C;

    cross = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

    open("../../results/SVM crossvalidation.txt","a") do io
        println(io, "\nparametros", modelHyperparameters);
        println(io,"validacion sin %=",cross[1:4]);
     end;

    open("../../results/SVM confusion.txt","a") do io
        println(io, "\nparametros", modelHyperparameters);
        println(io, "matriz confusion=",cross[5]);
     end;

##########################################################################################################################################
##################################### Entrenamos los arboles de decision #################################################################
##########################################################################################################################################
    maxDepth = 39;

    cross = modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, crossValidationIndices);

    open("../../results/DT crossvalidation.txt","a") do io
        println(io, "\nprofundidad=", maxDepth);
        println(io,"validacion sin %=",cross[1:4]);
     end;
    
    open("../../results/DT confusion.txt","a") do io
        println(io, "\nprofundidad=", maxDepth);
        println(io,"matriz confusion=",cross[5]);

     end;
##########################################################################################################################################
############################################ Entrenamos los kNN ##########################################################################
##########################################################################################################################################
    numNeighbors = 5;
        
    cross = modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, crossValidationIndices);

    open("../../results/KNN crossvalidation.txt","a") do io
        println(io, "\nnumero vecinos=", numNeighbors);
        println(io,"validacion sin %=",cross[1:4]);
     end;

    open("../../results/KNN confusion.txt","a") do io
        println(io, "\nnumero vecinos=", numNeighbors);
        println(io,"matriz confusion=",cross[5]);
    end;
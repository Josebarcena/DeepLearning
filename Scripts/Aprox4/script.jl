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


####################################################################################################################################   
############################################ DATOS PACIENTE 106 ####################################################################
####################################################################################################################################

inputs106  = Float64.(readdata("../../e0106/e0106.xlsx","e0106","A1:A10000"));
targets106 = Float64.(readdata("../../e0106/e0106.xlsx","e0106","B1:B10000"));
posicionesInst1 = Int64.(readdata("../../e0106/e0106.xlsx","aprox3","A1:A50"));
booleanosInst1 = Int64.(readdata("../../e0106/e0106.xlsx","aprox3","B1:B50"));

posicionesInst1 = reshape(posicionesInst1,size(posicionesInst1,1));
booleanosInst1 = reshape(booleanosInst1, size(booleanosInst1,1));

inputs106 = Float64.(getInstancia3(posicionesInst1,booleanosInst1,inputs106,20,10));
normalizeZeroMean!(inputs106);

####################################################################################################################################   
############################################ DATOS PACIENTE 208 ####################################################################
####################################################################################################################################


inputs208 = Float64.(readdata("../../e0208/e0208.xlsx","e0208","A1:A7000"));
targets208 = Float64.(readdata("../../e0208/e0208.xlsx","e0208","B1:B7000"));
posicionesInst2 = Int64.(readdata("../../e0208/dAprox4.xlsx","e0208","A1:A100"));
booleanosInst2 = Int64.(readdata("../../e0208/dAprox4.xlsx","e0208","B1:B100"));

posicionesInst2 = reshape(posicionesInst2,size(posicionesInst2,1));
booleanosInst2 = reshape(booleanosInst2, size(booleanosInst2,1));

inputs208 = Float64.(getInstancia3(posicionesInst2,booleanosInst2,inputs208,20,10));
normalizeZeroMean!(inputs208);

targets=vec([booleanosInst1;booleanosInst2]);
inputs=Float64.([inputs106;inputs208]);

targetsEncod = oneHotEncoding(targets);
numFolds = 10;
crossValidationIndices = crossvalidation(targets, numFolds);



##########################################################################################################################################    
############################################ Entrenamos las RR.NN.AA. ####################################################################
##########################################################################################################################################

for i in 1:8

    learningRate=0.01;
    numMaxEpochs=1000;
    validationRatio= 0.2;
    testRatio= 0.2;
    maxEpochsVal= 6;
    topology=[[5,7],[5,30],[10,1],[10,7],[15,8],[25,8],[35,8],[45,8]];
    numRepetitionsANNTraining = 50;
        
    modelHyperparameters = Dict();
    modelHyperparameters["topology"] = topology[i];
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
    end;
##########################################################################################################################################
######################################################## Entrenamos las SVM ##############################################################
##########################################################################################################################################

for i in 1:8     
    kernel = ["linear","linear","rbf","rbf","poly","poly","sigmoid","sigmoid"];
    kernelDegree = [1,1,1,1,3,2,1,1];
    kernelGamma = [1,2,1,2,2,3,2,3];
    C=[2,1,1,1,1,4,1,2];
    
    modelHyperparameters = Dict();
    modelHyperparameters["kernel"] = kernel[i];
    modelHyperparameters["kernelDegree"] = kernelDegree[i];
    modelHyperparameters["kernelGamma"] = kernelGamma[i];
    modelHyperparameters["C"] = C[i];

    cross = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

    open("../../results/SVM crossvalidation.txt","a") do io
        println(io, "\nparametros", modelHyperparameters);
        println(io,"validacion sin %=",cross[1:4]);
     end;

    open("../../results/SVM confusion.txt","a") do io
        println(io, "\nparametros", modelHyperparameters);
        println(io, "matriz confusion=",cross[5]);
     end;
end;
##########################################################################################################################################
##################################### Entrenamos los arboles de decision #################################################################
##########################################################################################################################################
    for i in 1:6
    maxDepth = [1,8,125,39,55,90];

    cross = modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth[i]), inputs, targets, crossValidationIndices);

    open("../../results/DT crossvalidation.txt","a") do io
        println(io, "\nprofundidad=", maxDepth[i]);
        println(io,"validacion sin %=",cross[1:4]);
     end;
    
    open("../../results/DT confusion.txt","a") do io
        println(io, "\nprofundidad=", maxDepth[i]);
        println(io,"matriz confusion=",cross[5]);

     end;
end;
##########################################################################################################################################
############################################ Entrenamos los kNN ##########################################################################
##########################################################################################################################################
    for i in 1:6
    numNeighbors = [3,9,12,15,31,50];
        
    cross = modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors[i]), inputs, targets, crossValidationIndices);

    open("../../results/KNN crossvalidation.txt","a") do io
        println(io, "\nnumero vecinos=", numNeighbors[i]);
        println(io,"validacion sin %=",cross[1:4]);
     end;

    open("../../results/KNN confusion.txt","a") do io
        println(io, "\nnumero vecinos=", numNeighbors[i]);
        println(io,"matriz confusion=",cross[5]);
    end;
end;
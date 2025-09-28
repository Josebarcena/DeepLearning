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
include("../../aa/Funciones/p1_bole7.jl");


########################################################################################
##################################### E0106 ############################################
########################################################################################

inputs106  = Float32.(readdata("../../BDD/Aprox1/e0106.xlsx","e0106","A1:A10000"));
targets106 = Int64.(readdata("../../BDD/Aprox1/e0106.xlsx","e0106","B1:B10000"));

normalizeZeroMean!(inputs106);

posicionesInst1 = Int64.(readdata("../../e0106/e0106.xlsx","aprox3","A1:A50"));
booleanosInst1 = Int64.(readdata("../../e0106/e0106.xlsx","aprox3","B1:B50"));

posicionesInst1 = reshape(posicionesInst1,size(posicionesInst1,1));
booleanosInst1 = reshape(booleanosInst1, size(booleanosInst1,1));

########################################################################################
##################################### E0208 ############################################
########################################################################################

inputs208 = Float32.(readdata("../../e0208/e0208.xlsx","e0208","A1:A7000"));
targets208 = Int64.(readdata("../../e0208/e0208.xlsx","e0208","B1:B7000"));

normalizeZeroMean!(inputs208);

posicionesInst2 = Int64.(readdata("../../e0208/dAprox4.xlsx","e0208","A1:A100"));
booleanosInst2 = Int64.(readdata("../../e0208/dAprox4.xlsx","e0208","B1:B100"));

posicionesInst2 = reshape(posicionesInst2,size(posicionesInst2,1));
booleanosInst2 = reshape(booleanosInst2, size(booleanosInst2,1));


inputs = [inputs106;inputs208];
posicionesInst2 = posicionesInst2 .+10000
posiciones = [posicionesInst1;posicionesInst2];
posiciones = reshape(posiciones,(length(posiciones)),1)


targets = [booleanosInst1;booleanosInst2];

numFolds = 10;
crossValidationIndices = crossvalidation(targets, numFolds);

learningRate=0.01;
numMaxEpochs=1000;
validationRatio= 0.2;
testRatio= 0.2;
maxEpochsVal= 6;
numRepetitionsANNTraining = 50;

modelHyperparameters = Dict();
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;
modelHyperparameters["numExecutions"] = numRepetitionsANNTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;

for i in 5:8
local topology=[[9,6],[12,9],[15,12],[21,15],[24,18],[30,21],[33,24],[36,27]];
modelHyperparameters["topology"] = topology[i];
local cross = modelCrossValidation2(modelHyperparameters, posiciones, inputs, targets, crossValidationIndices);

    open("../../results/Deep crossvalidation.txt","a") do io
       println(io, "\nparametros", modelHyperparameters);
       println(io,"validacion sin %=",cross[1:4]);
    end;
    
    open("../../results/Deep confusion.txt","a") do io
        println(io, "parametros", modelHyperparameters);
        println(io, "matriz confusion=",cross[5]);
     end;    
end;

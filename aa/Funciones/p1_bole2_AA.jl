import Pkg;
using Statistics;
using Flux;
using Flux.Losses;



# -------------------------------------------------------------------------
# Funciones para codificar entradas y salidas categóricas

# Funcion para realizar la codificacion, recibe el vector de caracteristicas (uno por patron), y las clases
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    # Primero se comprueba que todos los elementos del vector esten en el vector de clases (linea adaptada del final de la practica 4)
    @assert(all([in(value, classes) for value in feature]));
    numClasses = length(classes);
    @assert(numClasses>1)
    if (numClasses==2)
        # Si solo hay dos clases, se devuelve una matriz con una columna
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        # Si hay mas de dos clases se devuelve una matriz con una columna por clase
        # Cualquiera de estos dos tipos (Array{Bool,2} o BitArray{2}) vale perfectamente
        # oneHot = Array{Bool,2}(undef, length(targets), numClasses);
        oneHot =  BitArray{2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;
# Esta funcion es similar a la anterior, pero si no es especifican las clases, se toman de la propia variable
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));
# Sobrecargamos la funcion oneHotEncoding por si acaso pasan un vector de valores booleanos
#  En este caso, el propio vector ya está codificado, simplemente lo convertimos a una matriz columna
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);
# Cuando se llame a la funcion oneHotEncoding, según el tipo del argumento pasado, Julia realizará
#  la llamada a la función correspondiente


# -------------------------------------------------------------------------
# Funciones para calcular los parametros de normalizacion y normalizar

# Para calcular los parametros de normalizacion, segun la forma de normalizar que se desee:
calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2}) = ( minimum(dataset, dims=1), maximum(dataset, dims=1) );
calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2}) = ( mean(dataset, dims=1), std(dataset, dims=1) );

# 4 versiones de la funcion para normalizar entre 0 y 1:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;
normalizeMinMax!(dataset::AbstractArray{<:Real,2})                                                              = normalizeMinMax!(     dataset , calculateMinMaxNormalizationParameters(dataset));
normalizeMinMax( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) = normalizeMinMax!(copy(dataset), normalizationParameters)
normalizeMinMax( dataset::AbstractArray{<:Real,2})                                                              = normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));

# 4 versiones similares de la funcion para normalizar de media 0:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= avgValues;
    dataset ./= stdValues;
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset;
end;
normalizeZeroMean!(dataset::AbstractArray{<:Real,2})                                                              = normalizeZeroMean!(     dataset , calculateZeroMeanNormalizationParameters(dataset));
normalizeZeroMean( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) = normalizeZeroMean!(copy(dataset), normalizationParameters)
normalizeZeroMean( dataset::AbstractArray{<:Real,2})                                                              = normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset));


# -------------------------------------------------------
# Funcion que permite transformar una matriz de valores reales con las salidas del clasificador o clasificadores en una matriz de valores booleanos con la clase en la que sera clasificada

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        # Miramos donde esta el valor mayor de cada instancia con la funcion findmax
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Creamos la matriz de valores booleanos con valores inicialmente a false y asignamos esos indices a true
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Comprobamos que efectivamente cada patron solo este clasificado en una clase
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;


# -------------------------------------------------------
# Funciones para calcular la precision

accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs.==targets);
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = accuracy(outputs.>=threshold, targets);
function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end;
end;


# -------------------------------------------------------
# Funciones para crear y entrenar una RNA

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    (inputs, targets) = dataset;

    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide
    @assert(size(inputs,1)==size(targets,1));

    # Creamos la RNA
    ann = buildClassANN(size(inputs,2), topology, size(targets,2));

    # Definimos la funcion de loss
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses = Float32[];

    # Empezamos en el ciclo 0
    numEpoch = 0;
    # Calculamos el loss para el ciclo 0 (sin entrenar nada)
    trainingLoss = loss(inputs', targets');
    #  almacenamos el valor de loss y precision en este ciclo
    push!(trainingLosses, trainingLoss);
    #  y lo mostramos por pantalla
    println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)

        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, params(ann), [(inputs', targets')], ADAM(learningRate));

        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;
        # Calculamos las metricas en este ciclo
        trainingLoss = loss(inputs', targets');
        #  almacenamos el valor de loss
        push!(trainingLosses, trainingLoss);
        #  lo mostramos por pantalla
        println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    end;

    # Devolvemos la RNA entrenada y el vector con los valores de loss
    return (ann, trainingLosses);
end;


trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) = trainClassANN(topology, (inputs, reshape(targets, length(targets), 1)); maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)

function toBol(Hola::Vector{Int64})
    list=Bool[];
    for hey=Hola
        if(hey==0)
            push!(list,false);
        else
            push!(list,true);  
        end;    
    end;
    return list;
end;  

function getInstancia(posiciones::Vector{<:Any},booleanos::Vector{<:Any},inputs::Matrix{Float64},amplitud::Int64)
    @assert(maximum(posiciones)+amplitud<=size(inputs,1));
    @assert(minimum(posiciones)-amplitud>0);
    @assert(size(booleanos,1)==size(posiciones,1));
    avgList = Float64[];
    desviacionList = Float64[];
    valoresList= Float64[];
    for coordenadas = posiciones
        auxiliar=inputs[coordenadas-amplitud:coordenadas+amplitud];
        valor=inputs[coordenadas];
        avg=mean(auxiliar);
        desviacion=cov(auxiliar);
        push!(avgList,avg);
        push!(desviacionList,desviacion); 
        push!(valoresList,valor);  
    end;
    return[avgList desviacionList valoresList];
end;       

function getInstancia2(posiciones::Vector{<:Any},booleanos::Vector{<:Any},inputs::Matrix{Float64},amplitud::Int64)
    @assert(maximum(posiciones)+amplitud<=size(inputs,1));
    @assert(minimum(posiciones)-amplitud>0);
    @assert(size(booleanos,1)==size(posiciones,1));
    avgList1 = Float64[];
    avgList2 = Float64[];
    desviacionList1 = Float64[];
    desviacionList2 = Float64[];
    valoresList= Float64[];
    for coordenadas = posiciones
        auxiliar1=inputs[coordenadas-amplitud:coordenadas];
        auxiliar2=inputs[coordenadas:coordenadas+amplitud];
        valor=inputs[coordenadas];
        avg1=mean(auxiliar1);
        avg2=mean(auxiliar2);
        desviacion1=cov(auxiliar1);
        desviacion2=cov(auxiliar2);
        push!(avgList1,avg1);
        push!(avgList2,avg2);
        push!(desviacionList1,desviacion1);
        push!(desviacionList2,desviacion2); 
        push!(valoresList,valor);   
    end;
    return[desviacionList1 desviacionList2 avgList1 avgList2 valoresList];
end; 

function getInstancia3(posiciones::Vector{<:Any},booleanos::Vector{<:Any},inputs::Matrix{Float64},amplitud::Int64,nextvalor::Int64)
    @assert(maximum(posiciones)+amplitud<=size(inputs,1));
    @assert(minimum(posiciones)-amplitud>0);
    @assert(size(booleanos,1)==size(posiciones,1));
    avgList1 = Float64[];
    avgList2 = Float64[];
    desviacionList1 = Float64[];
    desviacionList2 = Float64[];
    desviacionList3 = Float64[];
    valoresList= Float64[];
    for coordenadas = posiciones
        auxiliar1=inputs[coordenadas-amplitud:coordenadas];
        auxiliar2=inputs[coordenadas:coordenadas+amplitud];
        auxiliar3=inputs[coordenadas:coordenadas+nextvalor];
        valor=inputs[coordenadas];
        avg1=mean(auxiliar1);
        avg2=mean(auxiliar2);
        desviacion1=std(auxiliar1);
        desviacion2=std(auxiliar2);
        desviacion3=std(auxiliar3);
        push!(avgList1,avg1);
        push!(avgList2,avg2);
        push!(desviacionList1,desviacion1);
        push!(desviacionList2,desviacion2); 
        push!(valoresList,valor);
        push!(desviacionList3,desviacion3);   
    end;
    return[desviacionList1 desviacionList2 avgList1 avgList2 valoresList desviacionList3];
end; 

function getInstancia2(posiciones::Vector{<:Any},booleanos::Vector{<:Any},inputs::Matrix{Float64},amplitud::Int64)
    @assert(maximum(posiciones)+amplitud<=size(inputs,1));
    @assert(minimum(posiciones)-amplitud>0);
    @assert(size(booleanos,1)==size(posiciones,1));
    avgList1 = Float64[];
    avgList2 = Float64[];
    desviacionList1 = Float64[];
    desviacionList2 = Float64[];
    valoresList= Float64[];
    for coordenadas = posiciones
        auxiliar1=inputs[coordenadas-amplitud:coordenadas];
        auxiliar2=inputs[coordenadas:coordenadas+amplitud];
        valor=inputs[coordenadas];
        avg1=mean(auxiliar1);
        avg2=mean(auxiliar2);
        desviacion1=std(auxiliar1);
        desviacion2=std(auxiliar2);
        push!(avgList1,avg1);
        push!(avgList2,avg2);
        push!(desviacionList1,desviacion1);
        push!(desviacionList2,desviacion2); 
        push!(valoresList,valor);   
    end;
    return[desviacionList1 desviacionList2 avgList1 avgList2 valoresList];
end; 

function toBol(Hola::Vector{Int64})
    list=Bool[];
    for hey=Hola
        if(hey==0)
            push!(list,false);
        else
            push!(list,true);  
        end;    
    end;
    return list;
end;              


function toBol(Hola::Vector{Int64})
    list=Bool[];
    for hey=Hola
        if(hey==0)
            push!(list,false);
        else
            push!(list,true);  
        end;    
    end;
    return list;
end;  

function getInstancia(posiciones::Vector{<:Any},booleanos::Vector{<:Any},inputs::Matrix{Float64},amplitud::Int64)
    @assert(maximum(posiciones)+amplitud<=size(inputs,1));
    @assert(minimum(posiciones)-amplitud>0);
    @assert(size(booleanos,1)==size(posiciones,1));
    avgList = Float64[];
    desviacionList = Float64[];
    valoresList= Float64[];
    for coordenadas = posiciones
        auxiliar=inputs[coordenadas-amplitud:coordenadas+amplitud];
        valor=inputs[coordenadas];
        avg=mean(auxiliar);
        desviacion=cov(auxiliar);
        push!(avgList,avg);
        push!(desviacionList,desviacion); 
        push!(valoresList,valor);  
    end;
    return[avgList desviacionList valoresList];
end;       

function getInstancia2(posiciones::Vector{<:Any},booleanos::Vector{<:Any},inputs::Matrix{Float64},amplitud::Int64)
    @assert(maximum(posiciones)+amplitud<=size(inputs,1));
    @assert(minimum(posiciones)-amplitud>0);
    @assert(size(booleanos,1)==size(posiciones,1));
    avgList1 = Float64[];
    avgList2 = Float64[];
    desviacionList1 = Float64[];
    desviacionList2 = Float64[];
    valoresList= Float64[];
    for coordenadas = posiciones
        auxiliar1=inputs[coordenadas-amplitud:coordenadas];
        auxiliar2=inputs[coordenadas:coordenadas+amplitud];
        valor=inputs[coordenadas];
        avg1=mean(auxiliar1);
        avg2=mean(auxiliar2);
        desviacion1=cov(auxiliar1);
        desviacion2=cov(auxiliar2);
        push!(avgList1,avg1);
        push!(avgList2,avg2);
        push!(desviacionList1,desviacion1);
        push!(desviacionList2,desviacion2); 
        push!(valoresList,valor);   
    end;
    return[desviacionList1 desviacionList2 avgList1 avgList2 valoresList];
end; 


function getInstancia(posiciones::Vector{<:Any},inputs::Matrix{Float32},amplitud::Int64)
    @assert(maximum(posiciones)+amplitud<=size(inputs,1));
    @assert(minimum(posiciones)-amplitud>0);
    inputsDL = Array{Float32,4}(undef, 2*amplitud+1,1,1,length(posiciones));
    n = 1;
    for coordenadas in posiciones
        inputsDL[:,1,1,n] .= inputs[coordenadas-amplitud:coordenadas+amplitud];
        n +=1;
    end;
    return inputsDL;
end; 
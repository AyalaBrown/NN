const tf = require('@tensorflow/tfjs');
const { log10 } = require('mathjs');
const { Builder} = require('xml2js');


function calculateRSquared(yTrue, yPred) {
    const yTrueMean = tf.mean(yTrue);
    const totalSumOfSquares = tf.sum(tf.square(tf.sub(yTrue, yTrueMean)));
    const residualSumOfSquares = tf.sum(tf.square(tf.sub(yTrue, yPred)));
    const rSquared = tf.sub(1, tf.div(residualSumOfSquares, totalSumOfSquares));
    return rSquared.dataSync()[0];
}

function getStandardDeviation(array) {
    const len = array.length
    if (len) {
        const avg = getAvg(array);
        return Math.sqrt(array.map(x => Math.pow(x - avg, 2)).reduce((a, b) => a + b) / len);
    }
    return len;
}

function getAvg(array) {
    return sum(array) / array.length;
}

function stdAvg(data, name, multiplier) {
    return {
        "name": name,
        "std": getStandardDeviation(data.flatMap(i => (i[name]))),
        "avg": data.length > 0 ? getAvg(data.flatMap(i => (i[name]))) : 0,
        "multiplier": multiplier
    };
}

function minMax(data, name) {
    return data.flatMap(x => x[name]).reduce((a, b) => {
        if (a.min > b) a.min = b;
        if (a.max < b) a.max = b;
        return a;
    }, {
        min: 999999999999999999999999999999,
        max: -99999999999999999999999999999,
        name: name,
    });
}

function sum(array) {
    return array.reduce((a, b) => a + b, 0);
}

function normelizeRow(name, tmpName, data, multiplier) {
    if (Array.isArray(data[name["name"]])) {
        // If data is an array, normalize each element separately
        let result = data[name["name"]].map(value =>
            normalizeValue(value, tmpName, multiplier));
        return result;
    } else {
        return normalizeValue(data[name["name"]], tmpName, multiplier);
    }
}
  
function normalizeValue(value, tmpName, multiplier) {
    if (tmpName.max!=null && tmpName.min!=null) {
      if (tmpName.max === tmpName.min) {
        tmpName.max += 1;
      }
      return (value - tmpName.min) / (tmpName.max - tmpName.min);
    } else if (tmpName.avg!=null && tmpName.std!=null) {
      return ((value - tmpName.avg) / tmpName.std) * multiplier;
    } else if(tmpName.decrement!=null && tmpName.devider!=null){
        return (value-tmpName.decrement) / tmpName.devider;
    } else{
      // Handle other cases or return a default value
      return value;
    }
}

function normelizeData(_trainingData, input, nameOutputData, outputData, multiplier) {
    const inputArrays = _trainingData.map(data => {
        return input.map(inputItem => {
            const s = nameOutputData+'s';
            let tmpName = outputData.model[s][nameOutputData].find(x => x["$"].name == inputItem["name"])
            let x =  normelizeRow(inputItem, tmpName["$"], data, multiplier);
            return x;
        })
    })
    return tf.stack(inputArrays[0], 1);
}

let dataRet = {
    status: '',
}

async function fittingModel(model, trainingData, outputTrainingData, dataEpochs, testingData, outputTestingData) {
    let valHistory = [];
    try {
        await model.fit(trainingData, outputTrainingData, { epochs: dataEpochs, validationData: [testingData, outputTestingData] }).then((h) => {
            for (let i = 0; i < dataEpochs; i++) {
                h.history.val_loss.forEach(val_loss => {
                    valHistory.push(val_loss);
                });
            }
        });
    } catch (error) {
        throw error.message + " train model ERROR";
    }
    return valHistory;
}

async function training(post){

    let data = post["data"],
    _trainingData = data,
    dense = post["setting"]["dense"],
    dataEpochs = post["setting"]["settingsData"]["epochs"] != undefined ? post["setting"]["settingsData"]["epochs"] : 20,
    multiplier = post["setting"]["settingsData"]["multiplier"] != undefined ? post["setting"]["settingsData"]["multiplier"] : 1,
    percentOfTraining = post["setting"]["settingsData"]["percentOfTraining"] != undefined ? post["setting"]["settingsData"]["percentOfTraining"]/100 : 0,
    minCycles = post["setting"]["settingsData"]["minCycles"] != undefined ? post["setting"]["settingsData"]["minCycles"] : 50,
    numCyclesCheck = post["setting"]["settingsData"]["numCyclesCheck"] != undefined ? post["setting"]["settingsData"]["numCyclesCheck"] : minCycles / 5,
    outputData = {
        model: {
            '$': { type: 'neural net'},
            inputs: { input: [] },
            outputs: { output: [] },
            net: { layer: [] }
        }
    },
    trainingData,
    outputTrainingData,
    testingData,
    outputTestingData,
    output = post["setting"]["output"],
    input = post["setting"]["input"],
    inputLayerSize = post["setting"]["inputLayerSize"]

    try {
        if (typeof multiplier === 'string' || multiplier instanceof String)
            multiplier = parseFloat(multiplier);

        for (let i in input) {
            let obj = {}
            if(input[i]["typeOfNormalize"] == 'STD_AVG'){
                const d = stdAvg(data, input[i]["name"], multiplier)
                obj = {
                    '$': {
                        name: d.name,
                        std: d.std,
                        avg:  d.avg,
                        multiplier: d.multiplier,
                        type: 'STDAVG'
                    }
                }
            }
            else if(input[i]["typeOfNormalize"] == 'MIN_MAX'){
                const d = minMax(data, input[i]["name"])
                obj = {
                    '$': {
                        name: d.name,
                        min: d.min,
                        max:  d.max,
                        type: 'MINMAX'
                    }
                }
            }
            else {
                obj = {
                    '$': {
                        name: input[i]["name"],
                        decrement: input[i]["decrement"],
                        devider:  input[i]["divider"],
                        type: 'CONSTANT'
                    }
                }
            }
            outputData.model.inputs.input.push(obj);
        }

        for (let i in output) {
            let obj = {}
            if(output[i]["typeOfNormalize"] == 'STD_AVG'){
                const d = stdAvg(data, output[i]["name"], multiplier)
                obj = {
                    '$': {
                        name: d.name,
                        error: "",
                        std: d.std,
                        avg:  d.avg,
                        multiplier: d.multiplier,
                        type: 'STDAVG'
                    }
                }
            }
            else if(output[i]["typeOfNormalize"] == 'MIN_MAX'){
                const d = minMax(data, output[i]["name"])
                obj = {
                    '$': {
                        name: d.name,
                        error: "",
                        min: d.min,
                        max:  d.max,
                        type: 'MINMAX'
                    }
                }
            }
            else {
                obj = {
                    '$': {
                        name: output[i]["name"],
                        error: "",
                        decrement: output[i]["decrement"],
                        devider:  output[i]["divider"],
                        type: 'CONSTANT'
                    }
                }
            }
            outputData.model.outputs.output.push(obj);
        }

        // console.log("input", input);
        let X_normalized = normelizeData(_trainingData, input, "input", outputData, multiplier);
        // console.log("X_normalized before reshape: ", X_normalized.arraySync());
        // X_normalized = X_normalized.reshape([inputLayerSize, 2])
        // console.log("X_normalized", X_normalized);
        // console.log("X_normalized values: ", X_normalized.arraySync());
        let y_normalized = normelizeData(_trainingData, output, "output", outputData, multiplier).reshape([inputLayerSize]);

        const totalSamples = X_normalized.shape[0];
        const trainSize = Math.floor(percentOfTraining * totalSamples);
        const indices = tf.util.createShuffledIndices(totalSamples);
        const trainIndices = indices.slice(0, trainSize);
        const testIndices = indices.slice(trainSize);
        const trainIndicesArray = Array.from(trainIndices);
        const testIndicesArray = Array.from(testIndices);


        trainingData = tf.gather(X_normalized, trainIndicesArray);
        outputTrainingData = tf.gather(y_normalized, trainIndicesArray);
        testingData = tf.gather(X_normalized, testIndicesArray);
        outputTestingData = tf.gather(y_normalized, testIndicesArray);
        trainingData = trainingData.reshape([trainingData.shape[0],2])
        outputTrainingData = outputTrainingData.reshape([outputTrainingData.shape[0],1]);
        
        let model = tf.sequential();
        for (let i = 0; i < dense.length; i++) {
            model.add(tf.layers.dense({
                inputShape: [dense[i]["input"]],
                activation: dense[i]["activisionFunction"] !== 'NONE' ? dense[i]["activisionFunction"].toLowerCase() : undefined,
                units: dense[i]["output"],
            }));
        }
        model.summary();

        model.compile({
            loss: 'meanSquaredError', 
            optimizer: tf.train.adam(0.01), 
            metrics: ['mse']
        });

        let valHistory = [],
        cycles = 0,
        prevModel,
        bestModel
        async function fit() {

            let regression = sum(valHistory.slice(-dataEpochs)) > sum(valHistory.slice(-dataEpochs * 2, -dataEpochs));
            prevModel = model;
            valHistory = [...valHistory, ...await fittingModel(model, trainingData, outputTrainingData, dataEpochs, testingData, outputTestingData)];
            if (regression && cycles > 0 || cycles++ >= minCycles) {
                let sumMinValHistory = sum(valHistory.slice(-dataEpochs));
                if (cycles + numCyclesCheck > minCycles) numCyclesCheck = minCycles - cycles;
                if (regression) {
                    bestModel = prevModel;
                    for (let i = 0; i < numCyclesCheck; i++) {
                        valHistory.concat(await fittingModel(model, trainingData, outputTrainingData, dataEpochs, testingData, outputTestingData));
                        if (sum(valHistory.slice(-dataEpochs)) < sumMinValHistory) {
                            cycles += i;
                            await fit();
                            break;
                        }
                    }
                } else {
                    bestModel = model;
                }

                let layers = [];

                for (let layer, i = 0; i < bestModel.layers.length && (layer = bestModel.getLayer(null, i).getWeights()); i++) {
                    let weightList = layer[0].arraySync();
                    let biasList = layer[1] ? layer[1].arraySync() : [];
                    let numNeurons = weightList[0].length;

                    for (let j = 0; j < numNeurons; j++) {
                        let neuron = weightList.map(l => l[j]);
                        neuron.unshift(biasList[j]);
                        (layers[i] || (layers[i] = [])).push(neuron);
                    }

                    const l = {
                        "$" : {
                            activation:  dense[i]["activisionFunction"]
                        },
                        neuron : []
                    };
            
                    for(let n = 0; n<layers[i].length; n++){
                        let ne = { w: [] }
                        for(let w = 0; w<layers[i][n].length; w++ ){
                            ne.w.push(layers[i][n][w]);
                        }
                        l.neuron.push(ne)
                    }
                    outputData.model.net.layer.push(l)
                }

                for (let i = 0; i < outputData.model.outputs.output.length; i++){
                    console.log(`error: ${valHistory[cycles * dataEpochs - 1 + i]}`);
                    outputData.model.outputs.output[i]["$"]["error"] = valHistory[cycles * dataEpochs - 1 + i];
                }
            } else await fit();
        }

        await fit();
        console.log('after fitting');
        // console.log("yPred", yPred)
        // console.log('-----------------------------------------------------------------');
        // console.log("yTrue", yTrue);
        const builder = new Builder({ headless: true, explicitRoot: false, rootName: 'root', xmldec: { encoding: 'utf-8' } });
        return builder.buildObject(outputData);
    }
    catch (err) {
        console.error(err)
        // dataRet.message = 'external ERROR';
        // dataRet.status = 'ERROR';
        // process.stdout.write(JSON.stringify(dataRet))
        return;
    }

}


module.exports = {
    training,
};

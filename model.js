const tf = require('@tensorflow/tfjs');
const {training} = require('./train.js')
const { saveToDb} = require('./readingFromDB.js');
const { Builder} = require('xml2js');

async function runModel(data, profiles) {
    let median = [];
    let noData = [];

    console.log("Starting model training...");

    const Busses = [...new Set(data.map(row => row.idtag))];
    for (const bus of Busses) {
        let filteredData = data.filter((row) => row.idtag === bus);
        if (filteredData.length === 0) {
            console.log(`No data found with idtag ${bus}.`);
            return;
        }
        
        const uniqueSocValues = {};

        // Iterate through the array and add 'soc' values to the object
        filteredData.forEach(item => {
            uniqueSocValues[item.soc] = 0;
        });

        // Get the count of distinct 'soc' values
        const distinctSocCount = Object.keys(uniqueSocValues).length;

        // console.log(`distinctSocCount: ${distinctSocCount}`);

        if (filteredData.length === 0) {
            console.log(`No data found with bus ${bus}.`);
            return;
        }
        
        if (distinctSocCount < 50) {
            console.log('less then 50 distinct soc', distinctSocCount);
            if (distinctSocCount < 5){
                noData.push(bus);
            }
            else {
                let distinctSoc = [];
                let avgs = []
                let med = 0;
                for(let k = 0; k < filteredData.length; k++) {
                    let soc = filteredData[k]['soc'];
                    if ((soc in distinctSoc)==false) {
                        distinctSoc.push(soc);
                        avgs.push(parseInt(filteredData[k]['avgDiffInSec']));
                    }
                }
                avgs.sort((a, b) => a - b);
                if (avgs.length%2 == 0){
                    med = (avgs[avgs.length/2]+avgs[avgs.length/2-1])/2;
                }
                else{
                    med = avgs[Math.floor(avgs.length/2)];
                }
                outputData = {
                    model: {
                        '$': { type: 'simple avg2'},
                        inputs: {input: [{$:{name: 'soc'}}], input: [{$:{name: 'ampereLevel'}}]},
                        outputs: { output: [{$:{name:'scaled', avg: med/60.}}] },
                    }
                }
                console.log(`less then 10, average ${med}`);
                const builder = new Builder({ headless: true, explicitRoot: false, rootName: 'root', xmldec: { encoding: 'utf-8' } });
                const xml = builder.buildObject(outputData);
                const singleLineString = xml.replace(/\n\s*/g, '');
                await saveToDb(`${bus}`, singleLineString, profiles);
                median.push(parseInt(med));
            }
        }
        else {
            let soc = filteredData.map((row) => [row.soc]).flat();
            let amperLevel = filteredData.map((row) => [row.amperLevel]).flat();

            soc = tf.tensor2d(soc, [soc.length, 1]);
            amperLevel = tf.tensor2d(amperLevel, [amperLevel.length, 1]);

            let X = tf.concat([soc, amperLevel], 1);
            let y = tf.tensor2d(filteredData.map((row) => [row.scaled]/60),  [filteredData.length, 1]);

            let socForSent = soc.arraySync().flat();
            let ampereLevelForSent = amperLevel.arraySync().flat();
            let yForSent = y.arraySync().flat();

            const input_layer = X.shape[0];
            const percentOfTraining = 90
            const post = {
                data: [
                    {"soc":socForSent,
                    "ampereLevel":ampereLevelForSent,
                    "scaled":yForSent}
                ],
                setting: {
                    inputLayerSize: input_layer,
                    dense: [
                        {
                            input:2,
                            activisionFunction: 'SIGMOID',
                            output:50
                        },
                        {
                            input:50,
                            activisionFunction: 'SIGMOID',
                            output:25
                        },
                        {
                            input:25,
                            activisionFunction: 'LINEAR',
                            output:1 
                        }
                    ],
                    settingsData:{
                        epochs: 30,
                        multiplier: 1,
                        percentOfTraining: percentOfTraining,
                        minCycles: 50,
                        numCyclesCheck: 5
                    },
                    input: [
                        {
                            name:'soc',
                            decrement: 0,
                            divider: 100,
                            typeOfNormalize: 'CONSTANT'
                        },
                        {
                            name:'ampereLevel',
                            decrement: 1,
                            divider: 4,
                            typeOfNormalize: 'CONSTANT'
                        }
                    ],
                    output: [
                        {
                            name:'scaled',
                            typeOfNormalize:'STD_AVG'
                        }
                    ]
                },
            }
            const xml = await training(post);
            if(xml){
                const singleLineString = xml.replace(/\n\s*/g, '');
                await saveToDb(`${bus}`, singleLineString, profiles);
            }
            else{
                console.log(`No XML created.`);
                console.log("soc: ", socForSent);
                console.log("ampere: ", ampereLevelForSent);
                console.log("scaled: ", yForSent);
                return;
            }   
            median.push(parseInt(filteredData[0]['avgDiffInSec']));
        }
    }

    // Iterate through all the noData and put median instead.
    med = 0;
    if(median.length <= 0) {
        med = 60;
    }
    else {
        median.sort((a, b) => a - b);
        if(median.length % 2 == 0){
            med = (median[median.length/2-1]+median[median.length/2])/2;
        }
        else{
            med = median[Math.floor(median.length/2)];
        }
    }
    for(j in noData){
        outputData = {
            model: {
                '$': { type: 'simple avg2'},
                inputs: {input: [{$:{name: 'soc'}}], input: [{$:{name: 'ampereLevel'}}]},
                outputs: { output: [{$:{name:'scaled', avg: med/60.}}] },
            }
        }
        const builder = new Builder({ headless: true, explicitRoot: false, rootName: 'root', xmldec: { encoding: 'utf-8' }});
        const xml = builder.buildObject(outputData);
        const singleLineString = xml.replace(/\n\s*/g, '');
        console.log(singleLineString)
        await saveToDb(`${noData[j]}`, singleLineString, profiles);
        continue;
    }
    console.log("Model training finished.");
}

module.exports = {
    runModel,
}
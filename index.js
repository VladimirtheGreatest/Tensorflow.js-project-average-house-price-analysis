require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs-node');
const loadCSV = require('./load-csv');

function knn(features, labels, predictionPoint, k){
     return features
     .sub(predictionPoint)
     .pow(2)
     .sum(1)
     .pow(0.5)
     .expandDims(1) // expand dimension due to different shapes
     .concat(labels, 1) //join tensors
     .unstack() //create js array of tensors from tensor
     .sort((a, b) => a.get(0) > b.get(0) ? 1 : -1)
     .slice(0, k)
     .reduce((acc, pair) => acc + pair.get(1), 0) / k     //we are using regression so getting an average house price

}

let {features, labels, testFeatures, testLabels} = loadCSV('kc_house_data.csv', {
     shuffle: true,
     splitTest: 10,
     dataColumns: ['lat', 'long'],
     labelColumns: ['price']
});

//converting arrays to tensors
features = tf.tensor(features);
labels = tf.tensor(labels);

const result = knn(features, labels, tf.tensor(testFeatures[0]), 10 );
console.log('Guess', result, testLabels[0][0])
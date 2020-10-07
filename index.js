require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs-node');
const loadCSV = require('./load-csv');

function knn(features, labels, predictionPoint, k){
   const{mean, variance} = tf.moments(features, 0);

   const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5))

     return features
     .sub(mean)
     .div(variance.pow(0.5))
     .sub(scaledPrediction)
     .pow(2)
     .sum(1)
     .pow(0.5)
     .expandDims(1) // expand dimension due to different shapes
     .concat(labels, 1) //join tensors
     .unstack() //create js array of tensors from tensor
     .sort((a,b)=> a.arraySync()[0]> b.arraySync()[0]? 1:-1)
     .slice(0, k)
     .reduce((acc,pair)=>acc+pair.arraySync()[1],0)/k;   //we are using regression so getting an average house price

}

let {features, labels, testFeatures, testLabels} = loadCSV('kc_house_data.csv', {
     shuffle: true,
     splitTest: 10,
     dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living' ],
     labelColumns: ['price']
});

//converting arrays to tensors
features = tf.tensor(features);
labels = tf.tensor(labels);

//applying standardization -1 => 1    (value -average) / standarddeviation             standarddeviation == squared(variance)

testFeatures.forEach((testPoint, i) => {
     const result = knn(features, labels, tf.tensor(testPoint), 10 );
//error = expected value - predicted value / expected value
const err =  (testLabels[i][0] - result) / testLabels[i][0];
console.log('error', err * 100);
});

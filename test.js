// const tf = require('@tensorflow/tfjs');

// // Define input and output matrices
// const inputMatrix = tf.tensor2d([
//     [100, 230, 0.95, 100],
//     [200, 245, 0.99, 121],
//     [40, 250, 0.91, 123]
// ], [3, 4]);
// console.log(inputMatrix);
// const outputMatrix = tf.tensor2d([
//     [120, 5, 120],
//     [123, 24, 100],
//     [154, 3, 121]
// ], [3, 3]);
// console.log(outputMatrix);
// // Define the model architecture
// const model = tf.sequential();
// model.add(tf.layers.dense({ units: 4, inputShape: [4] }));
// model.add(tf.layers.dense({ units: 64 }));
// model.add(tf.layers.dense({ units: 128 }));
// model.add(tf.layers.dense({ units: 3 }));

// // Compile the model
// model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

// // Train the model
// async function trainModel() {
//     await model.fit(inputMatrix, outputMatrix, { epochs: 500, verbose: false });
//     console.log("Finished training the model!");

//     // Predict a sample input
//     const prediction = model.predict(tf.tensor2d([[120, 260, 0.98, 110]], [1, 4]));
//     prediction.print();
// }

// trainModel()//.then(() => {
//     // Plotting of loss not available directly in TensorFlow.js
//     // However, you can save the loss values during training and plot them using another library like matplotlib.js or Chart.js
// //});
const tf = require('@tensorflow/tfjs');

// Length of the arrays
const length = 100;

// Generate x array
const x = Array(length).fill(0);

// Generate y array
const y = Array(length).fill().map((_, index) => index + 33);

// Combine x and y into one tensor
const combinedTensor = tf.stack(
  Array(length).fill().map((_, index) => [x[index], y[index]]),
  0
);

console.log(combinedTensor.arraySync());


// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
Super Collider
=== */

let video;
let knnClassifier;

let poseNet;
let poses = [];
let nose;
let positionHistory = {}

let numBalls = 3; //20
let spring = 0.02;
let gravity = 0.01;
let friction = -0.1;
let balls = [];
let leftWristPosition;
let rightWristPosition;

let poseClasses = ["A","B","C"]
let currentPrediction = "";
let wristDiameter = 100;
let untouchedColor;
let touchedColor;
let minBallSize = 70;
let maxBallSize = 120;
let touchedBallSize = 40;
let lastTouchedID = -1;

let destroyTimer = 2; // seconds

function setup() {
  //createCanvas(640, 480);
  createCanvas((window.innerHeight * (4/3)), window.innerHeight);
  video = createCapture(VIDEO);
  video.size(width, height);

  // Create a new poseNet method with a single detection
  poseNet = ml5.poseNet(video, modelReady);
  knnClassifier = ml5.KNNClassifier();
  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on('pose', function(results) {
    poses = results;
  });
  // Hide the video element, and just show the canvas
  video.hide();

  createButtons();

  untouchedColor = color(255,255,255,204);		// color of balls before being touched
  touchedColor = color(255,0,0,204);			// color of balls once touched
  addBalls();
}

function modelReady() {
  select('#status').html('Model Loaded');
}

function draw() {
  // image(video, 0, 0, width/2, height); //video on canvas, position, dimensions
  translate(width,0); // move to far corner
  scale(-1.0,1.0);    // flip x-axis backwards
  image(video, 0, 0, width, height); //video on canvas, position, dimensions

  
  // image(video, 0, 0, width, height);

  // We can call both functions to draw all keypoints and the skeletons
  drawKeypoints();
  drawBalls();
  // drawSkeleton();
}

// Add the current frame from the video to the classifier
function addExample(label) {
  // Convert poses results to a 2d array [[score0, x0, y0],...,[score16, x16, y16]]
  const poseArray = poses[0].pose.keypoints.map(p => [p.score, p.position.x, p.position.y]);

  // Add an example with a label to the classifier
  knnClassifier.addExample(poseArray, label);
  updateCounts();
}

// Update the example count for each label  
function updateCounts() {
  const counts = knnClassifier.getCountByLabel();

  select('#exampleA').html(counts['A'] || 0);
  select('#exampleB').html(counts['B'] || 0);
  // select('#exampleC').html(counts['C'] || 0);
}

// A util function to create UI buttons
function createButtons() {
  // When the A button is pressed, add the current frame
  // from the video with a label of "A" to the classifier
  buttonA = select('#addClassA');
  buttonA.mousePressed(function() {
    addExample('A');
  });

  // When the B button is pressed, add the current frame
  // from the video with a label of "B" to the classifier
  buttonB = select('#addClassB');
  buttonB.mousePressed(function() {
    addExample('B');
  });

  // Reset buttons
  resetBtnA = select('#resetA');
  resetBtnA.mousePressed(function() {
    clearLabel('A');
  });
  
  resetBtnB = select('#resetB');
  resetBtnB.mousePressed(function() {
    clearLabel('B');
  });

  // Predict button
  buttonPredict = select('#buttonPredict');
  buttonPredict.mousePressed(classify);

  // Clear all classes button
  buttonClearAll = select('#clearAll');
  buttonClearAll.mousePressed(clearAllLabels);
  
  
  // // Load saved classifier dataset
  // buttonSetData = select('#load');
  // buttonSetData.mousePressed(loadMyKNN);

  // // Get classifier dataset
  // buttonGetData = select('#save');
  // buttonGetData.mousePressed(saveMyKNN);
  
  
}

// Predict the current frame.
function classify() {
  // Get the total number of labels from knnClassifier
  const numLabels = knnClassifier.getNumLabels();
  if (numLabels <= 0) {
    console.error('There is no examples in any label');
    return;
  }
  // Convert poses results to a 2d array [[score0, x0, y0],...,[score16, x16, y16]]
  const poseArray = poses[0].pose.keypoints.map(p => [p.score, p.position.x, p.position.y]);

  // Use knnClassifier to classify which label do these features belong to
  // You can pass in a callback function `gotResults` to knnClassifier.classify function
  knnClassifier.classify(poseArray, gotResults);
}

// Show the results
function gotResults(err, result) {
  // Display any error
  if (err) {
    console.error(err);
  }

  if (result.confidencesByLabel) {
    const confidences = result.confidencesByLabel;
    // result.label is the label that has the highest confidence
    if (result.label) {
      currentPrediction = result.label;

      // if (currentPrediction == 'A') {
      //   filter(BLUR, 3);
      // } else {
      //   filter()
      // }

      select('#result').html(result.label);
      select('#confidence').html(`${confidences[result.label] * 100} %`);
    }



    select('#confidenceA').html(`${confidences['A'] ? confidences['A'] * 100 : 0} %`);
    select('#confidenceB').html(`${confidences['B'] ? confidences['B'] * 100 : 0} %`);
  }

  classify();
}



// Clear the examples in one label
function clearLabel(classLabel) {
  knnClassifier.clearLabel(classLabel);
  updateCounts();
}

// Clear all the examples in all labels
function clearAllLabels() {
  knnClassifier.clearAllLabels();
  updateCounts();
}


// Save dataset as myKNNDataset.json
function saveMyKNN() {
    knnClassifier.save('myKNN');
}

// Load dataset to the classifier
function loadMyKNN() {
    knnClassifier.load('./myKNN.json', updateCounts);
}


function addBalls() {
	for (let i = 0; i < numBalls; i++) {
    balls[i] = new Ball(
      random(width),
      random(height),
      random(minBallSize, maxBallSize),
      i,
      balls
    );
  }
  noStroke();
}

function drawBalls() {
  let isGameOver = true
  balls.forEach(ball => {
    fill(ball.color);
    noStroke();
    ball.collide();
    if (ball.status == 'untouched') {
      isGameOver = false
    }
    ball.collideWrists(leftWristPosition, rightWristPosition);
    
    ball.move();
    ball.display();
  });

  if (isGameOver) {
    gameOver()
  }
}

// do some cool shit when you get them all
function gameOver() {
  console.log('gameOver')
  balls.forEach(ball => {
//  	ball.destroy();
	if (ball.status != "destroyed") {
		ball.destroy();
	}
  });
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints()  {
  // Loop through all the poses detected
  for (let i = 0; i < poses.length; i++) {
    // For each pose detected, loop through all the keypoints
    let pose = poses[i].pose;
    
    for (let j = 0; j < pose.keypoints.length; j++) {
      
      let keypoint = pose.keypoints[j],
        	part = keypoint.part,
          score = keypoint.score,
          position = keypoint.position
     
      // A keypoint is an object describing a body part (like rightArm or leftShoulder)
      // Only draw an ellipse is the pose probability is bigger than 0.2
      if (score > 0.2) {
   			
        if (part == "leftWrist" || part == "rightWrist") {
          
          if (part == "leftWrist") {
            leftWristPosition = position;
          } else {
            rightWristPosition = position;
          }
          
          strokeWeight(4);
          stroke('rgb(255,231,66)');
          noFill();
          // point(position.x, position.y);
          ellipse(position.x, position.y, wristDiameter, wristDiameter);
        
      	} else {
          // fill(0, 255, 0);
          strokeWeight(2);
          stroke('rgb(0,0,255)');
          point(position.x, position.y);
        }

      }
    }
  }
}

// A function to draw the skeletons
function drawSkeleton() {
  // Loop through all the skeletons detected
  for (let i = 0; i < poses.length; i++) {
    let skeleton = poses[i].skeleton;
    // For every skeleton, loop through all body connections
    for (let j = 0; j < skeleton.length; j++) {
      let partA = skeleton[j][0];
      let partB = skeleton[j][1];
      stroke(255, 0, 0);
      line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    }
  }
}

class Ball {
  constructor(xin, yin, din, idin, oin) {
    this.x = xin;
    this.y = yin;
    this.color = untouchedColor;
    this.vx = 0;
    this.vy = 0;
    this.diameter = din;
    this.id = idin;
    this.others = oin;
    this.status = "untouched";
  }

  // A right or left wrist collides with a ball
  collideWrists(leftWristPosition, rightWristPosition) {	  
  	if (leftWristPosition) {
  	  this.touch(leftWristPosition);
  	}
  	if (rightWristPosition) {
  	  this.touch(rightWristPosition);
  	}
  }
  
    // Change the properties of the ball on collision
  touch(wristPosition) {
	  	
    let dx = wristPosition.x - this.x;
    let dy = wristPosition.y - this.y;
    let distance = sqrt(dx * dx + dy * dy);
    let minDist = wristDiameter / 2 + this.diameter / 2;

    // console.log(distance);
    // console.log(minDist);

    if (distance < minDist) {
      //console.log("2");
      let angle = atan2(dy, dx);
      let targetX = this.x + cos(angle) * minDist;
      let targetY = this.y + sin(angle) * minDist;
      let ax = (targetX - wristPosition.x) * spring*50;
      let ay = (targetY - wristPosition.y) * spring*50;
      this.vx -= ax;
      this.vy -= ay;

	  // change the color of the ball if it hasn't just been collided with            
      if (this.id != lastTouchedID) {
	      if (this.status == "untouched") {
	          this.color = touchedColor;
	          this.status = "touched";
	          this.diameter = touchedBallSize;
	      } else {
	          this.color = untouchedColor;
	          this.status = "untouched";
	          this.diameter = random(minBallSize, maxBallSize);
	      }
	      lastTouchedID = this.id;
      }
    }
  }
  
  // get rid of a ball
  destroy() {
	  let timer = destroyTimer;
	  
	  // animate the ball to disappear
	  if (this.status != "destroyed" && (frameCount % 10 == 0 && timer > 0)) {
		  this.diameter = this.diameter / 2;
		  timer--;
	  }
	  
	  // actually get rid of it from the array
	  if (timer <= 0) {
		  this.status = "destroyed";
		  ball.splice(id,1);	// remove the ball
		  console.log(this.id + "destroyed");
	  }
  }
  
  collide() {
    for (let i = this.id + 1; i < numBalls; i++) {
      // console.log(others[i]);
      let dx = this.others[i].x - this.x;
      let dy = this.others[i].y - this.y;
      let distance = sqrt(dx * dx + dy * dy);
      let minDist = this.others[i].diameter / 2 + this.diameter / 2;
      //   console.log(distance);
      //console.log(minDist);
      if (distance < minDist) {
        //console.log("2");
        let angle = atan2(dy, dx);
        let targetX = this.x + cos(angle) * minDist;
        let targetY = this.y + sin(angle) * minDist;
        let ax = (targetX - this.others[i].x) * spring;
        let ay = (targetY - this.others[i].y) * spring;
        this.vx -= ax;
        this.vy -= ay;
        this.others[i].vx += ax;
        this.others[i].vy += ay;
      }
    }
  }

  move() {
    this.vy += gravity;
    this.x += this.vx;
    this.y += this.vy;
    if (this.x + this.diameter / 2 > width) {
      this.x = width - this.diameter / 2;
      this.vx *= friction;
    } else if (this.x - this.diameter / 2 < 0) {
      this.x = this.diameter / 2;
      this.vx *= friction;
    }
    if (this.y + this.diameter / 2 > height) {
      this.y = height - this.diameter / 2;
      this.vy *= friction;
    } else if (this.y - this.diameter / 2 < 0) {
      this.y = this.diameter / 2;
      this.vy *= friction;
    }
  }

  display() {
    ellipse(this.x, this.y, this.diameter, this.diameter);
  }
}


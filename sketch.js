

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

let numBalls = 2;

let spring = 0.02;
let gravity = 0.01;
let friction = 0.05;//-0.1;
let balls = [];
let leftWristPosition = {x:0, y:0};
let rightWristPosition = {x:0, y:0};

let poseClasses = ["A","B","C"]
let currentPrediction = "";
let wristDiameter = 100;
let untouchedColor;    // color of balls before being touched
let touchedColor;   // color of balls once touched
let frozenColor;// color of balls during frozone
let magnetoColor;// color of balls during frozone

let minBallSize = 70;
let maxBallSize = 120;
let touchedBallSize = 40;
let lastTouchedID = -1;
let isFrozen = false;
let isMagento = false;
var canvas;


let destroyTimer = 2; // seconds

function setup() {
  //createCanvas(640, 480);
  canvas = createCanvas(window.innerHeight * (4/3), window.innerHeight);
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
  untouchedColor = color(255,255,255,204);    // color of balls before being touched
  touchedColor = color(255,0,0,204);      // color of balls once touched
  frozenColor = color(100,100,100, 100);
  magnetoColor = color(200, 100, 8, 100)
  createButtons();

  
  addBalls();
}

function modelReady() {
  // select('#status').html('Model Loaded');
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
  select('#exampleC').html(counts['C'] || 0);
}

// A util function to create UI buttons
function createButtons() {

 greeting = createElement('h2', 'Collision Course');
 greeting.position(20, 5);

 //Buttons for training
 //class A
 btnA = select('#addClassA');
 btnA.mousePressed(function() {
   addExample('A');
 });
 btnA.position(10, 55);

 resetA = select('#resetA');
 resetA.mousePressed(function() {
   clearLabel('A');
 });
 resetA.position(btnA.x + btnA.width, 55);

 expA = select('#exampleA')
 expA.position(btnA.x + btnA.width + resetA.width + 5, 55);

 confA = select('#confidenceA')
 confA.position(btnA.x + btnA.width + resetA.width + expA.width + 15, 55);

 //class B
 btnB = select('#addClassB');
 btnB.mousePressed(function() {
   addExample('B');
 });
 btnB.position(10, 75);

 resetB = select('#resetB');
 resetB.mousePressed(function() {
   clearLabel('B');
 });
 resetB.position(btnB.x + btnB.width, 75);

 expB = select('#exampleB')
 expB.position(btnB.x + btnB.width + resetB.width + 5, 75);

 confB = select('#confidenceB')
 confB.position(btnB.x + btnB.width + resetB.width + expB.width + 15, 75);

 //class C
 btnC = select('#addClassC');
 btnC.mousePressed(function() {
   addExample('C');
 });
 btnC.position(10, 95);

 resetC = select('#resetC');
 resetC.mousePressed(function() {
   clearLabel('C');
 });
 resetC.position(btnC.x + btnC.width, 95);

 expC = select('#exampleC')
 expC.position(btnC.x + btnC.width + resetC.width + 5, 95);

 confC = select('#confidenceC')
 confC.position(btnC.x + btnC.width + resetC.width + expC.width + 15, 95);


 // Predict button
 buttonPredict = select('#buttonPredict');
 buttonPredict.mousePressed(classify);
 buttonPredict.position(10, 115);

 // Clear all classes button
 buttonClearAll = select('#clearAll');
 buttonClearAll.mousePressed(clearAllLabels);
 buttonClearAll.position(buttonPredict.x + buttonPredict.width, 115);

 //save and load
 buttonLoad = select('#load');
 buttonLoad.mousePressed(loadMyKNN);
 buttonLoad.position(10, 135);

 buttonSave = select('#save');
 buttonSave.mousePressed(saveMyKNN);
 buttonSave.position(buttonLoad.x + buttonLoad.width, 135);


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

      if (currentPrediction == 'A') {
        if (!isFrozen) {
          isFrozen = true;
          // ball.frozone()
          setTimeout(() => {
            isFrozen = false;
          }, 3000)
        }
      } else if (currentPrediction == 'B' && confidences[result.label]>.9) {
        isMagento = true
        setTimeout(() => {
            isMagento = false;
          }, 3000)

      }
    }

    select('#confidenceA').html(`${confidences['A'] ? confidences['A'] * 100 : 0} %`);
    select('#confidenceB').html(`${confidences['B'] ? confidences['B'] * 100 : 0} %`);
    select('#confidenceC').html(`${confidences['C'] ? confidences['C'] * 100 : 0} %`);
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
    gameOver();
  }
}

// do some cool shit when you get them all
function gameOver() {
  console.log('gameOver')
  balls.forEach(ball => {

  	if (ball.status != "destroyed") {
  		ball.destroy();
  	}
  });
  
  textSize(100);
  textAlign(CENTER);
  fill (255,0,0);
  text("Game Over", window.innerHeight * (2/3), window.innerHeight / 2)
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

// // A function to draw the skeletons
// function drawSkeleton() {
//   // Loop through all the skeletons detected
//   for (let i = 0; i < poses.length; i++) {
//     let skeleton = poses[i].skeleton;
//     // For every skeleton, loop through all body connections
//     for (let j = 0; j < skeleton.length; j++) {
//       let partA = skeleton[j][0];
//       let partB = skeleton[j][1];
//       stroke(255, 0, 0);
//       line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
//     }
//   }
// }

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
      this.updateColor()
    }
  }

  updateColor() {
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

  restoreColor() {
    if (this.status == "untouched") {
        this.color = untouchedColor;
    } else {
        this.color = touchedColor;
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
		  balls.splice(id,1);	// remove the ball
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

  frozone() {
    if (this.color == untouchedColor) {
      this.vy = this.vy/2
      this.vx = this.vx/2
      fill(frozenColor)
      this.color = frozenColor
    }
  }

  magneto() {
    let r_dx = rightWristPosition.x - this.x;
    let r_dy = rightWristPosition.y - this.y;
    let l_dx = leftWristPosition.x - this.x;
    let l_dy = leftWristPosition.y - this.y;

    let r_distance = sqrt(r_dx * r_dx + r_dy * r_dy);
    let l_distance = sqrt(l_dx * l_dx + l_dy * l_dy);

    let closestWristPosition = leftWristPosition
    if (r_distance < l_distance) {
      closestWristPosition = rightWristPosition
    }

    if (this.color == untouchedColor) {
        this.vx = (closestWristPosition.x - this.x)/100
        this.vy = (closestWristPosition.y - this.y)/100
        fill(magnetoColor)
        this.color = magnetoColor
    }
  }

  repello() {
    this.vy = -this.vy*3
    this.vx = -this.vx*3
    this.restoreColor()
  }

  speedUp() {
    this.vy = this.vy*3
    this.vx = this.vx*3
    this.restoreColor()
  }

  move() {
    this.vy += gravity;
    this.x += this.vx;
    this.y += this.vy;
   
    if (this.x + this.diameter / 2 > width) {
      this.x = width - this.diameter / 2;
      this.vx  = -this.vx * 0.4 //*= friction;
    } else if (this.x - this.diameter / 2 < 0) {
      this.x = this.diameter / 2;
      this.vx = -this.vx * 0.4 //*= friction;
    }

    if (this.y + this.diameter / 2 > height) {
      this.y = height - this.diameter / 2;
      this.vy = -this.vy * 0.4 //*= friction;
    } else if (this.y - this.diameter / 2 < 0) {
      this.y = this.diameter / 2;
      this.vy = -this.vy * 0.4 //*= friction;
    }

    if (isFrozen) {
      this.frozone()
    } else {
      if (this.color == frozenColor) {
        this.speedUp()
      }
    }

    if (isMagento) {
      this.magneto()
    } else {
       if (this.color == magnetoColor) {
        this.repello()
      }
    }

  }

  display() {
    ellipse(this.x, this.y, this.diameter, this.diameter);
  }
}

window.onresize = function() {
  var w = window.innerWidth;
  var h = window.innerHeight;
  canvas.size(w,h);
  width = w;
  height = h;
};

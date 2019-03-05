// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
Super Collider
=== */

let video;
let poseNet;
let poses = [];
let nose;
let positionHistory = {}

let numBalls = 20;
let spring = 0.02;
let gravity = 0.01;
let friction = -0.1;
let balls = [];
let leftWristPosition;
let rightWristPosition;

let wristDiameter = 100;
let untouchedColor;
let touchedColor;
let minBallSize = 70;
let maxBallSize = 120;
let touchedBallSize = 40;
let lastTouchedID = -1;

function setup() {
  //createCanvas(640, 480);
  createCanvas((window.innerHeight * (4/3)), window.innerHeight);
  video = createCapture(VIDEO);
  video.size(width, height);

  // Create a new poseNet method with a single detection
  poseNet = ml5.poseNet(video, modelReady);
  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on('pose', function(results) {
    poses = results;
  });
  // Hide the video element, and just show the canvas
  video.hide();
  nose = loadImage("nose.png");
  textSize(16);
  
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
  //drawSkeleton();
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
  balls.forEach(ball => {
    fill(ball.color);
    noStroke();
    ball.collide();
    
    ball.collideWrists(leftWristPosition, rightWristPosition);
    
  	
    ball.move();
    ball.display();
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



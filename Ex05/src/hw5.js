import {OrbitControls} from './OrbitControls.js'

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
scene.background = new THREE.Color('ForestGreen');

function degrees_to_radians(degrees) {
	var pi = Math.PI;
	return degrees * (pi / 180);
}

const H_POST = 2
const W_POST = 0.07
const D_POSTS = 3 * H_POST
const THETA_POSTS = degrees_to_radians(45)

const goal = new THREE.Group();

const frontLeftPost = createPost(W_POST, W_POST, H_POST)
makeTranslation(frontLeftPost, -D_POSTS / 2, 0, 0);

const frontRightPost = createPost(W_POST, W_POST, H_POST)
makeTranslation(frontRightPost, D_POSTS / 2, 0, 0);

const backLeftPost = createPost(W_POST, W_POST, H_POST / Math.cos(THETA_POSTS))
makeRotationX(backLeftPost, THETA_POSTS)
makeTranslation(backLeftPost, -D_POSTS / 2, 0, -(H_POST * Math.tan(THETA_POSTS)) / 2)

const backRightPost = createPost(W_POST, W_POST, H_POST / Math.cos(THETA_POSTS));
makeRotationX(backRightPost, THETA_POSTS);
makeTranslation(backRightPost, D_POSTS / 2, 0, -(H_POST * Math.tan(THETA_POSTS)) / 2)

const crossbar = createPost(W_POST, W_POST, D_POSTS + W_POST)
makeRotationZ(crossbar, degrees_to_radians(90))
makeTranslation(crossbar, 0, H_POST / 2, 0)

const frontLeftTorus = createTorus();
makeTranslation(frontLeftTorus, -D_POSTS / 2, -H_POST / 2, 0);

const frontRightTorus = createTorus();
makeTranslation(frontRightTorus, D_POSTS / 2, -H_POST / 2, 0);

const backLeftTorus = createTorus();
makeTranslation(backLeftTorus, -D_POSTS / 2, -H_POST / 2, -(H_POST * Math.tan(THETA_POSTS)))

const backRightTorus = createTorus();
makeTranslation(backRightTorus, D_POSTS / 2, -H_POST / 2, -(H_POST * Math.tan(THETA_POSTS)))

goal.add(frontLeftPost, frontRightPost, backLeftPost, backRightPost, crossbar, frontLeftTorus, frontRightTorus, backLeftTorus, backRightTorus);

const backNetGeometry = new THREE.PlaneGeometry(D_POSTS, H_POST / Math.cos(THETA_POSTS));
const backNetMaterial = new THREE.MeshBasicMaterial({color: 0xCCCCCC, side: THREE.DoubleSide});
const backNet = new THREE.Mesh(backNetGeometry, backNetMaterial);
makeRotationX(backNet, THETA_POSTS);
makeTranslation(backNet, 0, 0, -(H_POST * Math.tan(THETA_POSTS)) / 2);

const leftNetShape = new THREE.Shape();
leftNetShape.lineTo(0, H_POST);
leftNetShape.lineTo(-(H_POST * Math.tan(THETA_POSTS)), 0);
leftNetShape.lineTo(0, 0);
const leftNetGeometry = new THREE.ShapeGeometry(leftNetShape);
const leftNetMaterial = new THREE.MeshBasicMaterial({color: 0xCCCCCC, side: THREE.DoubleSide});
const leftNet = new THREE.Mesh(leftNetGeometry, leftNetMaterial);
makeRotationY(leftNet, degrees_to_radians(-90));
makeTranslation(leftNet, -D_POSTS / 2, -H_POST / 2, 0);

const rightNetShape = new THREE.Shape();
rightNetShape.lineTo(0, H_POST);
rightNetShape.lineTo(H_POST * Math.tan(THETA_POSTS), 0);
rightNetShape.lineTo(0, 0);
const rightNetGeometry = new THREE.ShapeGeometry(rightNetShape);
const rightNetMaterial = new THREE.MeshBasicMaterial({color: 0xCCCCCC, side: THREE.DoubleSide});
const rightNet = new THREE.Mesh(rightNetGeometry, rightNetMaterial);
makeRotationY(rightNet, degrees_to_radians(90));
makeTranslation(rightNet, D_POSTS / 2, -H_POST / 2, 0)

goal.add(frontLeftPost, frontRightPost, backLeftPost, backRightPost, crossbar, frontLeftTorus, frontRightTorus,
	backLeftTorus, backRightTorus, backNet, leftNet, rightNet);
scene.add(goal);

const ballRadius = H_POST / 16;
const ballGeometry = new THREE.SphereGeometry(ballRadius, 32, 32);
const ballMaterial = new THREE.MeshBasicMaterial({color: 0x000000});
const ball = new THREE.Mesh(ballGeometry, ballMaterial);
makeTranslation(ball, 0, -H_POST / 2 + ballRadius, H_POST * Math.tan(THETA_POSTS))
scene.add(ball);

// Bonus
const confettiGroup = new THREE.Group();

// This defines the initial distance of the camera
const cameraTranslate = new THREE.Matrix4();
cameraTranslate.makeTranslation(0, 0, 5);
camera.applyMatrix4(cameraTranslate)

renderer.render(scene, camera);

const controls = new OrbitControls(camera, renderer.domElement);

let isOrbitEnabled = true;
let isWireframeEnabled = false;
let isBallVerticalRotationEnabled = false;
let isBallHorizontalRotationEnabled = false;
let isConfettiEnabled = false;
let isBallInsideGoal = false;
let ballSpeedFactor = 0.05;
let goalScaleFactor = 0.95;

const toggleOrbit = (e) => {
	if (e.key === 'o') {
		isOrbitEnabled = !isOrbitEnabled;
	}

	if (isOrbitEnabled) {
		switch (e.key) {
			case 'w':
				isWireframeEnabled = !isWireframeEnabled;
				goal.children.forEach(child => {
					child.material.wireframe = isWireframeEnabled;
				})
				ball.material.wireframe = isWireframeEnabled;
				break;

			case '1':
				isBallHorizontalRotationEnabled = !isBallHorizontalRotationEnabled;
				break;

			case '2':
				isBallVerticalRotationEnabled = !isBallVerticalRotationEnabled;
				break;

			case '3':
				makeScale(goal, goalScaleFactor);
				break;

			case '4':
				isConfettiEnabled = !isConfettiEnabled;
				break;

			case 'ArrowUp':
				ballSpeedFactor *= 2;
				break;

			case 'ArrowDown':
				ballSpeedFactor /= 2;
				break;

			default:
				break;
		}
	}
}

document.addEventListener('keydown', toggleOrbit);

//controls.update() must be called after any manual changes to the camera's transform
controls.update();

function animate() {

	requestAnimationFrame(animate);

	controls.enabled = isOrbitEnabled;

	if (isBallVerticalRotationEnabled) {
		makeRotationX(ball, ballSpeedFactor);
	}

	if (isBallHorizontalRotationEnabled) {
		makeRotationY(ball, ballSpeedFactor);
	}

	const goalBoundingBox = new THREE.Box3().setFromObject(goal);
	if (isConfettiEnabled && goalBoundingBox.containsPoint(ball.position) && !isBallInsideGoal) {
		isBallInsideGoal = true;
		createConfetti();
	}

	if (!goalBoundingBox.containsPoint(ball.position) && isBallInsideGoal) {
		isBallInsideGoal = false;
	}

	confettiGroup.children.forEach((confetti) => {
		makeTranslation(confetti, 0, -ballSpeedFactor, 0);

		if (confetti.position.y < -H_POST) {
			confettiGroup.remove(confetti);
		}
	});

	controls.update();

	renderer.render(scene, camera);
}

animate()

function makeTranslation(obj, x, y, z) {
	const translationMatrix = new THREE.Matrix4();
	translationMatrix.makeTranslation(x, y, z);
	obj.applyMatrix4(translationMatrix);
}

function makeRotationX(obj, theta) {
	const rotationMatrix = new THREE.Matrix4();
	rotationMatrix.makeRotationX(theta);
	obj.applyMatrix4(rotationMatrix);
}

function makeRotationY(obj, theta) {
	const rotationMatrix = new THREE.Matrix4();
	rotationMatrix.makeRotationY(theta);
	obj.applyMatrix4(rotationMatrix);
}

function makeRotationZ(obj, theta) {
	const rotationMatrix = new THREE.Matrix4();
	rotationMatrix.makeRotationZ(theta)
	obj.applyMatrix4(rotationMatrix);
}

function makeScale(obj, scaleFactor) {
	const scaleMatrix = new THREE.Matrix4();
	scaleMatrix.makeScale(scaleFactor, scaleFactor, scaleFactor);
	obj.applyMatrix4(scaleMatrix);
}

function createPost(radiusTop, radiusBottom, height) {
	const postGeometry = new THREE.CylinderGeometry(radiusTop, radiusBottom, height);
	const postMaterial = new THREE.MeshBasicMaterial({color: 0xFFFFFF});

	return new THREE.Mesh(postGeometry, postMaterial);
}

function createTorus() {
	const torusGeometry = new THREE.TorusGeometry(0.1, 0.05, 3, 100);
	const torusMaterial = new THREE.MeshBasicMaterial({color: 0xFFFFFF});
	const torus = new THREE.Mesh(torusGeometry, torusMaterial);
	makeRotationX(torus, degrees_to_radians(90));

	return torus;
}

function createConfetti() {
	const confettiColors = [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00];

	for (let i = 0; i < 500; i++) {
		const confettiGeometry = new THREE.PlaneGeometry(0.1, 0.1);
		const confettiMaterial = new THREE.MeshBasicMaterial({color: confettiColors[Math.floor(Math.random() * confettiColors.length)]});
		const confetti = new THREE.Mesh(confettiGeometry, confettiMaterial);

		makeTranslation(confetti, Math.random() * 6 - 3, Math.random() * 6 - 3, Math.random() * 6 - 3);
		makeRotationX(confetti, Math.random() * Math.PI);
		makeRotationY(confetti, Math.random() * Math.PI);
		makeRotationZ(confetti, Math.random() * Math.PI);

		confettiGroup.add(confetti);
	}

	scene.add(confettiGroup);
}
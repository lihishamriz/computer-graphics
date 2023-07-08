// Scene Declartion
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
// This defines the initial distance of the camera, you may ignore this as the camera is expected to be dynamic
camera.applyMatrix4(new THREE.Matrix4().makeTranslation(-5, 3, 110));
camera.lookAt(0, -4, 1)


const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);


// helper function for later on
function degrees_to_radians(degrees) {
	var pi = Math.PI;
	return degrees * (pi / 180);
}


// Here we load the cubemap and pitch images, you may change it

const loader = new THREE.CubeTextureLoader();
const texture = loader.load([
	'src/pitch/right.jpg',
	'src/pitch/left.jpg',
	'src/pitch/top.jpg',
	'src/pitch/bottom.jpg',
	'src/pitch/front.jpg',
	'src/pitch/back.jpg',
]);
scene.background = texture;


// TODO: Texture Loading
// We usually do the texture loading before we start everything else, as it might take processing time
const ballTexture = new THREE.TextureLoader().load('src/textures/soccer_ball.jpg');
const redCardTexture = new THREE.TextureLoader().load('src/textures/red_card.jpg');
const yellowCardTexture = new THREE.TextureLoader().load('src/textures/yellow_card.jpg');
const varCardTexture = new THREE.TextureLoader().load('src/textures/var_card.jpg');


// TODO: Add Lighting
const ambientLight = new THREE.AmbientLight(0xFFFFFF, 0.5);
const directionalLight_1 = new THREE.DirectionalLight(0xFFFFFF, 0.5);
const directionalLight_2 = new THREE.DirectionalLight(0xFFFFFF, 0.5);
const directionalLight_3 = new THREE.DirectionalLight(0xFFFFFF, 0.5);
scene.add(ambientLight);
scene.add(directionalLight_1);
scene.add(directionalLight_2);
scene.add(directionalLight_3);


// TODO: Goal
// You should copy-paste the goal from the previous exercise here
const H_POST = 10
const W_POST = 0.1
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
const backNetMaterial = new THREE.MeshPhongMaterial({
	color: 0xCCCCCC,
	side: THREE.DoubleSide,
	specular: 0x555555,
	shininess: 30,
	transparent: true,
	opacity: 0.5
});
const backNet = new THREE.Mesh(backNetGeometry, backNetMaterial);
makeRotationX(backNet, THETA_POSTS);
makeTranslation(backNet, 0, 0, -(H_POST * Math.tan(THETA_POSTS)) / 2);
directionalLight_1.target = backNet;

const leftNetShape = new THREE.Shape();
leftNetShape.lineTo(0, H_POST);
leftNetShape.lineTo(-(H_POST * Math.tan(THETA_POSTS)), 0);
leftNetShape.lineTo(0, 0);
const leftNetGeometry = new THREE.ShapeGeometry(leftNetShape);
const leftNetMaterial = new THREE.MeshPhongMaterial({
	color: 0xCCCCCC,
	side: THREE.DoubleSide,
	specular: 0x555555,
	shininess: 30,
	transparent: true,
	opacity: 0.5
});
const leftNet = new THREE.Mesh(leftNetGeometry, leftNetMaterial);
makeRotationY(leftNet, degrees_to_radians(-90));
makeTranslation(leftNet, -D_POSTS / 2, -H_POST / 2, 0);
directionalLight_2.target = leftNet;

const rightNetShape = new THREE.Shape();
rightNetShape.lineTo(0, H_POST);
rightNetShape.lineTo(H_POST * Math.tan(THETA_POSTS), 0);
rightNetShape.lineTo(0, 0);
const rightNetGeometry = new THREE.ShapeGeometry(rightNetShape);
const rightNetMaterial = new THREE.MeshPhongMaterial({
	color: 0xCCCCCC,
	side: THREE.DoubleSide,
	specular: 0x555555,
	shininess: 30,
	transparent: true,
	opacity: 0.5
});
const rightNet = new THREE.Mesh(rightNetGeometry, rightNetMaterial);
makeRotationY(rightNet, degrees_to_radians(90));
makeTranslation(rightNet, D_POSTS / 2, -H_POST / 2, 0);
directionalLight_3.target = rightNet

goal.add(frontLeftPost, frontRightPost, backLeftPost, backRightPost, crossbar, frontLeftTorus, frontRightTorus,
	backLeftTorus, backRightTorus, backNet, leftNet, rightNet);
scene.add(goal);


// TODO: Ball
// You should add the ball with the soccer.jpg texture here
const ballRadius = H_POST / 16;
const ballGeometry = new THREE.SphereGeometry(ballRadius, 32, 32);
const ballMaterial = new THREE.MeshPhongMaterial({map: ballTexture});
const ball = new THREE.Mesh(ballGeometry, ballMaterial);
makeTranslation(ball, 0, 0, 100)
scene.add(ball);


// TODO: Bezier Curves
const startPoint = new THREE.Vector3(0, 0, 100);
const endPoint = new THREE.Vector3(0, 0, 0);
const leftWingerRoute = new THREE.Vector3(-50, 0, 50);
const centerWingerRoute = new THREE.Vector3(0, 50, 50);
const rightWingerRoute = new THREE.Vector3(50, 0, 50);

const leftCurve = new THREE.CubicBezierCurve3(startPoint, leftWingerRoute, endPoint);
const centerCurve = new THREE.CubicBezierCurve3(startPoint, centerWingerRoute, endPoint);
const rightCurve = new THREE.CubicBezierCurve3(startPoint, rightWingerRoute, endPoint);

const curves = [leftCurve, centerCurve, rightCurve];
let selectedCurveIndex = 1;


// TODO: Camera Settings
// Set the camera following the ball here
makeTranslation(camera, 0, H_POST, 0);


// TODO: Add collectible cards with textures
class Card {
	constructor(curve, t, texture) {
		this.curve = curve;
		this.t = t;
		this.texture = texture;
		this.hit = false;
		this.createCard();
		this.setCardPosition();
		this.addCardToScene();
	}

	createCard() {
		const cardGeometry = new THREE.BoxGeometry(1, 2, 1);
		const cardMaterial = new THREE.MeshPhongMaterial({map: this.texture});
		this.object3d = new THREE.Mesh(cardGeometry, cardMaterial);
	}

	setCardPosition() {
		const point = this.curve.getPoint(this.t);
		makeTranslation(this.object3d, point.x, point.y, point.z);
	}

	addCardToScene() {
		scene.add(this.object3d);
	}
}

const textures = [yellowCardTexture, redCardTexture];
const cards = [];
const numberOfCards = 7;
const numberOfVarCards = 3;
for (let i = 1; i <= numberOfCards; i++) {
	const card = new Card(curves[i % 3], i / (numberOfCards + 1), textures[i % 2]);
	cards.push(card);
}
for (let i = 1; i <= numberOfVarCards; i++) {
	const t = 0.2 * i;
	const card = new Card(curves[i % 3], t, varCardTexture);
	cards.push(card);
}


// TODO: Add keyboard event
// We wrote some of the function for you
const handle_keydown = (e) => {
	if (e.code === 'ArrowLeft') {
		if (selectedCurveIndex > 0) {
			selectedCurveIndex -= 1
		}
	} else if (e.code === 'ArrowRight') {
		if (selectedCurveIndex < curves.length - 1) {
			selectedCurveIndex += 1
		}
	}
}
document.addEventListener('keydown', handle_keydown);

const totalIncrements = 3000;
let currentIncrement = 0;
let startPosition = rightCurve.getPoints(totalIncrements)[0];
ball.position.copy(startPosition);

function animate() {
	requestAnimationFrame(animate);
	renderer.render(scene, camera);

	// TODO: Animation for the ball's position
	const t = currentIncrement / totalIncrements;
	const position = curves[selectedCurveIndex].getPoint(t);
	ball.position.copy(position);
	currentIncrement++;

	makeRotationY(ball, 0.01);
	makeTranslation(ball, position.x - ball.position.x, position.y - ball.position.y, position.z - ball.position.z)

	const camera_new_z = ball.position.z - camera.position.z + D_POSTS;
	makeTranslation(camera, 0, 0, camera_new_z);

	// TODO: Test for card-ball collision
	for (let card of cards) {
		if (card.t >= t && card.t - t < 0.01 && card.curve === curves[selectedCurveIndex]) {
			card.hit = true;
			card.object3d.visible = false;
		}
	}

	if (currentIncrement === totalIncrements) {
		const collectedCards = cards.filter(card => card.hit === true);
		const collectedYellowCards = collectedCards.filter(card => card.texture === yellowCardTexture).length;
		const collectedRedCards = collectedCards.filter(card => card.texture === redCardTexture).length;
		const collectedVarCards = collectedCards.filter(card => card.texture === varCardTexture).length;
		const power = -(collectedYellowCards + 10 * collectedRedCards) / 10;
		const fairPlayScore = 100 * Math.pow(2, power) + 3 * collectedVarCards;
		alert("Your fair play score is: " + fairPlayScore.toFixed(2));
		window.location.reload(true);
	}
}

animate();

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
	const postMaterial = new THREE.MeshPhongMaterial({color: 0xFFFFFF});

	return new THREE.Mesh(postGeometry, postMaterial);
}

function createTorus() {
	const torusGeometry = new THREE.TorusGeometry(0.1, 0.05, 3, 100);
	const torusMaterial = new THREE.MeshPhongMaterial({color: 0xFFFFFF});
	const torus = new THREE.Mesh(torusGeometry, torusMaterial);
	makeRotationX(torus, degrees_to_radians(90));

	return torus;
}
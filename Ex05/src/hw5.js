import {OrbitControls} from './OrbitControls.js'

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );
scene.background = new THREE.Color( 'ForestGreen' );

function degrees_to_radians(degrees)
{
  var pi = Math.PI;
  return degrees * (pi/180);
}

const H_POST = 2
const W_POST = 0.07
const D_POSTS = 3 * H_POST
const THETA_POSTS = degrees_to_radians(40)

// Add here the rendering of your goal
const frontLeftPostGeometry = new THREE.CylinderGeometry(W_POST, W_POST, H_POST);
const frontLeftPostMaterial = new THREE.MeshBasicMaterial( {color: 0xFFFFFF} );
const frontLeftPost = new THREE.Mesh( frontLeftPostGeometry, frontLeftPostMaterial );
makeTranslation(frontLeftPost, -D_POSTS / 2, 0, 0);
scene.add(frontLeftPost);

const frontRightPostGeometry = new THREE.CylinderGeometry(W_POST, W_POST, H_POST);
const frontRightPostMaterial = new THREE.MeshBasicMaterial( {color: 0xFFFFFF} );
const frontRightPost = new THREE.Mesh( frontRightPostGeometry, frontRightPostMaterial );
makeTranslation(frontRightPost, D_POSTS / 2, 0, 0);
scene.add(frontRightPost)

const crossbarGeometry = new THREE.CylinderGeometry(W_POST, W_POST, D_POSTS + W_POST);
const crossbarMaterial = new THREE.MeshBasicMaterial( {color: 0xFFFFFF} );
const crossbar = new THREE.Mesh( crossbarGeometry, crossbarMaterial );
makeRotationZ(crossbar, degrees_to_radians(90))
makeTranslation(crossbar, 0, H_POST / 2, 0)
scene.add(crossbar)

const backLeftPostGeometry = new THREE.CylinderGeometry(W_POST, W_POST, H_POST / Math.cos(THETA_POSTS));
console.log(Math.cos(THETA_POSTS))
const backLeftPostMaterial = new THREE.MeshBasicMaterial( {color: 0xFFFFFF} );
const backLeftPost = new THREE.Mesh( backLeftPostGeometry, backLeftPostMaterial );
makeRotationX(backLeftPost, THETA_POSTS)
makeTranslation(backLeftPost, -D_POSTS / 2, 0, -(H_POST * Math.tan(THETA_POSTS)) / 2)
scene.add(backLeftPost);

const backRightPostGeometry = new THREE.CylinderGeometry(W_POST, W_POST, H_POST / Math.cos(THETA_POSTS));
const backRightPostMaterial = new THREE.MeshBasicMaterial( {color: 0xFFFFFF} );
const backRightPost = new THREE.Mesh( backRightPostGeometry, backRightPostMaterial );
makeRotationX(backRightPost, THETA_POSTS);
makeTranslation(backRightPost, D_POSTS / 2, 0, -(H_POST * Math.tan(THETA_POSTS)) / 2)
scene.add(backRightPost);

const frontLeftTorusGeometry = new THREE.TorusGeometry(0.1, 0.05, 2, 100)
const frontLeftTorusMaterial = new THREE.MeshBasicMaterial( {color: 0xFFFFFF} );
const frontLeftTorus = new THREE.Mesh( frontLeftTorusGeometry, frontLeftTorusMaterial );
makeRotationX(frontLeftTorus, degrees_to_radians(90));
makeTranslation(frontLeftTorus, -D_POSTS / 2, -H_POST / 2, 0);
scene.add(frontLeftTorus);

const frontRightTorusGeometry = new THREE.TorusGeometry(0.1, 0.05, 2, 100)
const frontRightTorusMaterial = new THREE.MeshBasicMaterial( {color: 0xFFFFFF} );
const frontRightTorus = new THREE.Mesh( frontRightTorusGeometry, frontRightTorusMaterial );
makeRotationX(frontRightTorus, degrees_to_radians(90))
makeTranslation(frontRightTorus, D_POSTS / 2, -H_POST / 2, 0)
scene.add(frontRightTorus);

const backLeftTorusGeometry = new THREE.TorusGeometry(0.1, 0.05, 2, 100)
const backLeftTorusMaterial = new THREE.MeshBasicMaterial( {color: 0xFFFFFF} );
const backLeftTorus = new THREE.Mesh( backLeftTorusGeometry, backLeftTorusMaterial );
makeRotationX(backLeftTorus, degrees_to_radians(90))
makeTranslation(backLeftTorus, -D_POSTS / 2, -H_POST / 2, -(H_POST * Math.tan(THETA_POSTS)))
scene.add(backLeftTorus);

const backRightTorusGeometry = new THREE.TorusGeometry(0.1, 0.05, 2, 100)
const backRightTorusMaterial = new THREE.MeshBasicMaterial( {color: 0xFFFFFF} );
const backRightTorus = new THREE.Mesh( backRightTorusGeometry, backRightTorusMaterial );
makeRotationX(backRightTorus, degrees_to_radians(90))
makeTranslation(backRightTorus, D_POSTS / 2, -H_POST / 2, -(H_POST * Math.tan(THETA_POSTS)))
scene.add(backRightTorus);



const netWidth = D_POSTS;
const netHeight = H_POST / Math.cos(THETA_POSTS);
const netGeometry = new THREE.PlaneGeometry(netWidth, netHeight);

const netMaterial = new THREE.MeshBasicMaterial({ color: 0xCCCCCC, side: THREE.DoubleSide });  // Light gray, visible from both sides

const net = new THREE.Mesh(netGeometry, netMaterial);
makeRotationX(net, THETA_POSTS);
makeTranslation(net, 0, 0, -(H_POST * Math.tan(THETA_POSTS)) / 2);
scene.add(net);

const leftNetShape = new THREE.Shape();
leftNetShape.lineTo(0, H_POST);
leftNetShape.lineTo(-(H_POST * Math.tan(THETA_POSTS)), 0);
leftNetShape.lineTo(0,0);

const leftNetGeometry = new THREE.ShapeGeometry(leftNetShape);

const leftNetMaterial = new THREE.MeshBasicMaterial({ color: 0xCCCCCC, side: THREE.DoubleSide });

const leftNetMesh = new THREE.Mesh(leftNetGeometry, leftNetMaterial);

const rotateMinus90YMatrix = new THREE.Matrix4().makeRotationY(degrees_to_radians(-90));
const translationLeftNet = new THREE.Matrix4().makeTranslation(-D_POSTS / 2, -H_POST / 2, 0);
const leftNetMatrix = translationLeftNet.multiply(rotateMinus90YMatrix);
leftNetMesh.applyMatrix4(leftNetMatrix);

scene.add(leftNetMesh);


const rightNetShape = new THREE.Shape();
rightNetShape.lineTo(0, H_POST);
rightNetShape.lineTo(H_POST * Math.tan(THETA_POSTS), 0);
rightNetShape.lineTo(0,0);

const rightNetGeometry = new THREE.ShapeGeometry(rightNetShape);

const rightNetMaterial = new THREE.MeshBasicMaterial({ color: 0xCCCCCC, side: THREE.DoubleSide });

const rightNetMesh = new THREE.Mesh(rightNetGeometry, rightNetMaterial);

const rotate90YMatrix = new THREE.Matrix4().makeRotationY(degrees_to_radians(90));
const translationRightNet = new THREE.Matrix4().makeTranslation(D_POSTS / 2, -H_POST / 2, 0);
const rightNetMatrix = translationRightNet.multiply(rotate90YMatrix);
rightNetMesh.applyMatrix4(rightNetMatrix);

scene.add(rightNetMesh);

const ballRadius  = H_POST / 16;

const balPosition = {
    x: 0,                              // Centered between the posts
    y: -H_POST / 2 + ballRadius,      // Halfway up the goal's height, accounting for the radius of the ball
    z: (H_POST * Math.tan(THETA_POSTS))    // Positioned in front of the goal by a distance of 1 unit (or your choice), accounting for the radius of the ball
};

const ballGeometry = new THREE.SphereGeometry(ballRadius, 32, 32);  // Adjust the second and third arguments for smoother sphere

const ballMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });  // White ball, adjust color as needed

const ball = new THREE.Mesh(ballGeometry, ballMaterial);

ball.position.set(balPosition.x, balPosition.y, balPosition.z);

scene.add(ball);















// This defines the initial distance of the camera
const cameraTranslate = new THREE.Matrix4();
cameraTranslate.makeTranslation(0,0,5);
camera.applyMatrix4(cameraTranslate)

renderer.render( scene, camera );

const controls = new OrbitControls( camera, renderer.domElement );

let isOrbitEnabled = true;
let isWireframeEnabled = false;

const toggleOrbit = (e) => {
	if (e.key == "o") {
		isOrbitEnabled = !isOrbitEnabled;
	}

	if (e.key == "w") {
		isWireframeEnabled = !isWireframeEnabled;
		scene.children.forEach(child => {
			child.material.wireframe = isWireframeEnabled;
		})
	}
}

document.addEventListener('keydown',toggleOrbit)

//controls.update() must be called after any manual changes to the camera's transform
controls.update();

function animate() {

	requestAnimationFrame( animate );

	controls.enabled = isOrbitEnabled;
	controls.update();

	renderer.render( scene, camera );

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


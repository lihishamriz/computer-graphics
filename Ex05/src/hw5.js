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

import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import GestureTrainer from './gesture-trainer.js';

// --- HAND TRACKING SETUP ---
let handLandmarker = undefined;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastVideoTimeInternal = -1;
let results = undefined;

// --- STATE MANAGEMENT ---
let lastGesture = 'NONE';
let isPreviewMode = false;
let isMouseFallback = false;
let lastPinchTime = 0;
let lastPinchDist = -1;
let lastPoint = null;
let lastClosedTime = 0; // Guard against rapid Pinch after Closed
let pinchStartTime = 0; // Timer for hold-to-build
let undoStartTime = 0; // Timer for hold-to-undo
let interactionState = 'IDLE'; // IDLE, DRAW_WAIT, BLOCK_PLACED, UNDO_WAIT, UNDO_COMPLETE, ROTATING, ZOOMING
let blockPlacedThisPinch = false; // Track if block was placed during current pinch

// --- GESTURE RECOGNIZER ---
class GestureRecognizer {
    constructor(trainer = null) {
        this.trainer = trainer;
        this.history = []; // Store last N positions for swipe detection
        this.historySize = 15;
        this.swipeThreshold = 0.05; // Movement threshold

        // Use personalized pinch threshold if available
        const personalizedPinch = trainer?.getPersonalizedThreshold('pinch');
        this.pinchThreshold = personalizedPinch || 0.05; // Default or personalized

        // Gesture stabilization - prevent flickering
        this.gestureHistory = [];
        this.gestureHistorySize = 10; // Increased for better stability
        this.lastStableGesture = 'NONE';
        this.gestureConfidenceThreshold = 0.6; // 60% of samples must agree

        // Smoothing for hand position
        this.positionBuffer = [];
        this.positionBufferSize = 5; // Increased for smoother tracking

        console.log(`🎯 Pinch threshold: ${this.pinchThreshold.toFixed(3)} ${personalizedPinch ? '(personalized)' : '(default)'}`);
    }

    update(landmarks) {
        if (!landmarks || landmarks.length === 0) return { gesture: 'NONE', hand: null };

        const hands = landmarks.map(hand => this.analyzeHand(hand));

        // Two Hand Gestures
        if (hands.length === 2) {
            const g1 = hands[0].gesture;
            const g2 = hands[1].gesture;

            // Zoom In: Two Fists
            if (g1 === 'CLOSED' && g2 === 'CLOSED') {
                return { gesture: 'ZOOM_IN', hands: hands };
            }

            // Zoom Out: One Fist, One Palm
            if ((g1 === 'CLOSED' && g2 === 'OPEN_PALM') || (g1 === 'OPEN_PALM' && g2 === 'CLOSED')) {
                return { gesture: 'ZOOM_OUT', hands: hands };
            }
        }

        // Single hand priority
        const primaryHand = hands[0];

        // Swipe detection (simplified - using palm center movement)
        const palmCenter = primaryHand.center;
        this.history.push(palmCenter);
        if (this.history.length > this.historySize) this.history.shift();

        let swipe = null;
        if (this.history.length === this.historySize) {
            const start = this.history[0];
            const end = this.history[this.history.length - 1];
            const dx = end.x - start.x;

            if (Math.abs(dx) > this.swipeThreshold && Math.abs(dx) > Math.abs(end.y - start.y) * 2) {
                if (primaryHand.gesture === 'OPEN_PALM') {
                    swipe = dx > 0 ? 'SWIPE_RIGHT' : 'SWIPE_LEFT';
                }
            }
        }

        if (swipe) {
            return { gesture: swipe, hand: primaryHand };
        }

        // Gesture stabilization - reduce flickering
        const currentGesture = primaryHand.gesture;
        this.gestureHistory.push(currentGesture);
        if (this.gestureHistory.length > this.gestureHistorySize) {
            this.gestureHistory.shift();
        }

        // Find most common gesture in recent history with confidence check
        const gestureCounts = {};
        this.gestureHistory.forEach(g => {
            gestureCounts[g] = (gestureCounts[g] || 0) + 1;
        });

        // Get the most frequent gesture
        const stableGesture = Object.keys(gestureCounts).reduce((a, b) =>
            gestureCounts[a] > gestureCounts[b] ? a : b
        );

        // Only update if confidence threshold is met
        const confidence = gestureCounts[stableGesture] / this.gestureHistory.length;
        if (confidence >= this.gestureConfidenceThreshold) {
            this.lastStableGesture = stableGesture;
        }

        return { gesture: this.lastStableGesture, hand: primaryHand };
    }

    analyzeHand(landmarks) {
        const thumbTip = landmarks[4];
        const indexTip = landmarks[8];
        const middleTip = landmarks[12];
        const ringTip = landmarks[16];
        const pinkyTip = landmarks[20];
        const wrist = landmarks[0];
        const palmBase = landmarks[0]; // Wrist as palm base

        // Euclidean distance
        const dist = (p1, p2) => Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2);

        // PINCH detection - thumb and index close BUT other fingers must be extended
        const pinchDist = dist(thumbTip, indexTip);
        const thumbIndexClose = pinchDist < this.pinchThreshold;

        // For a TRUE pinch, middle/ring/pinky should be more extended (not curled)
        const middleExtended = dist(middleTip, wrist) > dist(landmarks[9], wrist) * 1.1; // Stricter
        const ringExtended = dist(ringTip, wrist) > dist(landmarks[13], wrist) * 1.1;    // Stricter
        const pinkyExtended = dist(pinkyTip, wrist) > dist(landmarks[17], wrist) * 1.1;  // Stricter

        // ALL three of the other fingers should be extended for a pinch
        const otherFingersExtended = middleExtended && ringExtended && pinkyExtended;

        const isPinching = thumbIndexClose && otherFingersExtended;

        // FIST detection - all fingers curled (including thumb)
        const curlFactor = 1.3;
        const indexCurled = dist(indexTip, palmBase) < dist(landmarks[5], palmBase) * curlFactor;
        const middleCurled = dist(middleTip, palmBase) < dist(landmarks[9], palmBase) * curlFactor;
        const ringCurled = dist(ringTip, palmBase) < dist(landmarks[13], palmBase) * curlFactor;
        const pinkyCurled = dist(pinkyTip, palmBase) < dist(landmarks[17], palmBase) * curlFactor;

        // For fist, ALL 4 fingers should be curled AND thumb should be close to palm
        const curledCount = [indexCurled, middleCurled, ringCurled, pinkyCurled].filter(Boolean).length;
        const thumbCurled = dist(thumbTip, palmBase) < dist(landmarks[2], palmBase) * 1.4;

        // VICTORY (Peace Sign) - Index & Middle Extended, Ring & Pinky Curled
        const victoryCondition =
            !indexCurled && !middleCurled && // Index and Middle UP
            ringCurled && pinkyCurled &&     // Ring and Pinky DOWN
            !thumbCurled;

        // THUMBS UP - Thumb Extended UP, All fingers Curled (except thumb)
        // 1. Thumb tip is significantly higher (lower Y) than index knuckle (5)
        // 2. Thumb is extended (distance from base large)
        // 3. Thumb tip is FAR from index tip (to distinguish from fist where thumb rests on fingers)

        const indexMCP = landmarks[5]; // Index knuckle
        const thumbIndexDist = dist(thumbTip, indexTip);

        const thumbIsHigh = (thumbTip.y < indexMCP.y - 0.05) && (thumbTip.z < indexMCP.z); // Must be clearly above and forward
        const thumbIsExtended = dist(thumbTip, palmBase) > dist(landmarks[2], palmBase) * 1.2;
        const thumbFarFromIndex = thumbIndexDist > 0.1; // Euclidean distance in normalized 3D

        const isThumbsUp = thumbIsHigh && thumbIsExtended && thumbFarFromIndex && curledCount >= 3;

        // FIST detection - Priority over Thumbs Up if ambiguous
        // If thumb is close to fingers, it's a Fist
        const isFist = curledCount === 4 && !isThumbsUp;

        // OPEN PALM - at least 3 fingers extended
        const extendFactor = 1.05;
        const indexExtended = dist(indexTip, wrist) > dist(landmarks[5], wrist) * extendFactor;
        const middleExt = dist(middleTip, wrist) > dist(landmarks[9], wrist) * extendFactor;
        const ringExt = dist(ringTip, wrist) > dist(landmarks[13], wrist) * extendFactor;
        const pinkyExt = dist(pinkyTip, wrist) > dist(landmarks[17], wrist) * extendFactor;

        const extendedCount = [indexExtended, middleExt, ringExt, pinkyExt].filter(Boolean).length;
        const isOpen = extendedCount >= 3;

        // Gesture priority: PINCH > THUMBS_UP > VICTORY > FIST > OPEN
        let gesture = 'UNKNOWN';
        if (isPinching) {
            gesture = 'PINCH';
        } else if (isThumbsUp) {
            gesture = 'THUMBS_UP';
        } else if (victoryCondition) {
            gesture = 'VICTORY';
        } else if (isFist) {
            gesture = 'CLOSED';
        } else if (isOpen) {
            gesture = 'OPEN_PALM';
        } else {
            gesture = 'OPEN_PALM';
        }

        // Debug logging every 30 frames
        if (Math.random() < 0.03) {
            console.log('Gesture Debug:', {
                gesture,
                curledCount,
                thumbIsHigh,
                thumbFarFromIndex
            });
        }

        // Smooth the pinch center position to reduce jitter
        const rawPinchCenter = {
            x: (thumbTip.x + indexTip.x) / 2,
            y: (thumbTip.y + indexTip.y) / 2,
            z: (thumbTip.z + indexTip.z) / 2
        };

        // Dynamic Buffer Size: Smaller for PINCH (drag) to reduce lag, Larger for others (hover) to reduce jitter
        const activeBufferSize = gesture === 'PINCH' ? 2 : this.positionBufferSize;

        this.positionBuffer.push(rawPinchCenter);
        while (this.positionBuffer.length > activeBufferSize) {
            this.positionBuffer.shift();
        }

        // Average the buffered positions
        const smoothedPinchCenter = {
            x: this.positionBuffer.reduce((sum, p) => sum + p.x, 0) / this.positionBuffer.length,
            y: this.positionBuffer.reduce((sum, p) => sum + p.y, 0) / this.positionBuffer.length,
            z: this.positionBuffer.reduce((sum, p) => sum + p.z, 0) / this.positionBuffer.length
        };

        return {
            gesture,
            isPinching,
            isFist,
            center: landmarks[9],
            pinchCenter: smoothedPinchCenter
        };
    }
}

// Helper to draw live preview in training modal
function drawTrainingPreview(landmarks) {
    const tCanvas = document.getElementById('training-canvas');
    if (!tCanvas) return;

    const ctx = tCanvas.getContext('2d');

    // Draw Video Feed
    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-tCanvas.width, 0);
    if (video.readyState >= 2) { // Assuming 'video' is a globally accessible video element
        ctx.drawImage(video, 0, 0, tCanvas.width, tCanvas.height);
    }
    ctx.restore();

    // Draw Skeleton
    if (landmarks) {
        ctx.strokeStyle = '#00f0ff';
        ctx.lineWidth = 2;

        // Assuming HAND_CONNECTIONS is defined globally or accessible
        // Example: [[0,1],[1,2],...]
        const HAND_CONNECTIONS = [
            [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8], // Index
            [0, 9], [9, 10], [10, 11], [11, 12], // Middle
            [0, 13], [13, 14], [14, 15], [15, 16], // Ring
            [0, 17], [17, 18], [18, 19], [19, 20]  // Pinky
        ];

        for (const [start, end] of HAND_CONNECTIONS) {
            const p1 = landmarks[start];
            const p2 = landmarks[end];
            ctx.beginPath();
            ctx.moveTo(p1.x * tCanvas.width, p1.y * tCanvas.height);
            ctx.lineTo(p2.x * tCanvas.width, p2.y * tCanvas.height);
            ctx.stroke();
        }

        ctx.fillStyle = '#ff0000';
        for (const pt of landmarks) {
            ctx.beginPath();
            ctx.arc(pt.x * tCanvas.width, pt.y * tCanvas.height, 3, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
}

// --- VOXEL WORLD MODULE ---
class VoxelWorld {
    constructor(canvas) {
        this.canvas = canvas;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });

        // Setup
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.autoClear = true;

        // "Desk level" view
        this.camera.position.set(0, 5, 8);
        this.camera.lookAt(0, 0, 0);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 2);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 2);
        dirLight.position.set(5, 10, 7);
        dirLight.castShadow = true;
        this.scene.add(dirLight);

        // Grid (The "Desk")
        const gridHelper = new THREE.GridHelper(20, 20, 0x00ffff, 0x555555);
        this.scene.add(gridHelper);

        // Plane for Raycasting (Invisible floor)
        this.plane = new THREE.Mesh(
            new THREE.PlaneGeometry(50, 50),
            new THREE.MeshBasicMaterial({ visible: false })
        );
        this.plane.rotation.x = -Math.PI / 2;
        this.scene.add(this.plane);

        // Voxel Data
        this.voxels = new Map(); // "x,y,z" -> Mesh
        this.voxelSize = 1;

        // Cursor
        const cursorGeo = new THREE.BoxGeometry(1, 1, 1);
        const cursorMat = new THREE.MeshBasicMaterial({
            color: 0x3b82f6,
            wireframe: true,
            transparent: true,
            opacity: 0.5
        });
        this.cursor = new THREE.Mesh(cursorGeo, cursorMat);
        this.scene.add(this.cursor);
        this.cursorVisible = false;
        this.cursor.visible = false;

        // Raycaster
        this.raycaster = new THREE.Raycaster();
        this.pointer = new THREE.Vector2();

        // Controls (Fallback)
        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;

        // Resize Listener
        window.addEventListener('resize', this.onWindowResize.bind(this));

        // Undo History
        this.history = [];
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    updateCursorFromHand(ndcX, ndcY) {
        this.pointer.set(ndcX, ndcY);
        this.raycaster.setFromCamera(this.pointer, this.camera);

        // Helper: Snap to grid (Integers for X/Z, Half-Integers for Y)
        const snap = (val) => Math.round(val);
        const snapY = (val) => Math.floor(val) + 0.5;

        // 1. Raycast against existing voxels (Priority: Stacking/Side-building)
        const voxelMeshes = Array.from(this.voxels.values());
        const voxelIntersects = this.raycaster.intersectObjects(voxelMeshes);

        if (voxelIntersects.length > 0) {
            const hit = voxelIntersects[0];
            const hitPos = hit.object.position;
            const normal = hit.face.normal;

            // Calculate target position based on face normal
            const targetPos = hitPos.clone().add(normal);

            const x = snap(targetPos.x);
            const z = snap(targetPos.z);
            const y = snapY(targetPos.y);

            this.cursor.position.set(x, y, z);
            this.cursorVisible = true;
            this.cursor.visible = true;
            return { x, y, z };
        }

        // 2. Raycast against the floor plane (Ground Placement)
        const intersects = this.raycaster.intersectObject(this.plane);

        if (intersects.length > 0) {
            const intersect = intersects[0];

            // Snap X and Z to integers
            const x = snap(intersect.point.x);
            const z = snap(intersect.point.z);

            // Find highest block in this column to stack on top
            // Base level is 0.5 (since size is 1 and centered)
            let targetY = 0.5;

            // Check if any voxel occupies this vertical column
            let highestInColumn = -Infinity;

            this.voxels.forEach((mesh) => {
                const pos = mesh.position;
                // Check X/Z alignment (tolerance for float errors)
                if (Math.abs(pos.x - x) < 0.1 && Math.abs(pos.z - z) < 0.1) {
                    if (pos.y > highestInColumn) {
                        highestInColumn = pos.y;
                    }
                }
            });

            if (highestInColumn > -Infinity) {
                targetY = highestInColumn + 1; // Stack on top
            }

            this.cursor.position.set(x, targetY, z);
            this.cursorVisible = true;
            this.cursor.visible = true;
            return { x, y: targetY, z };
        }

        this.cursorVisible = false;
        this.cursor.visible = false;
        return null;
    }

    createVoxelAtCursor() {
        if (!this.cursorVisible) return false;

        const pos = this.cursor.position;
        const key = `${pos.x},${pos.y},${pos.z}`;

        if (this.voxels.has(key)) return false;

        const geometry = new THREE.BoxGeometry(0.95, 0.95, 0.95);
        const material = new THREE.MeshStandardMaterial({
            color: 0x00ffff,
            emissive: 0x0044ff,
            emissiveIntensity: 0.6,
            roughness: 0.2,
            metalness: 0.8
        });
        const cube = new THREE.Mesh(geometry, material);
        cube.position.copy(pos);
        cube.castShadow = true;
        cube.receiveShadow = true;

        // Pop animation
        cube.scale.set(0, 0, 0);

        this.scene.add(cube);
        this.voxels.set(key, cube);

        // History
        this.history.push({ type: 'ADD', key: key, position: pos.clone(), color: 0x00ffff });

        // Simple animation
        let s = 0;
        const animateIn = () => {
            s += 0.1;
            cube.scale.set(s, s, s);
            if (s < 1) requestAnimationFrame(animateIn);
        };
        animateIn();
        return true;
    }

    removeVoxelAtCursor() {
        if (!this.cursorVisible) return;

        const pos = this.cursor.position;
        const key = `${pos.x},${pos.y},${pos.z}`;

        if (this.voxels.has(key)) {
            const trash = this.voxels.get(key);
            this.scene.remove(trash);
            this.voxels.delete(key);

            // History
            this.history.push({
                type: 'REMOVE',
                key: key,
                position: pos.clone(),
                color: trash.material.color.getHex()
            });
            if (trash.geometry) trash.geometry.dispose();
            if (trash.material) trash.material.dispose();
        }
    }

    undo() {
        console.log("Undoing last action. History size:", this.history.length);
        if (this.history.length === 0) return;
        const action = this.history.pop();

        if (action.type === 'ADD') {
            if (this.voxels.has(action.key)) {
                const mesh = this.voxels.get(action.key);
                this.scene.remove(mesh);
                if (mesh.geometry) mesh.geometry.dispose();
                if (mesh.material) mesh.material.dispose();
                this.voxels.delete(action.key);
            }
        } else if (action.type === 'REMOVE') {
            const geometry = new THREE.BoxGeometry(0.95, 0.95, 0.95);
            const material = new THREE.MeshStandardMaterial({
                color: action.color,
                emissive: 0x0044ff,
                emissiveIntensity: 0.6,
                roughness: 0.2,
                metalness: 0.8
            });
            const cube = new THREE.Mesh(geometry, material);
            cube.position.copy(action.position);
            cube.castShadow = true;
            cube.receiveShadow = true;
            this.scene.add(cube);
            this.voxels.set(action.key, cube);
        }
    }

    reset() {
        console.log("Resetting world...");
        Array.from(this.voxels.values()).forEach(mesh => {
            this.scene.remove(mesh);
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material) mesh.material.dispose();
        });
        this.voxels.clear();
        this.history = [];
        this.scene.rotation.set(0, 0, 0);
        console.log("World reset complete.");
    }

    exportToJSON() {
        const data = [];
        this.voxels.forEach((mesh) => {
            data.push({
                x: mesh.position.x,
                y: mesh.position.y,
                z: mesh.position.z,
                color: mesh.material.color.getHex()
            });
        });
        return JSON.stringify(data);
    }

    loadFromJSON(jsonString) {
        try {
            const data = JSON.parse(jsonString);
            this.reset(); // Use reset instead of manual clear

            data.forEach(v => {
                const geometry = new THREE.BoxGeometry(0.95, 0.95, 0.95);
                const material = new THREE.MeshStandardMaterial({
                    color: v.color || 0x00ffff,
                    emissive: 0x0044ff,
                    emissiveIntensity: 0.6
                });
                const cube = new THREE.Mesh(geometry, material);
                cube.position.set(v.x, v.y, v.z);
                cube.castShadow = true;
                this.scene.add(cube);
                this.voxels.set(`${v.x},${v.y},${v.z}`, cube);
            });
        } catch (e) {
            console.error("Failed to load JSON", e);
        }
    }

    render() {
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

// --- MAIN APPLICATION LOGIC ---

// DOM Elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas-3d');
const handCanvas = document.getElementById('hand-canvas');
const handCtx = handCanvas.getContext('2d');
const startScreen = document.getElementById('start-screen');
const btnStart = document.getElementById('btn-start');
const loader = document.getElementById('loader');
const statusHand = document.getElementById('status-hand');
const gestureName = document.getElementById('gesture-name');
const statusIcon = document.getElementById('gesture-icon');
const trackingAccuracy = document.getElementById('tracking-accuracy');

// Hand visualization toggle
let showHandTracking = true;

// Modules
const gestureTrainer = new GestureTrainer();
const gestureRecognizer = new GestureRecognizer(gestureTrainer);
const world = new VoxelWorld(canvas);

// --- FUNCTIONS ---

// --- FUNCTIONS ---

async function enableCamera() {
    console.log("=== Start Camera Clicked ===");
    btnStart.innerText = "Starting...";
    btnStart.disabled = true;

    try {
        console.log("Calling startWebcam...");
        const success = await startWebcam(video);
        console.log("startWebcam returned:", success);

        if (success) {
            console.log("Webcam started successfully!");
            console.log("Hiding start screen...");
            startScreen.classList.add('hidden');
            startScreen.style.display = 'none';
            console.log("Starting render loop...");
            loop();
            console.log("=== Camera Enabled Successfully ===");
        } else {
            console.error("startWebcam returned false");
            btnStart.disabled = false;
            btnStart.innerText = "Start Camera";
            loader.innerText = "Failed to start camera. Please check permissions.";
        }
    } catch (error) {
        console.error("Error in enableCamera:", error);
        btnStart.disabled = false;
        btnStart.innerText = "Start Camera";
        loader.innerText = `Error: ${error.message}`;
        alert("Failed to start camera: " + error.message);
    }
}

async function initializeHandLandmarker() {
    try {
        console.log("Initializing Vision Tasks...");
        loader.innerText = "Loading hand tracking model...";
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
        );
        console.log("FilesetResolver loaded");
        loader.innerText = "Creating hand landmarker...";
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                delegate: "GPU"
            },
            runningMode: runningMode,
            numHands: 2,
            minHandDetectionConfidence: 0.7,  // Increased for better accuracy
            minHandPresenceConfidence: 0.7,   // Increased for better accuracy
            minTrackingConfidence: 0.7        // Increased for better accuracy
        });
        console.log("HandLandmarker initialized successfully!");
        loader.innerText = "Ready to start";
        btnStart.disabled = false;
        btnStart.onclick = () => enableCamera();
        return true;
    } catch (e) {
        console.error("Error initializing HandLandmarker:", e);
        loader.innerText = `Error: ${e.message}. Check Console.`;
        alert("ML Error: " + e.message);
        return false;
    }
}

// Hand landmark connections for drawing skeleton
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],      // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],      // Index
    [0, 9], [9, 10], [10, 11], [11, 12], // Middle
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [5, 9], [9, 13], [13, 17]             // Palm
];

function resizeHandCanvas() {
    handCanvas.width = window.innerWidth;
    handCanvas.height = window.innerHeight;
    console.log('Hand canvas sized:', handCanvas.width, 'x', handCanvas.height);
}

const gestureToDisplayMap = {
    'OPEN_PALM': 'STOP',
    'CLOSED': 'FIST',
    'PINCH': 'PINCH',
    'VICTORY': 'VICTORY',
    'THUMBS_UP': 'THUMBS UP',
    'NONE': 'Searching...'
};

function drawHandLandmarks(landmarks, currentGesture) {
    handCtx.clearRect(0, 0, handCanvas.width, handCanvas.height);

    // Draw Gesture Text Overlay (Top Left)
    handCtx.save();
    handCtx.scale(-1, 1); // Flip text back since canvas is mirrored
    handCtx.translate(-handCanvas.width, 0);

    handCtx.font = "bold 40px Arial";
    handCtx.fillStyle = "#00FF00"; // Green text like reference
    handCtx.shadowColor = "rgba(0,0,0,0.5)";
    handCtx.shadowBlur = 4;

    const displayText = gestureToDisplayMap[currentGesture] || '';
    if (displayText) {
        handCtx.fillText(`Gesture: ${displayText}`, 20, 60);
    }
    handCtx.restore();

    if (!showHandTracking || !landmarks || landmarks.length === 0) return;

    landmarks.forEach((hand) => {
        // 1. Draw connections (skeleton) - WHITE LINES
        handCtx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        handCtx.lineWidth = 3;
        handCtx.lineCap = 'round';
        handCtx.lineJoin = 'round';
        handCtx.shadowBlur = 0; // consistent clean look

        HAND_CONNECTIONS.forEach(([start, end]) => {
            const startPoint = hand[start];
            const endPoint = hand[end];

            handCtx.beginPath();
            handCtx.moveTo(startPoint.x * handCanvas.width, startPoint.y * handCanvas.height);
            handCtx.lineTo(endPoint.x * handCanvas.width, endPoint.y * handCanvas.height);
            handCtx.stroke();
        });

        // 2. Draw landmarks (points) - RED DOTS with White Border
        hand.forEach((landmark) => {
            const x = landmark.x * handCanvas.width;
            const y = landmark.y * handCanvas.height;

            // White border
            handCtx.beginPath();
            handCtx.arc(x, y, 6, 0, 2 * Math.PI);
            handCtx.fillStyle = '#FFFFFF';
            handCtx.fill();

            // Red center
            handCtx.beginPath();
            handCtx.arc(x, y, 4, 0, 2 * Math.PI);
            handCtx.fillStyle = '#FF0000';
            handCtx.fill();
        });
    });
}

// Initialize hand canvas size
resizeHandCanvas();
window.addEventListener('resize', resizeHandCanvas);

async function startWebcam(video) {
    console.log("=== startWebcam called ===");
    console.log("handLandmarker status:", handLandmarker ? "initialized" : "NOT initialized");
    console.log("video element:", video);

    if (!handLandmarker) {
        console.error("HandLandmarker not initialized!");
        alert("Hand tracking not ready. Please refresh the page.");
        return false;
    }

    const constraints = {
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user"
        },
        audio: false
    };

    console.log("Requesting camera access with constraints:", constraints);

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log("Got media stream:", stream);
        console.log("Video tracks:", stream.getVideoTracks());

        video.srcObject = stream;

        return new Promise((resolve) => {
            video.addEventListener("loadeddata", () => {
                console.log("Video loaded! Resolution:", video.videoWidth, "x", video.videoHeight);
                webcamRunning = true;
                video.play().then(() => {
                    console.log("Video playing successfully");
                    resolve(true);
                }).catch(err => {
                    console.error("Error playing video:", err);
                    resolve(false);
                });
            }, { once: true });

            // Timeout fallback
            setTimeout(() => {
                if (!webcamRunning) {
                    console.warn("Webcam timeout - forcing start anyway");
                    video.play().catch(console.error);
                    resolve(true);
                }
            }, 2000);
        });
    } catch (e) {
        console.error("Error accessing webcam:", e);
        console.error("Error name:", e.name);
        console.error("Error message:", e.message);

        let errorMsg = "Camera Error: ";
        if (e.name === "NotAllowedError") {
            errorMsg += "Please allow camera access in your browser settings.";
        } else if (e.name === "NotFoundError") {
            errorMsg += "No camera found on this device.";
        } else {
            errorMsg += e.message;
        }

        alert(errorMsg);
        return false;
    }
}

async function getHandData(video) {
    if (!webcamRunning || video.readyState < 2) return null;
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await handLandmarker.setOptions({ runningMode: "VIDEO" });
    }

    if (video.currentTime !== lastVideoTimeInternal) {
        lastVideoTimeInternal = video.currentTime;
        results = handLandmarker.detectForVideo(video, performance.now());
    }

    return results;
}

// --- INTERACTION HANDLING ---

const handleInteraction = (gesture, handState, landmarks) => {
    // 1. Calculate Cursor Position (NDC)
    let point = { x: 0.5, y: 0.5 };
    if (handState && handState.pinchCenter) {
        point = handState.pinchCenter;
    } else if (landmarks) {
        point = landmarks[8];
    }

    const ndcX = (1 - point.x) * 2 - 1;
    const ndcY = 1 - (point.y * 2);

    // Update 3D Cursor (Palm/Hover is default behavior when cursor moves)
    world.updateCursorFromHand(ndcX, ndcY);

    const now = Date.now();

    // 2. ZOOM Logic (High Priority)
    if (gesture === 'ZOOM_IN') {
        world.camera.position.z = Math.max(2, world.camera.position.z - 0.2);
        gestureName.innerText = "Zoom In 🔍";
        interactionState = 'ZOOMING';
        return;
    }
    if (gesture === 'ZOOM_OUT') {
        world.camera.position.z = Math.min(20, world.camera.position.z + 0.2);
        gestureName.innerText = "Zoom Out 🔭";
        interactionState = 'ZOOMING';
        return;
    }

    // 3. State Machine Transitions
    if (gesture === 'CLOSED') {
        // FIST -> ROTATE
        if (interactionState !== 'ROTATING') {
            interactionState = 'ROTATING';
            lastPoint = { x: ndcX, y: ndcY };
        }
        world.cursorVisible = false;
        world.cursor.visible = false;

    } else if (gesture === 'PINCH') {
        // PINCH -> BUILD (after delay)
        if (interactionState === 'ROTATING' || interactionState === 'ZOOMING') {
            if (now - lastClosedTime < 300) return; // Debounce
            interactionState = 'IDLE';
        }

        if (interactionState !== 'DRAW_WAIT' && interactionState !== 'BLOCK_PLACED') {
            interactionState = 'DRAW_WAIT';
            pinchStartTime = now;
            blockPlacedThisPinch = false;
        }

    } else if (gesture === 'VICTORY') {
        // VICTORY (2 Fingers) -> UNDO (after delay)
        if (interactionState !== 'UNDO_WAIT' && interactionState !== 'UNDO_COMPLETE') {
            interactionState = 'UNDO_WAIT';
            undoStartTime = now;
        }

    } else {
        // IDLE / OPEN PALM (Hover)
        interactionState = 'IDLE';
        blockPlacedThisPinch = false;
    }

    // 4. State Actions execution

    // --- ROTATING ---
    if (interactionState === 'ROTATING') {
        gestureName.innerText = "Fist: Rotating 🔄";
        gestureName.style.color = "cyan";

        if (lastPoint) {
            const deltaX = ndcX - lastPoint.x;
            const deltaY = ndcY - lastPoint.y;
            const sensitivity = 3.5;

            world.scene.rotation.y += deltaX * sensitivity;
            world.scene.rotation.x += deltaY * sensitivity;

            // Limit vertical rotation to avoid flipping
            // world.scene.rotation.x = Math.max(-Math.PI/2, Math.min(Math.PI/2, world.scene.rotation.x));
        }
        lastClosedTime = now;
    }

    // --- DRAWING (BUILD) ---
    else if (interactionState === 'DRAW_WAIT') {
        const elapsed = now - pinchStartTime;
        const BUILD_DELAY = 1000; // Reduced to 1s for better responsiveness

        if (elapsed > BUILD_DELAY) {
            // Place INITIAL block
            if (world.createVoxelAtCursor()) {
                triggerHapticFeedback();
                blockPlacedThisPinch = true;
                interactionState = 'DRAWING'; // Switch to continuous drawing
                gestureName.innerText = "Drawing Mode ✏️";
                gestureName.style.color = "#00ff00";
            }
        } else {
            const timeLeft = ((BUILD_DELAY - elapsed) / 1000).toFixed(1);
            gestureName.innerText = `Hold to Build: ${timeLeft}s`;
            gestureName.style.color = "yellow";
        }
    }
    // --- CONTINUOUS DRAWING (DRAG) ---
    else if (interactionState === 'DRAWING') {
        gestureName.innerText = "Drawing Mode ✏️";
        gestureName.style.color = "#00ff00";

        // Try to place block at current new position
        // createVoxelAtCursor handles duplication checks internally
        if (world.createVoxelAtCursor()) {
            triggerHapticFeedback();
        }
    }

    // --- UNDOING ---
    else if (interactionState === 'UNDO_WAIT') {
        const elapsed = now - undoStartTime;
        const UNDO_DELAY = 1500;

        if (elapsed > UNDO_DELAY) {
            world.undo();
            triggerHapticFeedback();
            interactionState = 'UNDO_COMPLETE';
            console.log("↩️ Undo triggered!");
        } else {
            const timeLeft = ((UNDO_DELAY - elapsed) / 1000).toFixed(1);
            gestureName.innerText = `Hold to Delete: ${timeLeft}s`;
            gestureName.style.color = "#ff4444";
        }
    }
    else if (interactionState === 'UNDO_COMPLETE') {
        gestureName.innerText = "Deleted! 🗑️ Release Fingers";
        gestureName.style.color = "#ff4444";
    }

    // --- IDLE (HOVER) ---
    else if (interactionState === 'IDLE') {
        gestureName.innerText = "Palm: Hover ✋";
        if (gesture === 'THUMBS_UP') {
            gestureName.innerText = "Thumbs Up! 👍";
            gestureName.style.color = "#00ff00";
        } else {
            gestureName.style.color = "white";
        }
    }

    lastPoint = { x: ndcX, y: ndcY };
    lastGesture = gesture;
};

// UI Updates
const updateUI = (gesture) => {
    let displayText = gesture;
    if (gesture === 'CLOSED') displayText = "Fist: Rotate 🔄";
    if (gesture === 'PINCH') displayText = "Pinch: Draw ✏️";
    if (gesture === 'OPEN_PALM') displayText = "Palm: Hover ✋";
    if (gesture === 'ZOOM_IN') displayText = "Zoom In 🔍";
    if (gesture === 'ZOOM_OUT') displayText = "Zoom Out 🔭";

    gestureName.innerText = displayText;

    // Icon mapping
    const icons = {
        'PINCH': '👌',
        'OPEN_PALM': '✋',
        'CLOSED': '✊',
        'SWIPE_LEFT': '👈',
        'SWIPE_RIGHT': '👉',
        'ZOOM_IN': '🔍',
        'ZOOM_OUT': '🔭',
        'NONE': '❌'
    };
    statusIcon.innerText = icons[gesture] || '✋';
};

const triggerHapticFeedback = () => {
    const feedback = document.getElementById('gesture-feedback');
    feedback.style.borderColor = '#3b82f6';
    setTimeout(() => {
        feedback.style.borderColor = 'rgba(255, 255, 255, 0.1)';
    }, 200);
};

// --- INITIALIZATION ---

const initApp = async () => {
    await initializeHandLandmarker();
}
// Start immediate init of non-webcam stuff? 
// Actually we wait for user to click button.
initApp();

// Main Loop
async function loop() {
    try {
        const now = performance.now();

        if (!isMouseFallback) {
            const results = await getHandData(video);

            if (results && results.landmarks && results.landmarks.length > 0) {
                statusHand.classList.add('connected');
                statusHand.innerHTML = '<span class="dot"></span> Hand Tracking';

                // Process Hand 1
                const hand = results.landmarks[0]; // Raw landmarks
                const analysis = gestureRecognizer.update(results.landmarks); // Analysis result

                // Get the gesture state for the primary hand
                const gestureAnalysis = analysis.hand || { gesture: 'NONE' };
                const gesture = analysis.gesture === 'NONE' ?
                    (analysis.hands ? 'MULTIH,AND' : gestureAnalysis.gesture) :
                    analysis.gesture; // Handle 2-hand or 1-hand

                // We mostly care about single hand interaction for now
                const primaryGesture = gestureAnalysis.gesture;

                // Tracking Accuracy
                // Just a mock calculation based on visibility/jitter could go here, 
                // but let's just use confidence if available or set to 100% when tracking.
                trackingAccuracy.innerText = '100%';
                trackingAccuracy.style.color = '#00ff00';

                updateUI(primaryGesture);

                if (!isPreviewMode) {
                    // Pass the analyzed gesture
                    handleInteraction(primaryGesture, gestureAnalysis, hand);
                }

                // Draw hand landmarks visualization
                drawHandLandmarks(results.landmarks, primaryGesture);

                // Collect training samples if in training mode
                if (gestureTrainer.isTraining) {
                    // Draw preview on the small canvas
                    if (results.landmarks[0]) {
                        drawTrainingPreview(results.landmarks[0]);
                        gestureTrainer.collectSample(results.landmarks[0]);
                    } else {
                        drawTrainingPreview(null);
                    }
                }
            } else {
                statusHand.classList.remove('connected');
                updateUI('NONE');
                trackingAccuracy.innerText = '--';
                trackingAccuracy.style.color = 'white';
                drawHandLandmarks(null, 'NONE'); // Clear visualization

                // Clear training preview if lost tracking
                if (gestureTrainer.isTraining) drawTrainingPreview(null);
            }
        }
        world.render();
    } catch (e) {
        console.error("Loop Error:", e);
    }
    requestAnimationFrame(loop);
}

// Button Listeners
document.getElementById('btn-minimize').onclick = () => {
    document.getElementById('controls-panel').classList.toggle('minimized');
};

const btnCheckCam = document.getElementById('btn-toggle-camera');
let isCameraOn = true;

btnCheckCam.onclick = async () => {
    isCameraOn = !isCameraOn;

    if (isCameraOn) {
        const success = await startWebcam(video);
        if (success) {
            btnCheckCam.innerHTML = `
            <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            <span>Stop Camera</span>`;
            statusHand.innerHTML = '<span class="dot"></span> Hand Tracking';
        }
    } else {
        if (video.srcObject) {
            const tracks = video.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            video.srcObject = null;
        }
        webcamRunning = false;
        btnCheckCam.innerHTML = `
        <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
        </svg>
        <span>Start Camera</span>`;
        statusHand.innerHTML = '<span class="dot" style="background:red; box-shadow:none;"></span> Camera Off';
        statusHand.classList.remove('connected');
    }
};

document.getElementById('btn-toggle-mode').onclick = () => {
    isPreviewMode = !isPreviewMode;
    const btn = document.getElementById('btn-toggle-mode');
    btn.innerHTML = isPreviewMode ? "<span>✏️ Edit Mode</span>" : "<span>👁️ Preview Mode</span>";
    world.cursor.visible = !isPreviewMode;
};

document.getElementById('btn-save').onclick = () => {
    const json = world.exportToJSON();
    localStorage.setItem('volex_scene', json);
    alert('Scene saved to LocalStorage!');
};

document.getElementById('btn-load').onclick = () => {
    const json = localStorage.getItem('volex_scene');
    if (json) {
        world.loadFromJSON(json);
    } else {
        alert('No saved scene found.');
    }
};

// Button Listeners Setup
function setupEventListeners() {
    // 1. Minimize Panel
    const btnMinimize = document.getElementById('btn-minimize');
    if (btnMinimize) {
        btnMinimize.onclick = () => {
            document.getElementById('controls-panel').classList.toggle('minimized');
        };
    }

    // 2. Camera Toggle
    const btnCheckCam = document.getElementById('btn-toggle-camera');
    if (btnCheckCam) {
        btnCheckCam.onclick = async () => {
            isCameraOn = !isCameraOn;

            if (isCameraOn) {
                const success = await startWebcam(video);
                if (success) {
                    btnCheckCam.innerHTML = `
                    <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    <span>Stop Camera</span>`;
                    statusHand.innerHTML = '<span class="dot"></span> Hand Tracking';
                }
            } else {
                if (video.srcObject) {
                    const tracks = video.srcObject.getTracks();
                    tracks.forEach(track => track.stop());
                    video.srcObject = null;
                }
                webcamRunning = false;
                btnCheckCam.innerHTML = `
                <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                     <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                </svg>
                <span>Start Camera</span>`;
                statusHand.innerHTML = '<span class="dot" style="background:red; box-shadow:none;"></span> Camera Off';
                statusHand.classList.remove('connected');
            }
        };
    }

    // 3. Train Gestures
    const btnTrain = document.getElementById('btn-train-gestures');
    if (btnTrain) {
        btnTrain.onclick = () => {
            console.log("Train button clicked");
            if (gestureTrainer) {
                gestureTrainer.startTraining();
            } else {
                alert("Error: Gesture Trainer system not ready. Please refresh.");
                console.error("GestureTrainer not initialized");
            }
        };
    }

    const btnSkip = document.getElementById('btn-skip-training');
    if (btnSkip) {
        btnSkip.onclick = () => {
            gestureTrainer.hideTrainingModal();
        };
    }

    const btnFinish = document.getElementById('btn-finish-training');
    if (btnFinish) {
        btnFinish.onclick = () => {
            gestureTrainer.finishTraining();
        };
    }
}

// Call setup when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupEventListeners);
} else {
    setupEventListeners();
}

// Undo/Reset
document.getElementById('btn-reset').onclick = () => {
    if (confirm("Clear all blocks?")) {
        world.reset();
    }
};

document.getElementById('btn-undo').onclick = () => {
    world.undo();
};

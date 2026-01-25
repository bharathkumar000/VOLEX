// Gesture Training System
class GestureTrainer {
    constructor() {
        this.gestures = [
            {
                name: 'Open Palm',
                emoji: '✋',
                description: 'Show your open hand with all fingers extended',
                type: 'OPEN_PALM'
            },
            {
                name: 'Closed Fist',
                emoji: '✊',
                description: 'Make a fist with all fingers curled',
                type: 'CLOSED'
            },
            {
                name: 'Pinch',
                emoji: '🤏',
                description: 'Pinch thumb and index finger together, keep other fingers extended',
                type: 'PINCH'
            },
            {
                name: 'Peace Sign',
                emoji: '✌️',
                description: 'Show index and middle fingers extended, others curled',
                type: 'PEACE'
            }
        ];

        this.currentGestureIndex = 0;
        this.samples = [];
        this.samplesPerGesture = 10;
        this.isTraining = false;
        this.isScanning = false; // New state
        this.trainingData = this.loadTrainingData();

        // Attach listener
        const btnScan = document.getElementById('btn-start-scan');
        if (btnScan) {
            btnScan.onclick = () => {
                this.isScanning = true;
                btnScan.style.display = 'none'; // Hide button while scanning
                // Feedback
                document.getElementById('training-description').innerText = "Hold steady... Scanning...";
                document.getElementById('training-description').style.color = "#00f0ff";
            };
        }
    }

    startTraining() {
        this.isTraining = true;
        this.isScanning = false; // Waiting for user to click start
        this.currentGestureIndex = 0;
        this.samples = [];
        this.showTrainingModal();
        this.updateTrainingUI();
    }

    showTrainingModal() {
        const modal = document.getElementById('training-modal');
        modal.style.display = 'flex';
    }

    hideTrainingModal() {
        const modal = document.getElementById('training-modal');
        modal.style.display = 'none';
        this.isTraining = false;
        this.isScanning = false;
    }

    updateTrainingUI() {
        const currentGesture = this.gestures[this.currentGestureIndex];

        document.getElementById('training-emoji').innerText = currentGesture.emoji;
        document.getElementById('training-gesture-name').innerText = currentGesture.name;
        document.getElementById('training-description').innerText = currentGesture.description;
        document.getElementById('training-description').style.color = "#888"; // Reset color
        document.getElementById('training-step').innerText = this.currentGestureIndex + 1;
        document.getElementById('training-samples').innerText = this.samples.length;

        const progress = (this.samples.length / this.samplesPerGesture) * 100;
        document.getElementById('training-progress-bar').style.width = `${progress}%`;

        // Show Start Button if waiting
        const btnScan = document.getElementById('btn-start-scan');
        if (btnScan) {
            if (!this.isScanning && this.samples.length === 0) {
                btnScan.style.display = 'block';
                btnScan.innerText = `📷 Start Scanning: ${currentGesture.name}`;
            } else {
                btnScan.style.display = 'none';
            }
        }

        // Show finish button on last gesture
        if (this.currentGestureIndex === this.gestures.length - 1 && this.samples.length === this.samplesPerGesture) {
            document.getElementById('btn-finish-training').style.display = 'block';
            document.getElementById('btn-skip-training').style.display = 'none';
        }
    }

    collectSample(landmarks) {
        if (!this.isTraining) return;

        // ** NEW: Only collect if scanning started **
        if (!this.isScanning) return;

        const currentGesture = this.gestures[this.currentGestureIndex];

        // Collect sample
        if (this.samples.length < this.samplesPerGesture) {
            this.samples.push({
                landmarks: JSON.parse(JSON.stringify(landmarks)), // Deep copy
                timestamp: Date.now()
            });

            // Update progress bar but don't reset whole UI (flicker)
            const progress = (this.samples.length / this.samplesPerGesture) * 100;
            document.getElementById('training-progress-bar').style.width = `${progress}%`;
            document.getElementById('training-samples').innerText = this.samples.length;

            // Auto-advance to next gesture when samples collected
            if (this.samples.length === this.samplesPerGesture) {
                this.isScanning = false; // Stop scanning
                setTimeout(() => {
                    if (this.currentGestureIndex < this.gestures.length - 1) {
                        this.nextGesture();
                    } else {
                        // Last gesture completed - show finish button
                        this.updateTrainingUI();
                        document.getElementById('training-description').innerText = "Training complete! Click Finish to save your personalized gestures.";
                        document.getElementById('training-description').style.color = "#00ff00";
                    }
                }, 500);
            }
        }
    }

    nextGesture() {
        // Save current gesture samples
        const currentGesture = this.gestures[this.currentGestureIndex];
        if (!this.trainingData[currentGesture.type]) {
            this.trainingData[currentGesture.type] = [];
        }
        this.trainingData[currentGesture.type] = this.samples;

        this.currentGestureIndex++;
        this.samples = [];
        this.isScanning = false; // Require click for next gesture

        if (this.currentGestureIndex < this.gestures.length) {
            this.updateTrainingUI();
        }
    }

    finishTraining() {
        // Save final gesture data
        const currentGesture = this.gestures[this.currentGestureIndex];
        this.trainingData[currentGesture.type] = this.samples;

        // Calculate optimal thresholds from training data
        this.calculateThresholds();

        // Save to localStorage
        this.saveTrainingData();

        alert('✅ Training Complete! Your personalized gestures have been saved.');
        this.hideTrainingModal();
    }

    calculateThresholds() {
        // Analyze training data and update gesture recognizer thresholds
        console.log('🧠 Calculating personalized thresholds from training data...');

        // Calculate pinch distance threshold
        if (this.trainingData.PINCH && this.trainingData.PINCH.length > 0) {
            const pinchDistances = [];
            this.trainingData.PINCH.forEach(sample => {
                const thumbTip = sample.landmarks[4];
                const indexTip = sample.landmarks[8];
                const dist = Math.sqrt(
                    (thumbTip.x - indexTip.x) ** 2 +
                    (thumbTip.y - indexTip.y) ** 2 +
                    (thumbTip.z - indexTip.z) ** 2
                );
                pinchDistances.push(dist);
            });

            const avgPinchDist = pinchDistances.reduce((a, b) => a + b, 0) / pinchDistances.length;
            const maxPinchDist = Math.max(...pinchDistances);

            // Update threshold with some tolerance
            this.trainingData.thresholds = this.trainingData.thresholds || {};
            this.trainingData.thresholds.pinch = maxPinchDist * 1.2;

            console.log(`📏 Pinch threshold set to: ${this.trainingData.thresholds.pinch.toFixed(3)}`);
        }

        // Could add more threshold calculations for fist curl factor, etc.
    }

    getPersonalizedThreshold(type) {
        if (this.trainingData.thresholds && this.trainingData.thresholds[type]) {
            return this.trainingData.thresholds[type];
        }
        return null;
    }

    saveTrainingData() {
        localStorage.setItem('volex_gesture_training', JSON.stringify(this.trainingData));
        console.log('💾 Training data saved!');
    }

    loadTrainingData() {
        const data = localStorage.getItem('volex_gesture_training');
        if (data) {
            console.log('📂 Loaded existing training data');
            return JSON.parse(data);
        }
        return {};
    }
}

export default GestureTrainer;

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { useRun } from "@/lib/run-context";
import { Button } from "../ui/button";

const TransformerEducational = () => {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const sceneRef = useRef<THREE.Scene>(null);
	const cameraRef = useRef<THREE.Camera>(null);
	const rendererRef = useRef<THREE.WebGLRenderer>(null);
	const frameRef = useRef<number | null>(null);
	const objectsRef = useRef<Array<THREE.Object3D>>([]);
	const timeRef = useRef(0);

	// Live training telemetry (from `caramba serve` control-plane via SSE).
	const {
		metrics,
		lastEvent,
		status: runStatus,
		selection,
		modelSummary,
		attentionLayers,
	} = useRun();
	const telemetryRef = useRef({ loss: 0, tok_s: 0, step: 0 });
	const pulseRef = useRef(0);
	const lastStepRef = useRef<number | null>(null);

	useEffect(() => {
		if (runStatus !== "running") return;

		const loss = metrics.loss;
		const tokS = metrics.tok_s;
		if (typeof loss === "number") telemetryRef.current.loss = loss;
		if (typeof tokS === "number") telemetryRef.current.tok_s = tokS;

		const step = lastEvent.step;
		if (typeof step === "number") {
			telemetryRef.current.step = step;
			if (lastStepRef.current !== step) {
				lastStepRef.current = step;
				pulseRef.current = 1.0;
			}
		}
	}, [metrics, lastEvent.step, runStatus]);

	const ablationFlags = useMemo(() => {
		const flags = {
			mode: "—",
			null_attn: false,
			tie_qk: false,
			rope_semantic: false,
			decoupled_gate: false,
		};
		if (!attentionLayers || attentionLayers.length === 0) return flags;
		const first = attentionLayers[0];
		if (first) flags.mode = first.mode || "—";
		for (const a of attentionLayers) {
			flags.null_attn = flags.null_attn || Boolean(a.null_attn);
			flags.tie_qk = flags.tie_qk || Boolean(a.tie_qk);
			flags.rope_semantic = flags.rope_semantic || Boolean(a.rope_semantic);
			flags.decoupled_gate = flags.decoupled_gate || Boolean(a.decoupled_gate);
		}
		return flags;
	}, [attentionLayers]);

	// Educational state
	const [currentStage, setCurrentStage] = useState(0);
	const [isPlaying, setIsPlaying] = useState(false);
	const [playbackSpeed, setPlaybackSpeed] = useState(1);
	const [showExplanation, setShowExplanation] = useState(true);

	// Camera control
	const cameraStateRef = useRef({
		radius: 20,
		theta: 0.3,
		phi: Math.PI / 3,
		target: new THREE.Vector3(0, 0, 0),
		isDragging: false,
		lastMouse: { x: 0, y: 0 },
	});

	// The input sequence we're processing
	const inputSequence = useMemo(
		() => ["The", "cat", "sat", "on", "the", "mat"],
		[],
	);

	// Stages of the forward pass
	const stages = useMemo(
		() => [
			{
				id: "input",
				name: "Input Tokens",
				description:
					"We start with raw text tokens. Each word is converted to a token ID - just a number that identifies the word in our vocabulary.",
				detail:
					'The tokenizer maps "The" → 464, "cat" → 9246, etc. These are arbitrary IDs, they don\'t encode any meaning yet.',
				focus: "tokens",
				camera: { radius: 18, theta: 0, phi: Math.PI / 3 },
			},
			{
				id: "embedding",
				name: "Token Embeddings",
				description:
					"Each token ID is converted into a dense vector (768 numbers). This embedding captures semantic meaning - similar words have similar vectors.",
				detail:
					'"cat" and "dog" would have similar embeddings because they appear in similar contexts during training. The embedding is learned, not hand-coded.',
				focus: "embeddings",
				camera: { radius: 20, theta: 0.2, phi: Math.PI / 3.5 },
			},
			{
				id: "position",
				name: "Position Encoding",
				description:
					'We add position information to each embedding. Without this, the model wouldn\'t know that "cat" comes before "sat".',
				detail:
					"Position encodings can be learned or use fixed patterns (sine waves at different frequencies). They're added directly to the token embeddings.",
				focus: "positions",
				camera: { radius: 20, theta: 0, phi: Math.PI / 3 },
			},
			{
				id: "attention_qkv",
				name: "Query, Key, Value Projections",
				description:
					"Each token creates three vectors: Query (what am I looking for?), Key (what do I contain?), and Value (what information do I provide?).",
				detail:
					"These are just linear projections (matrix multiplications) of the embeddings. The model learns what questions to ask and what answers to provide.",
				focus: "qkv",
				camera: { radius: 22, theta: 0.4, phi: Math.PI / 4 },
			},
			{
				id: "attention_scores",
				name: "Attention Scores",
				description:
					'Each token\'s Query is compared with every token\'s Key. High similarity = high attention. "sat" might attend strongly to "cat" because it needs to know what sat.',
				detail:
					"Attention(Q,K) = softmax(QK^T / √d). The softmax ensures weights sum to 1, so it's like a weighted average of where to look.",
				focus: "attention",
				camera: { radius: 18, theta: 0, phi: Math.PI / 2.5 },
			},
			{
				id: "attention_output",
				name: "Attention Output",
				description:
					"Each token collects information from other tokens, weighted by attention scores. The Value vectors are combined according to these weights.",
				detail:
					'If "sat" attends 60% to "cat" and 40% to "The", its output is 0.6×Value(cat) + 0.4×Value(The). Information flows between tokens.',
				focus: "attention_flow",
				camera: { radius: 20, theta: -0.3, phi: Math.PI / 3 },
			},
			{
				id: "residual_1",
				name: "Residual Connection",
				description:
					'The attention output is ADDED to the original embedding, not replacing it. This "residual stream" lets information flow directly through the network.',
				detail:
					'Residual connections are crucial - they let gradients flow during training and allow layers to learn "refinements" rather than complete transformations.',
				focus: "residual",
				camera: { radius: 20, theta: 0, phi: Math.PI / 3 },
			},
			{
				id: "ffn",
				name: "Feed-Forward Network",
				description:
					'Each token passes through a small neural network independently. This is where the model "thinks" about each position.',
				detail:
					'FFN expands to 4× the dimension (768→3072), applies ReLU/GELU activation, then compresses back. This adds non-linearity and stores "knowledge".',
				focus: "ffn",
				camera: { radius: 22, theta: 0.5, phi: Math.PI / 4 },
			},
			{
				id: "residual_2",
				name: "Another Residual",
				description:
					"Again, we ADD the FFN output to the residual stream. The representation is getting richer with each layer.",
				detail:
					"Think of it as: ResidualStream = Original + Attention(Original) + FFN(Original + Attention(Original)). Information accumulates.",
				focus: "residual_2",
				camera: { radius: 20, theta: 0, phi: Math.PI / 3 },
			},
			{
				id: "layers",
				name: "Repeat for N Layers",
				description:
					"This attention → FFN → residual pattern repeats for each layer (typically 6-96 layers). Each layer refines the representation further.",
				detail:
					"Early layers tend to learn syntax and local patterns. Later layers learn more abstract, semantic relationships. It's hierarchical feature learning.",
				focus: "all_layers",
				camera: { radius: 30, theta: 0, phi: Math.PI / 4 },
			},
			{
				id: "output",
				name: "Final Output",
				description:
					"The final residual stream is projected to vocabulary size. For the last token, this predicts the next word in the sequence.",
				detail:
					'Softmax over vocabulary gives probabilities: P("down"|"The cat sat on the mat") might be high. This is how GPT generates text.',
				focus: "output",
				camera: { radius: 20, theta: 0, phi: Math.PI / 3 },
			},
		],
		[],
	);

	const currentStageData = stages[currentStage];

	// Colors
	const colors = {
		token: "#60a5fa",
		embedding: "#818cf8",
		position: "#c084fc",
		query: "#f472b6",
		key: "#4ade80",
		value: "#fbbf24",
		attention: "#f97316",
		ffn: "#fb923c",
		residual: "#22d3ee",
		output: "#a3e635",
		linear: "#4ade80",
		background: "#0f172a",
		text: "#e2e8f0",
		dim: "#475569",
	};

	// Update camera position
	const updateCamera = useCallback(() => {
		if (!cameraRef.current) return;
		const state = cameraStateRef.current;
		const camera = cameraRef.current;

		state.phi = Math.max(0.1, Math.min(Math.PI - 0.1, state.phi));
		state.radius = Math.max(8, Math.min(50, state.radius));

		camera.position.x =
			state.target.x +
			state.radius * Math.sin(state.phi) * Math.sin(state.theta);
		camera.position.y = state.target.y + state.radius * Math.cos(state.phi);
		camera.position.z =
			state.target.z +
			state.radius * Math.sin(state.phi) * Math.cos(state.theta);
		camera.lookAt(state.target);
	}, []);

	// Smoothly animate camera to stage position
	const animateCameraToStage = useCallback(
		(stageCamera: { radius: number; theta: number; phi: number }) => {
			const state = cameraStateRef.current;
			const animate = () => {
				let needsUpdate = false;

				const dr = (stageCamera.radius - state.radius) * 0.05;
				const dt = (stageCamera.theta - state.theta) * 0.05;
				const dp = (stageCamera.phi - state.phi) * 0.05;

				if (Math.abs(dr) > 0.01) {
					state.radius += dr;
					needsUpdate = true;
				}
				if (Math.abs(dt) > 0.01) {
					state.theta += dt;
					needsUpdate = true;
				}
				if (Math.abs(dp) > 0.01) {
					state.phi += dp;
					needsUpdate = true;
				}

				if (needsUpdate) {
					updateCamera();
					requestAnimationFrame(animate);
				}
			};
			animate();
		},
		[updateCamera],
	);

	// Initialize Three.js
	useEffect(() => {
		if (!canvasRef.current) return;

		const scene = new THREE.Scene();
		scene.background = new THREE.Color(colors.background);
		scene.fog = new THREE.Fog(colors.background, 40, 80);
		sceneRef.current = scene;

		const camera = new THREE.PerspectiveCamera(
			50,
			canvasRef.current.clientWidth / canvasRef.current.clientHeight,
			0.1,
			1000,
		);
		cameraRef.current = camera;
		updateCamera();

		const renderer = new THREE.WebGLRenderer({
			canvas: canvasRef.current,
			antialias: true,
		});
		renderer.setSize(
			canvasRef.current.clientWidth,
			canvasRef.current.clientHeight,
		);
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		rendererRef.current = renderer;

		// Lighting
		scene.add(new THREE.AmbientLight(0xffffff, 0.6));
		const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
		dirLight.position.set(10, 20, 15);
		scene.add(dirLight);

		// Subtle grid
		const grid = new THREE.GridHelper(40, 40, 0x1e293b, 0x0f172a);
		grid.position.y = -4;
		scene.add(grid);

		// Resize handler
		const handleResize = () => {
			if (!canvasRef.current) return;
			camera.aspect =
				canvasRef.current.clientWidth / canvasRef.current.clientHeight;
			camera.updateProjectionMatrix();
			renderer.setSize(
				canvasRef.current.clientWidth,
				canvasRef.current.clientHeight,
			);
		};
		window.addEventListener("resize", handleResize);

		// Mouse controls
		const canvas = canvasRef.current;

		const handleMouseDown = (e: MouseEvent) => {
			cameraStateRef.current.isDragging = true;
			cameraStateRef.current.lastMouse = { x: e.clientX, y: e.clientY };
		};

		const handleMouseUp = () => {
			cameraStateRef.current.isDragging = false;
		};

		const handleMouseMove = (e: MouseEvent) => {
			if (!cameraStateRef.current.isDragging) return;
			const dx = e.clientX - cameraStateRef.current.lastMouse.x;
			const dy = e.clientY - cameraStateRef.current.lastMouse.y;
			cameraStateRef.current.theta -= dx * 0.008;
			cameraStateRef.current.phi += dy * 0.008;
			cameraStateRef.current.lastMouse = { x: e.clientX, y: e.clientY };
			updateCamera();
		};

		const handleWheel = (e: WheelEvent) => {
			e.preventDefault();
			cameraStateRef.current.radius += e.deltaY * 0.02;
			updateCamera();
		};

		canvas.addEventListener("mousedown", handleMouseDown);
		canvas.addEventListener("mouseup", handleMouseUp);
		canvas.addEventListener("mouseleave", handleMouseUp);
		canvas.addEventListener("mousemove", handleMouseMove);
		canvas.addEventListener("wheel", handleWheel, { passive: false });

		return () => {
			window.removeEventListener("resize", handleResize);
			canvas.removeEventListener("mousedown", handleMouseDown);
			canvas.removeEventListener("mouseup", handleMouseUp);
			canvas.removeEventListener("mouseleave", handleMouseUp);
			canvas.removeEventListener("mousemove", handleMouseMove);
			canvas.removeEventListener("wheel", handleWheel);
			if (frameRef.current !== null) {
				cancelAnimationFrame(frameRef.current);
			}
			renderer.dispose();
		};
	}, [updateCamera]);

	// Clear scene helper
	const clearScene = useCallback(() => {
		const scene = sceneRef.current;
		if (!scene) return;
		objectsRef.current.forEach((obj) => {
			scene.remove(obj);
			if (obj instanceof THREE.Mesh) {
				if (obj.geometry) obj.geometry.dispose();
				if (obj.material) {
					if (Array.isArray(obj.material))
						obj.material.forEach((m) => {
							m.dispose();
						});
					else obj.material.dispose();
				}
			}
		});
		objectsRef.current = [];
	}, []);

	// Create a text sprite (for labels)
	const createLabel = useCallback(
		(
			text: string,
			position: THREE.Vector3,
			color: string = colors.text,
			scale: number = 1,
		) => {
			const canvas = document.createElement("canvas");
			const ctx = canvas.getContext("2d");
			if (!ctx) {
				throw new Error("Failed to get 2d context from canvas");
			}
			canvas.width = 256;
			canvas.height = 64;
			ctx.fillStyle = color;
			ctx.font = "bold 32px Inter, system-ui, sans-serif";
			ctx.textAlign = "center";
			ctx.fillText(text, 128, 42);

			const texture = new THREE.CanvasTexture(canvas);
			const material = new THREE.SpriteMaterial({
				map: texture,
				transparent: true,
			});
			const sprite = new THREE.Sprite(material);
			sprite.position.copy(position);
			sprite.scale.set(2 * scale, 0.5 * scale, 1);
			return sprite;
		},
		[],
	);

	// Build scene for current stage
	const buildScene = useCallback(() => {
		const scene = sceneRef.current;
		if (!scene) return;
		clearScene();

		const stage = currentStageData;
		const numTokens = inputSequence.length;
		const tokenSpacing = 2.5;
		const startX = (-(numTokens - 1) * tokenSpacing) / 2;

		// Always show tokens as base layer
		inputSequence.forEach((token, i) => {
			const x = startX + i * tokenSpacing;

			// Token sphere
			const isActive =
				stage.focus === "tokens" ||
				stage.focus === "embeddings" ||
				stage.focus === "positions" ||
				stage.focus === "qkv" ||
				stage.focus === "attention" ||
				stage.focus === "attention_flow" ||
				stage.focus === "residual" ||
				stage.focus === "ffn" ||
				stage.focus === "residual_2" ||
				stage.focus === "all_layers" ||
				stage.focus === "output";

			const geom = new THREE.SphereGeometry(0.5, 32, 32);
			const mat = new THREE.MeshPhongMaterial({
				color: new THREE.Color(colors.token),
				emissive: new THREE.Color(colors.token),
				emissiveIntensity: isActive ? 0.3 : 0.1,
				transparent: true,
				opacity: isActive ? 1 : 0.4,
			});
			const mesh = new THREE.Mesh(geom, mat);
			mesh.position.set(x, 0, 0);
			mesh.userData = { tokenIndex: i, type: "token" };
			scene.add(mesh);
			objectsRef.current.push(mesh);

			// Token label
			const label = createLabel(
				token,
				new THREE.Vector3(x, -1.2, 0),
				colors.text,
				0.8,
			);
			scene.add(label);
			objectsRef.current.push(label);
		});

		// Stage-specific visualizations
		if (
			stage.focus === "embeddings" ||
			stage.focus === "positions" ||
			stage.focus === "qkv" ||
			stage.focus === "attention" ||
			stage.focus === "attention_flow" ||
			stage.focus === "residual" ||
			stage.focus === "ffn" ||
			stage.focus === "residual_2" ||
			stage.focus === "all_layers" ||
			stage.focus === "output"
		) {
			// Show embedding vectors as vertical bars
			inputSequence.forEach((_, i) => {
				const x = startX + i * tokenSpacing;
				const numDims = 12; // Simplified representation of 768 dims

				for (let d = 0; d < numDims; d++) {
					const value = Math.sin(i * 0.7 + d * 0.5) * 0.5 + 0.5;
					const barHeight = 0.1 + value * 0.4;

					const geom = new THREE.BoxGeometry(0.12, barHeight, 0.12);
					const mat = new THREE.MeshPhongMaterial({
						color: new THREE.Color(colors.embedding),
						transparent: true,
						opacity: stage.focus === "embeddings" ? 0.9 : 0.5,
					});
					const bar = new THREE.Mesh(geom, mat);
					bar.position.set(
						x + (d - numDims / 2) * 0.15,
						1.2 + barHeight / 2,
						0,
					);
					scene.add(bar);
					objectsRef.current.push(bar);
				}
			});
		}

		if (stage.focus === "positions") {
			// Show position encoding wave pattern
			inputSequence.forEach((_, i) => {
				const x = startX + i * tokenSpacing;

				// Position indicator ring
				const ringGeom = new THREE.TorusGeometry(0.7, 0.05, 8, 32);
				const ringMat = new THREE.MeshBasicMaterial({
					color: new THREE.Color(colors.position),
					transparent: true,
					opacity: 0.8,
				});
				const ring = new THREE.Mesh(ringGeom, ringMat);
				ring.rotation.x = Math.PI / 2;
				ring.position.set(x, 0, 0);
				ring.userData = { type: "position", index: i };
				scene.add(ring);
				objectsRef.current.push(ring);

				// Position number
				const posLabel = createLabel(
					`pos ${i}`,
					new THREE.Vector3(x, 0.9, 0.8),
					colors.position,
					0.6,
				);
				scene.add(posLabel);
				objectsRef.current.push(posLabel);
			});
		}

		if (stage.focus === "qkv") {
			// Show Q, K, V projections for each token
			inputSequence.forEach((_, i) => {
				const x = startX + i * tokenSpacing;

				["Q", "K", "V"].forEach((name, idx) => {
					const color = [colors.query, colors.key, colors.value][idx];
					const angle = (idx - 1) * 0.8;
					const projX = x + Math.sin(angle) * 1.2;
					const projZ = Math.cos(angle) * 1.2;

					const geom = new THREE.SphereGeometry(0.25, 16, 16);
					const mat = new THREE.MeshPhongMaterial({
						color: new THREE.Color(color),
						emissive: new THREE.Color(color),
						emissiveIntensity: 0.4,
					});
					const sphere = new THREE.Mesh(geom, mat);
					sphere.position.set(projX, 2.5, projZ);
					sphere.userData = { type: name, tokenIndex: i };
					scene.add(sphere);
					objectsRef.current.push(sphere);

					// Connection line from embedding to Q/K/V
					const lineGeom = new THREE.BufferGeometry().setFromPoints([
						new THREE.Vector3(x, 1.8, 0),
						new THREE.Vector3(projX, 2.5, projZ),
					]);
					const lineMat = new THREE.LineBasicMaterial({
						color: new THREE.Color(color),
						transparent: true,
						opacity: 0.5,
					});
					const line = new THREE.Line(lineGeom, lineMat);
					scene.add(line);
					objectsRef.current.push(line);
				});
			});

			// Labels
			const qLabel = createLabel(
				'Query: "What am I looking for?"',
				new THREE.Vector3(-5, 4, 0),
				colors.query,
				0.7,
			);
			const kLabel = createLabel(
				'Key: "What do I contain?"',
				new THREE.Vector3(0, 4.5, 0),
				colors.key,
				0.7,
			);
			const vLabel = createLabel(
				'Value: "What info do I provide?"',
				new THREE.Vector3(5, 4, 0),
				colors.value,
				0.7,
			);
			scene.add(qLabel, kLabel, vLabel);
			objectsRef.current.push(qLabel, kLabel, vLabel);
		}

		if (stage.focus === "attention" || stage.focus === "attention_flow") {
			// Show attention connections between tokens
			const attentionWeights = [
				[0.7, 0.1, 0.1, 0.05, 0.03, 0.02], // The
				[0.2, 0.6, 0.1, 0.05, 0.03, 0.02], // cat
				[0.1, 0.5, 0.3, 0.05, 0.03, 0.02], // sat - attends to cat!
				[0.05, 0.1, 0.2, 0.5, 0.1, 0.05], // on
				[0.3, 0.1, 0.1, 0.1, 0.3, 0.1], // the
				[0.05, 0.1, 0.1, 0.2, 0.15, 0.4], // mat
			];

			// Draw attention arcs
			for (let i = 0; i < numTokens; i++) {
				for (let j = 0; j < numTokens; j++) {
					const weight = attentionWeights[i][j];
					if (weight < 0.1) continue;

					const fromX = startX + i * tokenSpacing;
					const toX = startX + j * tokenSpacing;
					const height = 2 + Math.abs(i - j) * 0.5 + weight * 1.5;

					const curve = new THREE.QuadraticBezierCurve3(
						new THREE.Vector3(fromX, 0.6, 0),
						new THREE.Vector3((fromX + toX) / 2, height, 0),
						new THREE.Vector3(toX, 0.6, 0),
					);

					const tubeGeom = new THREE.TubeGeometry(
						curve,
						20,
						0.02 + weight * 0.06,
						8,
						false,
					);
					const tubeMat = new THREE.MeshBasicMaterial({
						color: new THREE.Color(colors.attention),
						transparent: true,
						opacity: weight * 0.8,
					});
					const tube = new THREE.Mesh(tubeGeom, tubeMat);
					tube.userData = { type: "attention", from: i, to: j, weight };
					scene.add(tube);
					objectsRef.current.push(tube);

					// Animated particle on strong connections
					if (weight > 0.3 && stage.focus === "attention_flow") {
						const particleGeom = new THREE.SphereGeometry(0.1);
						const particleMat = new THREE.MeshBasicMaterial({
							color: new THREE.Color(colors.value),
						});
						const particle = new THREE.Mesh(particleGeom, particleMat);
						const baseSpeed = 0.3 + weight * 0.3;
						particle.userData = {
							type: "flowParticle",
							curve,
							speed: baseSpeed,
							baseSpeed,
							offset: Math.random(),
						};
						scene.add(particle);
						objectsRef.current.push(particle);
					}
				}
			}

			// Highlight specific attention example
			if (stage.focus === "attention") {
				const noteLabel = createLabel(
					'"sat" attends to "cat" (50%)',
					new THREE.Vector3(0, 5, 0),
					colors.attention,
					0.8,
				);
				scene.add(noteLabel);
				objectsRef.current.push(noteLabel);
			}
		}

		if (stage.focus === "residual" || stage.focus === "residual_2") {
			// Show residual stream as accumulating layers
			const numLayers = stage.focus === "residual_2" ? 3 : 2;
			const layerNames =
				stage.focus === "residual_2"
					? ["Original", "+ Attention", "+ FFN"]
					: ["Original", "+ Attention"];
			const layerColors = [colors.embedding, colors.attention, colors.ffn];

			inputSequence.forEach((_, i) => {
				const x = startX + i * tokenSpacing;

				for (let l = 0; l < numLayers; l++) {
					const ringGeom = new THREE.TorusGeometry(0.6 + l * 0.25, 0.08, 8, 32);
					const ringMat = new THREE.MeshBasicMaterial({
						color: new THREE.Color(layerColors[l]),
						transparent: true,
						opacity: 0.7,
					});
					const ring = new THREE.Mesh(ringGeom, ringMat);
					ring.rotation.x = Math.PI / 2;
					ring.position.set(x, 0.5 + l * 0.3, 0);
					ring.userData = { type: "residual", layer: l };
					scene.add(ring);
					objectsRef.current.push(ring);
				}
			});

			// Layer labels
			layerNames.forEach((name, l) => {
				const label = createLabel(
					name,
					new THREE.Vector3(startX - 2.5, 0.5 + l * 0.3, 0),
					layerColors[l],
					0.6,
				);
				scene.add(label);
				objectsRef.current.push(label);
			});

			const streamLabel = createLabel(
				"Residual Stream: Information accumulates",
				new THREE.Vector3(0, 3, 0),
				colors.residual,
				0.8,
			);
			scene.add(streamLabel);
			objectsRef.current.push(streamLabel);
		}

		if (stage.focus === "ffn") {
			// Show FFN expansion/compression for one token
			const focusToken = 2; // "sat"
			const x = startX + focusToken * tokenSpacing;

			// Input layer
			for (let i = 0; i < 8; i++) {
				const geom = new THREE.SphereGeometry(0.15);
				const mat = new THREE.MeshPhongMaterial({ color: colors.embedding });
				const node = new THREE.Mesh(geom, mat);
				node.position.set(x - 3, (i - 3.5) * 0.4, 2);
				scene.add(node);
				objectsRef.current.push(node);
			}

			// Hidden layer (expanded)
			for (let i = 0; i < 16; i++) {
				const active = Math.random() > 0.6;
				const geom = new THREE.SphereGeometry(0.12);
				const mat = new THREE.MeshPhongMaterial({
					color: active ? colors.ffn : colors.dim,
					emissive: active ? new THREE.Color(colors.ffn) : undefined,
					emissiveIntensity: active ? 0.3 : 0,
				});
				const node = new THREE.Mesh(geom, mat);
				node.position.set(x, (i - 7.5) * 0.35, 2);
				node.userData = { type: "ffn_hidden", active };
				scene.add(node);
				objectsRef.current.push(node);
			}

			// Output layer
			for (let i = 0; i < 8; i++) {
				const geom = new THREE.SphereGeometry(0.15);
				const mat = new THREE.MeshPhongMaterial({ color: colors.linear });
				const node = new THREE.Mesh(geom, mat);
				node.position.set(x + 3, (i - 3.5) * 0.4, 2);
				scene.add(node);
				objectsRef.current.push(node);
			}

			// Labels
			const inLabel = createLabel(
				"768 dims",
				new THREE.Vector3(x - 3, -2.5, 2),
				colors.embedding,
				0.6,
			);
			const hidLabel = createLabel(
				"3072 dims (4x expansion)",
				new THREE.Vector3(x, -3.5, 2),
				colors.ffn,
				0.6,
			);
			const outLabel = createLabel(
				"768 dims",
				new THREE.Vector3(x + 3, -2.5, 2),
				colors.linear,
				0.6,
			);
			scene.add(inLabel, hidLabel, outLabel);
			objectsRef.current.push(inLabel, hidLabel, outLabel);

			const note = createLabel(
				"ReLU/GELU makes it sparse: many neurons inactive",
				new THREE.Vector3(0, 4, 2),
				colors.ffn,
				0.7,
			);
			scene.add(note);
			objectsRef.current.push(note);
		}

		if (stage.focus === "all_layers") {
			// Show multiple transformer layers stacked
			const numLayers = 4;

			for (let l = 0; l < numLayers; l++) {
				const z = l * 4 - 6;

				inputSequence.forEach((_, i) => {
					const x = startX + i * tokenSpacing;

					// Token representation at this layer
					const geom = new THREE.SphereGeometry(0.35);
					const mat = new THREE.MeshPhongMaterial({
						color: new THREE.Color().lerpColors(
							new THREE.Color(colors.embedding),
							new THREE.Color(colors.output),
							l / (numLayers - 1),
						),
						transparent: true,
						opacity: 0.8,
					});
					const sphere = new THREE.Mesh(geom, mat);
					sphere.position.set(x, 0, z);
					scene.add(sphere);
					objectsRef.current.push(sphere);
				});

				// Layer label
				const label = createLabel(
					`Layer ${l + 1}`,
					new THREE.Vector3(startX - 2, 0, z),
					colors.text,
					0.6,
				);
				scene.add(label);
				objectsRef.current.push(label);

				// Connections to next layer
				if (l < numLayers - 1) {
					inputSequence.forEach((_, i) => {
						const x = startX + i * tokenSpacing;
						const lineGeom = new THREE.BufferGeometry().setFromPoints([
							new THREE.Vector3(x, 0, z + 0.4),
							new THREE.Vector3(x, 0, z + 3.6),
						]);
						const lineMat = new THREE.LineBasicMaterial({
							color: colors.dim,
							transparent: true,
							opacity: 0.4,
						});
						const line = new THREE.Line(lineGeom, lineMat);
						scene.add(line);
						objectsRef.current.push(line);
					});
				}
			}

			const note = createLabel(
				"Each layer refines the representation",
				new THREE.Vector3(0, 3, 0),
				colors.text,
				0.8,
			);
			scene.add(note);
			objectsRef.current.push(note);
		}

		if (stage.focus === "output") {
			// Show final prediction
			const lastTokenX = startX + (numTokens - 1) * tokenSpacing;

			// Prediction distribution
			const predictions = [
				{ word: ".", prob: 0.35 },
				{ word: "today", prob: 0.15 },
				{ word: "down", prob: 0.12 },
				{ word: "still", prob: 0.08 },
				{ word: "quietly", prob: 0.06 },
			];

			predictions.forEach((pred, i) => {
				const barWidth = pred.prob * 6;
				const geom = new THREE.BoxGeometry(barWidth, 0.4, 0.3);
				const mat = new THREE.MeshPhongMaterial({
					color: new THREE.Color(colors.output),
					emissive: new THREE.Color(colors.output),
					emissiveIntensity: pred.prob * 0.5,
				});
				const bar = new THREE.Mesh(geom, mat);
				bar.position.set(lastTokenX + 3 + barWidth / 2, 2 - i * 0.6, 0);
				scene.add(bar);
				objectsRef.current.push(bar);

				const label = createLabel(
					`"${pred.word}" ${(pred.prob * 100).toFixed(0)}%`,
					new THREE.Vector3(lastTokenX + 3 + barWidth + 1.5, 2 - i * 0.6, 0),
					colors.text,
					0.5,
				);
				scene.add(label);
				objectsRef.current.push(label);
			});

			const title = createLabel(
				"Next token prediction:",
				new THREE.Vector3(lastTokenX + 4, 3, 0),
				colors.output,
				0.7,
			);
			scene.add(title);
			objectsRef.current.push(title);

			// Arrow from last token
			const arrowGeom = new THREE.ConeGeometry(0.15, 0.4, 8);
			const arrowMat = new THREE.MeshBasicMaterial({ color: colors.output });
			const arrow = new THREE.Mesh(arrowGeom, arrowMat);
			arrow.rotation.z = -Math.PI / 2;
			arrow.position.set(lastTokenX + 1.5, 0.5, 0);
			scene.add(arrow);
			objectsRef.current.push(arrow);
		}

		// Animate camera to stage position
		animateCameraToStage(stage.camera);
	}, [
		currentStageData,
		inputSequence,
		clearScene,
		createLabel,
		animateCameraToStage,
	]);

	// Rebuild scene when stage changes
	useEffect(() => {
		buildScene();
	}, [buildScene]);

	// Animation loop
	useEffect(() => {
		if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;

		const animate = () => {
			frameRef.current = requestAnimationFrame(animate);
			timeRef.current += 0.016 * playbackSpeed;

			// Telemetry-driven accents (pulse when a training step lands).
			const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
			const pulse = pulseRef.current;
			pulseRef.current = Math.max(0, pulseRef.current - 0.05);
			const loss = telemetryRef.current.loss;
			const tokS = telemetryRef.current.tok_s;
			const heat = clamp01((loss - 2.0) / 6.0);
			const tokFactor = tokS > 0 ? Math.min(3, tokS / 1500.0) : 1;

			// Animate objects
			objectsRef.current.forEach((obj) => {
				if (obj.userData.type === "token") {
					obj.position.y =
						Math.sin(timeRef.current * 2 + obj.userData.tokenIndex * 0.5) *
						(0.05 + pulse * 0.08);
				}
				if (obj.userData.type === "position") {
					obj.rotation.z =
						(timeRef.current + obj.userData.index * 0.5) * (1 + pulse * 0.5);
				}
				if (obj.userData.type === "flowParticle" && obj instanceof THREE.Mesh) {
					const baseSpeed = obj.userData.baseSpeed ?? obj.userData.speed ?? 0.3;
					const speed = baseSpeed * (0.6 + 0.4 * tokFactor) * (1 + pulse * 0.2);
					const t = (timeRef.current * speed + obj.userData.offset) % 1;
					const point = obj.userData.curve.getPoint(t);
					obj.position.copy(point);
					// Make particles a bit "hotter" when loss is high.
					try {
						const mat = obj.material;
						if (mat && !Array.isArray(mat) && mat.color?.setHSL) {
							mat.color.setHSL(0.35 - heat * 0.35, 1, 0.55);
						}
					} catch {
						console.error("Error setting material color");
					}
				}
				if (obj.userData.type === "residual") {
					obj.rotation.z = timeRef.current * 0.5 + obj.userData.layer * 0.3;
				}
				if (obj.userData.type === "ffn_hidden" && obj.userData.active) {
					obj.scale.setScalar(
						1 + Math.sin(timeRef.current * 3) * 0.1 + pulse * 0.15,
					);
				}
				if (
					obj.userData.type === "Q" ||
					obj.userData.type === "K" ||
					obj.userData.type === "V"
				) {
					obj.position.y =
						2.5 +
						Math.sin(timeRef.current * 2 + obj.userData.tokenIndex) *
							(0.1 + pulse * 0.08);
				}
			});

			if (rendererRef.current && sceneRef.current && cameraRef.current) {
				rendererRef.current.render(sceneRef.current, cameraRef.current);
			}
		};

		animate();
		return () => {
			if (frameRef.current !== null) {
				cancelAnimationFrame(frameRef.current);
			}
		};
	}, [playbackSpeed]);

	// Auto-advance when playing
	useEffect(() => {
		if (!isPlaying) return;

		const interval = setInterval(() => {
			setCurrentStage((prev) => {
				if (prev >= stages.length - 1) {
					setIsPlaying(false);
					return prev;
				}
				return prev + 1;
			});
		}, 5000 / playbackSpeed);

		return () => clearInterval(interval);
	}, [isPlaying, playbackSpeed, stages.length]);

	const goToStage = (index: number) => {
		setCurrentStage(Math.max(0, Math.min(stages.length - 1, index)));
	};

	return (
		<div className="w-full h-screen bg-slate-900 flex flex-col overflow-hidden">
			{/* Header */}
			<div className="bg-slate-800 border-b border-slate-700 px-4 py-3 shrink-0">
				<div className="flex items-center justify-between">
					<div>
						<h1 className="text-lg font-bold text-white">
							Transformer Forward Pass
						</h1>
						<p className="text-slate-400 text-xs">
							Step-by-step visualization • target{" "}
							<span className="text-slate-200">{selection.target}</span>
						</p>
						{modelSummary && (
							<p className="text-slate-500 text-xs">
								{modelSummary.type} • layers {modelSummary.n_layers} • d_model{" "}
								{modelSummary.d_model ?? "—"} • heads{" "}
								{modelSummary.n_heads ?? "—"} • {ablationFlags.mode}
								{ablationFlags.null_attn ? " • null" : ""}
								{ablationFlags.tie_qk ? " • tie_qk" : ""}
								{ablationFlags.rope_semantic ? " • rope_sem" : ""}
								{ablationFlags.decoupled_gate ? " • gate" : ""}
							</p>
						)}
					</div>
					<div className="flex items-center gap-3">
						<span className="text-slate-400 text-sm">Input:</span>
						<div className="flex gap-1">
							{inputSequence.map((token) => (
								<span
									key={token}
									className="px-2 py-0.5 bg-slate-700 rounded text-sm text-blue-400"
								>
									{token}
								</span>
							))}
						</div>
					</div>
				</div>
			</div>

			<div className="flex flex-1 min-h-0">
				{/* Left panel - explanation */}
				{showExplanation && (
					<div className="w-80 bg-slate-800 border-r border-slate-700 p-4 flex flex-col overflow-hidden shrink-0">
						<div className="flex items-center justify-between mb-4">
							<h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wide">
								Stage {currentStage + 1} / {stages.length}
							</h2>
							<Button onClick={() => setShowExplanation(false)}>Hide</Button>
						</div>

						<div className="flex-1 overflow-y-auto">
							<h3 className="text-xl font-bold text-white mb-3">
								{currentStageData.name}
							</h3>
							<p className="text-slate-300 mb-4 leading-relaxed">
								{currentStageData.description}
							</p>
							<div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700">
								<p className="text-sm text-slate-400 leading-relaxed">
									{currentStageData.detail}
								</p>
							</div>
						</div>

						{/* Stage progress dots */}
						<div className="flex justify-center gap-1.5 mt-4 pt-4 border-t border-slate-700">
							{stages.map((stage) => (
								<Button
									key={stage.id}
									onClick={() => goToStage(stages.indexOf(stage))}
								>
									{stage.name}
								</Button>
							))}
						</div>
					</div>
				)}

				{/* Main canvas */}
				<div className="flex-1 relative min-w-0">
					<canvas
						ref={canvasRef}
						className="w-full h-full cursor-grab active:cursor-grabbing"
					/>

					{!showExplanation && (
						<Button onClick={() => setShowExplanation(true)}>
							Show Explanation
						</Button>
					)}

					{/* Current stage name overlay */}
					<div className="absolute top-4 right-4 bg-slate-800/80 backdrop-blur rounded-lg px-4 py-2">
						<p className="text-white font-medium">{currentStageData.name}</p>
					</div>
				</div>
			</div>

			{/* Bottom controls */}
			<div className="bg-slate-800 border-t border-slate-700 px-4 py-3 shrink-0">
				<div className="flex items-center justify-between">
					<div className="flex items-center gap-2">
						<Button
							onClick={() => goToStage(currentStage - 1)}
							disabled={currentStage === 0}
						>
							← Previous
						</Button>
						<Button
							onClick={() => setIsPlaying(!isPlaying)}
							className={`px-4 py-1.5 rounded text-white ${
								isPlaying
									? "bg-red-600 hover:bg-red-500"
									: "bg-blue-600 hover:bg-blue-500"
							}`}
						>
							{isPlaying ? "⏸ Pause" : "▶ Play"}
						</Button>
						<Button
							onClick={() => goToStage(currentStage + 1)}
							disabled={currentStage === stages.length - 1}
						>
							Next →
						</Button>
					</div>

					<div className="flex items-center gap-4">
						<div className="flex items-center gap-2">
							<span className="text-slate-400 text-sm">Speed:</span>
							<input
								type="range"
								min="0.5"
								max="2"
								step="0.25"
								value={playbackSpeed}
								onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
								className="w-24 accent-blue-500"
							/>
							<span className="text-slate-300 text-sm w-8">
								{playbackSpeed}x
							</span>
						</div>
					</div>

					<div className="text-slate-500 text-sm">
						Drag to rotate • Scroll to zoom
					</div>
				</div>
			</div>
		</div>
	);
};

export default TransformerEducational;

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { useRun } from "@/lib/run-context";
import type {
	Activation,
	AttentionActivation,
	AttentionLayerConfig,
	EmbeddingLayerConfig,
	FFNActivation,
	FFNLayerConfig,
	LayerNormLayerConfig,
	LinearLayerConfig,
	SimpleActivation,
	TypedLayerConfig,
	VisualBlock,
} from "@/types/layer";
import { isAttentionActivation, isFFNActivation } from "@/types/layer";

const RealisticMLVisualization = () => {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const [activeView, setActiveView] = useState("graph");
	const [zoomLevel, setZoomLevel] = useState("model");
	type LayerInfo = {
		name?: string;
		type?: string;
		layers?: Array<Record<string, unknown>>;
		[key: string]: unknown;
	};
	const [selectedLayer, setSelectedLayer] = useState<LayerInfo | null>(null);
	const [showActivations, setShowActivations] = useState(true);
	const [isRotating, setIsRotating] = useState(true);
	const [inputToken, setInputToken] = useState(0);

	// Live training telemetry (from `caramba serve` control-plane via SSE).
	const {
		metrics,
		lastEvent,
		status: runStatus,
		modelSummary,
		attentionLayers,
		layerStats,
		vizLayers,
	} = useRun();
	const telemetryRef = useRef({ loss: 0, tok_s: 0, step: 0 });
	const pulseRef = useRef(0);
	const lastStepRef = useRef<number | null>(null);

	const sceneRef = useRef<THREE.Scene | null>(null);
	const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
	const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
	const composerRef = useRef<EffectComposer | null>(null);
	const bloomPassRef = useRef<UnrealBloomPass | null>(null);
	const controlsRef = useRef<OrbitControls | null>(null);
	const frameRef = useRef<number | null>(null);
	const objectsRef = useRef<Array<THREE.Object3D>>([]);
	const timeRef = useRef(0);

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

	const layerStatByIndex = useMemo(() => {
		const m = new Map<
			number,
			{ rms: number; mean_abs: number; max_abs: number }
		>();
		if (layerStats) {
			for (const s of layerStats) {
				m.set(s.index, {
					rms: s.rms,
					mean_abs: s.mean_abs,
					max_abs: s.max_abs,
				});
			}
		}
		return m;
	}, [layerStats]);

	const vizByIndex = useMemo(() => {
		const m = new Map<
			number,
			{ attn?: number[][][]; act?: number[][]; n_heads?: number }
		>();
		if (vizLayers) {
			for (const v of vizLayers) {
				m.set(v.index, {
					attn: v.attn?.matrices,
					act: v.act?.values,
					n_heads: v.n_heads,
				});
			}
		}
		return m;
	}, [vizLayers]);

	// Keep latest "viz" payload in a ref so the animation loop can smoothly
	// interpolate without rebuilding the entire scene.
	const vizByIndexRef = useRef(vizByIndex);
	useEffect(() => {
		vizByIndexRef.current = vizByIndex;
	}, [vizByIndex]);

	const activeViewRef = useRef(activeView);
	useEffect(() => {
		activeViewRef.current = activeView;
	}, [activeView]);

	const selectedBlockNum = useMemo(() => {
		const n =
			selectedLayer &&
			typeof (selectedLayer as Record<string, unknown>).blockNum === "number"
				? ((selectedLayer as Record<string, unknown>).blockNum as number)
				: 0;
		return typeof n === "number" && Number.isFinite(n) ? n : 0;
	}, [selectedLayer]);

	const selectedBlockNumRef = useRef(0);
	useEffect(() => {
		selectedBlockNumRef.current = selectedBlockNum;
	}, [selectedBlockNum]);

	// Selected block head count (used by attention view layout + active head cycling).
	const headCountRef = useRef(4);
	useEffect(() => {
		const n = selectedBlockNum;
		const heads =
			vizByIndex.get(n)?.n_heads ??
			attentionLayers?.[n]?.n_heads ??
			modelSummary?.n_heads ??
			null;
		headCountRef.current =
			typeof heads === "number" && Number.isFinite(heads) && heads > 0
				? Math.floor(heads)
				: 4;
	}, [attentionLayers, selectedBlockNum, vizByIndex, modelSummary?.n_heads]);

	const modelConfig = useMemo(() => {
		const layersFromSummary = (): {
			name: string;
			hidden_dim: number;
			num_layers: number;
			num_heads: number;
			ffn_dim: number;
			layers: Array<TypedLayerConfig>;
		} => {
			const n = Math.max(1, modelSummary?.n_layers ?? 6);
			const d = modelSummary?.d_model ?? 768;
			const h = modelSummary?.n_heads ?? 12;
			const vocab = modelSummary?.vocab_size ?? 50257;
			const blocks: Array<TypedLayerConfig> = Array(n)
				.fill(null)
				.flatMap(
					(_unused, i): Array<TypedLayerConfig> => [
						{
							name: `block_${i}_ln1`,
							type: "layernorm",
							dim: d,
							blockNum: i,
							layers: [],
						},
						{
							name: `block_${i}_attn`,
							type: "attention",
							dim: d,
							heads: h,
							head_dim: Math.max(1, Math.floor(d / h)),
							blockNum: i,
							layers: [],
						},
						{
							name: `block_${i}_ln2`,
							type: "layernorm",
							dim: d,
							blockNum: i,
							layers: [],
						},
						{
							name: `block_${i}_ffn`,
							type: "ffn",
							in_dim: d,
							hidden_dim: d * 4,
							out_dim: d,
							blockNum: i,
							layers: [],
						},
					],
				);

			return {
				name: modelSummary ? `${modelSummary.type}` : "TransformerLM-125M",
				hidden_dim: d,
				num_layers: n,
				num_heads: h,
				ffn_dim: d * 4,
				layers: [
					{
						name: "token_embed",
						type: "embedding",
						in_dim: vocab,
						out_dim: d,
						layers: [],
					},
					{
						name: "pos_embed",
						type: "embedding",
						in_dim: 1024,
						out_dim: d,
						layers: [],
					},
					...blocks,
					{ name: "ln_final", type: "layernorm", dim: d, layers: [] },
					{
						name: "lm_head",
						type: "linear",
						in_dim: d,
						out_dim: vocab,
						layers: [],
					},
				] as Array<TypedLayerConfig>,
			};
		};

		return layersFromSummary();
	}, [modelSummary]);

	// Simulated activations
	const activations = useMemo(() => {
		const acts: Record<string, Activation> = {};
		const seqLen = 16;

		modelConfig.layers.forEach((layer) => {
			if (layer.type === "attention" && typeof layer.name === "string") {
				const attnLayer = layer;
				acts[layer.name] = {
					patterns: Array(attnLayer.heads)
						.fill(0)
						.map((_unused1, h) =>
							Array(seqLen)
								.fill(0)
								.map((_unused2, i) =>
									Array(seqLen)
										.fill(0)
										.map((_unused3, j) => {
											if (h === 0) return i === j ? 0.7 : 0.02;
											if (h === 1) return j <= i ? 0.1 : 0;
											if (h === 2) return Math.abs(i - j) < 3 ? 0.3 : 0.01;
											if (h === 3) return (i + j) % 3 === 0 ? 0.4 : 0.05;
											return Math.random() * 0.15;
										}),
								),
						),
					output: Array(seqLen)
						.fill(0)
						.map(() =>
							Array(attnLayer.dim)
								.fill(0)
								.map(() => (Math.random() - 0.5) * 1.5),
						),
				} as AttentionActivation;
			} else if (layer.type === "ffn" && typeof layer.name === "string") {
				const ffnLayer = layer;
				acts[layer.name] = {
					hidden: Array(seqLen)
						.fill(0)
						.map(() =>
							Array(ffnLayer.hidden_dim)
								.fill(0)
								.map(() => (Math.random() > 0.7 ? Math.random() * 2 : 0)),
						),
					output: Array(seqLen)
						.fill(0)
						.map(() =>
							Array(ffnLayer.out_dim)
								.fill(0)
								.map(() => (Math.random() - 0.5) * 1.2),
						),
				} as FFNActivation;
			} else if (
				(layer.type === "embedding" ||
					layer.type === "layernorm" ||
					layer.type === "linear") &&
				typeof layer.name === "string"
			) {
				let dim: number;
				if (layer.type === "embedding") {
					dim = layer.out_dim;
				} else if (layer.type === "layernorm") {
					dim = layer.dim;
				} else {
					dim = layer.out_dim;
				}
				acts[layer.name] = Array(seqLen)
					.fill(0)
					.map(() =>
						Array(dim)
							.fill(0)
							.map(() => (Math.random() - 0.5) * 0.8),
					) as SimpleActivation;
			}
		});

		return acts;
	}, [modelConfig]);

	// --- NOTE ---
	// The rest of this file was originally a standalone demo and expects `modelConfig`
	// and `activations` to exist. We now compute those from the manifest summary when available.

	/*
	 * (The original hardcoded modelConfig/activations block below has been replaced.)
	 */

	const colors = {
		embedding: "#60a5fa",
		attention: "#f472b6",
		ffn: "#fb923c",
		layernorm: "#a78bfa",
		linear: "#4ade80",
		background: "#050a15",
		backgroundDark: "#020408",
		highlight: "#fbbf24",
		energy: "#00ffff",
		inactive: "#1a1a2e",
		inactiveGlow: "#0a0a15",
		particleGlow: "#4fc3f7",
		hotspot: "#ff6b6b",
		cold: "#2d3748",
	};

	// Initialize Three.js with OrbitControls for smooth natural camera movement
	useEffect(() => {
		if (!canvasRef.current) return;

		const scene = new THREE.Scene();
		scene.background = new THREE.Color(colors.background);
		scene.fog = new THREE.FogExp2(colors.backgroundDark, 0.015);
		sceneRef.current = scene;

		const camera = new THREE.PerspectiveCamera(
			50,
			canvasRef.current.clientWidth / canvasRef.current.clientHeight,
			0.1,
			1000,
		);
		// Set initial camera position
		camera.position.set(0, 15, 25);
		camera.lookAt(0, 0, 0);
		cameraRef.current = camera;

		const renderer = new THREE.WebGLRenderer({
			canvas: canvasRef.current,
			antialias: true,
			alpha: true,
		});
		renderer.setSize(
			canvasRef.current.clientWidth,
			canvasRef.current.clientHeight,
		);
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		renderer.toneMapping = THREE.ACESFilmicToneMapping;
		renderer.toneMappingExposure = 1.2;
		rendererRef.current = renderer;

		// OrbitControls with smooth damping for natural feeling
		const controls = new OrbitControls(camera, renderer.domElement);
		controls.enableDamping = true;
		controls.dampingFactor = 0.05;
		controls.rotateSpeed = 0.8;
		controls.zoomSpeed = 1.0;
		controls.panSpeed = 0.8;
		controls.minDistance = 5;
		controls.maxDistance = 60;
		controls.maxPolarAngle = Math.PI * 0.85; // Prevent going below the ground
		controls.target.set(0, 0, 0);
		controls.autoRotate = true;
		controls.autoRotateSpeed = 0.5;
		controlsRef.current = controls;

		// Stop auto-rotate when user interacts
		const onControlStart = () => {
			controls.autoRotate = false;
			setIsRotating(false);
		};
		controls.addEventListener("start", onControlStart);

		// Post-processing with bloom
		const composer = new EffectComposer(renderer);
		const renderPass = new RenderPass(scene, camera);
		composer.addPass(renderPass);

		const bloomPass = new UnrealBloomPass(
			new THREE.Vector2(
				canvasRef.current.clientWidth,
				canvasRef.current.clientHeight,
			),
			0.8, // strength
			0.4, // radius
			0.85, // threshold
		);
		composer.addPass(bloomPass);
		composerRef.current = composer;
		bloomPassRef.current = bloomPass;

		// Enhanced Lighting with colored point lights for atmosphere
		const ambientLight = new THREE.AmbientLight(0x111122, 0.3);
		scene.add(ambientLight);

		const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
		dirLight.position.set(10, 20, 15);
		scene.add(dirLight);

		// Cyan accent light (left)
		const cyanLight = new THREE.PointLight(0x00ffff, 0.8, 60);
		cyanLight.position.set(-15, 5, 10);
		scene.add(cyanLight);

		// Magenta accent light (right)
		const magentaLight = new THREE.PointLight(0xff00ff, 0.5, 60);
		magentaLight.position.set(15, 5, -10);
		scene.add(magentaLight);

		// Warm highlight light (top)
		const warmLight = new THREE.PointLight(0xffaa00, 0.4, 50);
		warmLight.position.set(0, 15, 0);
		scene.add(warmLight);

		// Animated grid with gradient fade
		const gridGeometry = new THREE.PlaneGeometry(80, 80, 80, 80);
		const gridMaterial = new THREE.ShaderMaterial({
			uniforms: {
				uTime: { value: 0 },
				uColor1: { value: new THREE.Color(0x1a1a3e) },
				uColor2: { value: new THREE.Color(0x0a0a1a) },
			},
			vertexShader: `
				varying vec2 vUv;
				varying float vDistance;
				void main() {
					vUv = uv;
					vec4 worldPos = modelMatrix * vec4(position, 1.0);
					vDistance = length(worldPos.xz);
					gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
				}
			`,
			fragmentShader: `
				uniform float uTime;
				uniform vec3 uColor1;
				uniform vec3 uColor2;
				varying vec2 vUv;
				varying float vDistance;
				void main() {
					vec2 grid = abs(fract(vUv * 40.0 - 0.5) - 0.5) / fwidth(vUv * 40.0);
					float line = min(grid.x, grid.y);
					float gridPattern = 1.0 - min(line, 1.0);
					float fade = 1.0 - smoothstep(0.0, 35.0, vDistance);
					float pulse = 0.5 + 0.5 * sin(uTime * 0.5 - vDistance * 0.1);
					vec3 color = mix(uColor2, uColor1, pulse * 0.3);
					gl_FragColor = vec4(color, gridPattern * fade * 0.4);
				}
			`,
			transparent: true,
			side: THREE.DoubleSide,
		});
		const gridMesh = new THREE.Mesh(gridGeometry, gridMaterial);
		gridMesh.rotation.x = -Math.PI / 2;
		gridMesh.position.y = -5;
		gridMesh.userData = { isAnimatedGrid: true, material: gridMaterial };
		scene.add(gridMesh);

		// Resize handler
		const handleResize = () => {
			if (!canvasRef.current) return;
			const width = canvasRef.current.clientWidth;
			const height = canvasRef.current.clientHeight;
			camera.aspect = width / height;
			camera.updateProjectionMatrix();
			renderer.setSize(width, height);
			composer.setSize(width, height);
			bloomPass.setSize(width, height);
		};
		window.addEventListener("resize", handleResize);

		// Prevent context menu on right-click (allows right-drag for panning)
		const handleContextMenu = (e: MouseEvent) => {
			e.preventDefault();
		};
		renderer.domElement.addEventListener("contextmenu", handleContextMenu);

		return () => {
			window.removeEventListener("resize", handleResize);
			renderer.domElement.removeEventListener("contextmenu", handleContextMenu);
			controls.removeEventListener("start", onControlStart);
			controls.dispose();
			if (frameRef.current !== null) {
				cancelAnimationFrame(frameRef.current);
			}
			composer.dispose();
			renderer.dispose();
		};
	}, []);

	// Click handler for layer selection
	const handleCanvasClick = useCallback((e: MouseEvent) => {
		if (!canvasRef.current || !cameraRef.current || !sceneRef.current) return;

		const rect = canvasRef.current.getBoundingClientRect();
		const mouse = new THREE.Vector2(
			((e.clientX - rect.left) / rect.width) * 2 - 1,
			-((e.clientY - rect.top) / rect.height) * 2 + 1,
		);

		const raycaster = new THREE.Raycaster();
		raycaster.setFromCamera(mouse, cameraRef.current);

		const clickableObjects = objectsRef.current.filter(
			(obj) => obj.userData.clickable,
		);
		const intersects = raycaster.intersectObjects(clickableObjects, false);

		if (intersects.length > 0) {
			const obj = intersects[0].object;
			if (obj.userData.layerData) {
				setSelectedLayer(obj.userData.layerData);
				setZoomLevel("layer");
				// Smoothly move camera to focus on selected layer
				if (controlsRef.current && cameraRef.current) {
					const targetPos = obj.position.clone();
					controlsRef.current.target.copy(targetPos);
				}
			}
		}
	}, []);

	useEffect(() => {
		if (!canvasRef.current) return;
		canvasRef.current.addEventListener("click", handleCanvasClick);
		return () =>
			canvasRef.current?.removeEventListener("click", handleCanvasClick);
	}, [handleCanvasClick]);

	// Clear and rebuild scene
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

	// Build model-level view
	const buildModelView = useCallback(() => {
		const scene = sceneRef.current;
		if (!scene) return;

		// Group layers into visual blocks
		const visualBlocks: Array<VisualBlock> = [
			{
				name: "Embed",
				type: "embedding",
				layers: modelConfig.layers.filter(
					(l): l is EmbeddingLayerConfig => l.type === "embedding",
				),
			},
			...Array(modelConfig.num_layers)
				.fill(null)
				.map(
					(_, i): VisualBlock => ({
						name: `Block ${i}`,
						type: "transformer",
						blockNum: i,
						layers: modelConfig.layers.filter(
							(l): l is TypedLayerConfig & { blockNum: number } =>
								"blockNum" in l && l.blockNum === i,
						),
					}),
				),
			{
				name: "Output",
				type: "output",
				layers: modelConfig.layers.filter(
					(l): l is LayerNormLayerConfig | LinearLayerConfig =>
						l.name === "ln_final" || l.name === "lm_head",
				),
			},
		];

		const spacing = 3;
		const startX = (-(visualBlocks.length - 1) * spacing) / 2;

		visualBlocks.forEach((block, i) => {
			// Determine geometry based on block type
			let geometry: THREE.BufferGeometry | undefined;
			let color: string | undefined;
			const height = 2;

			if (block.type === "embedding") {
				geometry = new THREE.CylinderGeometry(0.8, 1, height, 8);
				color = colors.embedding;
			} else if (
				block.type === "transformer" &&
				"blockNum" in block &&
				typeof block.blockNum === "number"
			) {
				// Use icosahedron for a more crystalline/tech look
				geometry = new THREE.IcosahedronGeometry(1.2, 1);
				// Color can reflect attention mode (decoupled vs standard).
				const attn = attentionLayers?.[block.blockNum] ?? null;
				const mode = String(attn?.mode ?? "").toLowerCase();
				color = mode === "decoupled" ? "#f472b6" : colors.attention;
			} else {
				geometry = new THREE.OctahedronGeometry(0.9);
				color = colors.linear;
			}

			// Enhanced material with fresnel-like edge glow effect
			const material = new THREE.MeshPhongMaterial({
				color: new THREE.Color(color),
				emissive: new THREE.Color(color),
				emissiveIntensity: 0.3,
				transparent: true,
				opacity: 0.92,
				shininess: 100,
				specular: new THREE.Color(0xffffff),
			});

			const mesh = new THREE.Mesh(geometry, material);
			mesh.position.set(startX + i * spacing, 0, 0);
			mesh.userData = {
				clickable: true,
				layerData: block,
				blockIndex: i,
				baseColor: color,
				layerMetricRms:
					block.type === "transformer" &&
					"blockNum" in block &&
					typeof block.blockNum === "number"
						? (layerStatByIndex.get(block.blockNum)?.rms ?? null)
						: null,
			};

			scene.add(mesh);
			objectsRef.current.push(mesh);

			// --- DBA-specific decorations for transformer blocks ---
			if (
				block.type === "transformer" &&
				"blockNum" in block &&
				typeof block.blockNum === "number"
			) {
				const attn = attentionLayers?.[block.blockNum] ?? null;
				if (attn && String(attn.mode).toLowerCase() === "decoupled") {
					// sem/geo split indicator (two thin plates behind the block)
					const sem = typeof attn.sem_dim === "number" ? attn.sem_dim : 0;
					const geo = typeof attn.geo_dim === "number" ? attn.geo_dim : 0;
					const total = Math.max(1, sem + geo);
					const semRatio = sem / total;
					const geoRatio = geo / total;

					const plateW = 1.8;
					const plateH = 0.08;
					const plateD = 1.8;
					const baseZ = -1.05;

					const semPlate = new THREE.Mesh(
						new THREE.BoxGeometry(
							plateW * Math.max(0.15, semRatio),
							plateH,
							plateD,
						),
						new THREE.MeshBasicMaterial({
							color: 0xf472b6, // semantic (pink)
							transparent: true,
							opacity: 0.7,
						}),
					);
					semPlate.position.set(
						startX +
							i * spacing -
							(plateW * (1 - Math.max(0.15, semRatio))) / 2,
						-height / 2 - 0.55,
						baseZ,
					);
					semPlate.userData = { isDbaPlate: true };
					scene.add(semPlate);
					objectsRef.current.push(semPlate);

					const geoPlate = new THREE.Mesh(
						new THREE.BoxGeometry(
							plateW * Math.max(0.15, geoRatio),
							plateH,
							plateD,
						),
						new THREE.MeshBasicMaterial({
							color: 0x4ade80, // geometric (green)
							transparent: true,
							opacity: 0.7,
						}),
					);
					geoPlate.position.set(
						startX +
							i * spacing +
							(plateW * (1 - Math.max(0.15, geoRatio))) / 2,
						-height / 2 - 0.55,
						baseZ,
					);
					geoPlate.userData = { isDbaPlate: true };
					scene.add(geoPlate);
					objectsRef.current.push(geoPlate);

					// null_attn indicator (small "sink" sphere above)
					if (attn.null_attn) {
						const sink = new THREE.Mesh(
							new THREE.SphereGeometry(0.18),
							new THREE.MeshBasicMaterial({ color: 0xe0af68 }),
						);
						sink.position.set(startX + i * spacing, height / 2 + 0.55, 0);
						sink.userData = { isNullSink: true, blockIndex: i };
						scene.add(sink);
						objectsRef.current.push(sink);
					}

					// tie_qk indicator (small ring)
					if (attn.tie_qk) {
						const ring = new THREE.Mesh(
							new THREE.TorusGeometry(0.35, 0.05, 8, 24),
							new THREE.MeshBasicMaterial({
								color: 0x60a5fa,
								transparent: true,
								opacity: 0.8,
							}),
						);
						ring.rotation.x = Math.PI / 2;
						ring.position.set(startX + i * spacing, height / 2 + 0.15, 0.9);
						ring.userData = { isTieQk: true };
						scene.add(ring);
						objectsRef.current.push(ring);
					}

					// rope_semantic indicator (small dot)
					if (attn.rope_semantic) {
						const dot = new THREE.Mesh(
							new THREE.SphereGeometry(0.08),
							new THREE.MeshBasicMaterial({ color: 0xbb9af7 }),
						);
						dot.position.set(startX + i * spacing, height / 2 + 0.15, -0.9);
						dot.userData = { isRopeSem: true };
						scene.add(dot);
						objectsRef.current.push(dot);
					}
				}
			}

			// Glow ring for activation intensity
			if (showActivations && block.layers[0]) {
				const ringGeom = new THREE.TorusGeometry(1.3, 0.06, 8, 32);
				const intensity = 0.3 + Math.random() * 0.4; // Simulated
				const ringMat = new THREE.MeshBasicMaterial({
					color: new THREE.Color().setHSL(0.35 - intensity * 0.35, 1, 0.5),
					transparent: true,
					opacity: 0.8,
				});
				const ring = new THREE.Mesh(ringGeom, ringMat);
				ring.rotation.x = Math.PI / 2;
				ring.position.set(startX + i * spacing, -height / 2 - 0.3, 0);
				ring.userData = { isGlowRing: true, blockIndex: i };
				scene.add(ring);
				objectsRef.current.push(ring);
			}

			// Connection to next block with glowing energy tube
			if (i < visualBlocks.length - 1) {
				const curve = new THREE.CatmullRomCurve3([
					new THREE.Vector3(startX + i * spacing + 1.3, 0, 0),
					new THREE.Vector3(startX + (i + 0.5) * spacing, 1.2, 0.4),
					new THREE.Vector3(startX + (i + 1) * spacing - 1.3, 0, 0),
				]);

				// Outer glow tube
				const glowTubeGeom = new THREE.TubeGeometry(curve, 32, 0.12, 8, false);
				const glowTubeMat = new THREE.MeshBasicMaterial({
					color: 0x00aaff,
					transparent: true,
					opacity: 0.15,
				});
				const glowTube = new THREE.Mesh(glowTubeGeom, glowTubeMat);
				glowTube.userData = { isConnectionGlow: true };
				scene.add(glowTube);
				objectsRef.current.push(glowTube);

				// Inner core tube
				const tubeGeom = new THREE.TubeGeometry(curve, 32, 0.04, 8, false);
				const tubeMat = new THREE.MeshBasicMaterial({
					color: 0x4fc3f7,
					transparent: true,
					opacity: 0.6,
				});
				const tube = new THREE.Mesh(tubeGeom, tubeMat);
				scene.add(tube);
				objectsRef.current.push(tube);

				// Animated particles along connection
				for (let p = 0; p < 3; p++) {
					const particleGeom = new THREE.SphereGeometry(0.08);
					const particleMat = new THREE.MeshBasicMaterial({
						color: 0x60a5fa,
						transparent: true,
						opacity: 0.8,
					});
					const particle = new THREE.Mesh(particleGeom, particleMat);
					const baseSpeed = 0.3 + Math.random() * 0.2;
					particle.userData = {
						curve,
						offset: p / 3,
						speed: baseSpeed,
						baseSpeed,
						isParticle: true,
						connectionIndex: i,
					};
					scene.add(particle);
					objectsRef.current.push(particle);
				}
			}
		});

		// Reset camera for model view
		if (controlsRef.current) {
			controlsRef.current.target.set(0, 0, 0);
		}
	}, [modelConfig, showActivations, attentionLayers, layerStatByIndex]);

	// Build layer-level view
	const buildLayerView = useCallback(() => {
		const scene = sceneRef.current;
		const layer = selectedLayer;
		if (!scene || !layer) return;

		if (layer.type === "transformer" && layer.layers) {
			// Show attention heads and FFN structure
			const attnLayer = layer.layers.find(
				(l): l is AttentionLayerConfig => l.type === "attention",
			);
			const ffnLayer = layer.layers.find(
				(l): l is FFNLayerConfig => l.type === "ffn",
			);

			// Attention heads on the left
			if (attnLayer) {
				const heads = attnLayer.heads;
				const cols = 4;
				const rows = Math.ceil(heads / cols);

				for (let h = 0; h < heads; h++) {
					const row = Math.floor(h / cols);
					const col = h % cols;

					const geom = new THREE.OctahedronGeometry(0.5);
					const mat = new THREE.MeshPhongMaterial({
						color: new THREE.Color(colors.attention),
						emissive: new THREE.Color(colors.attention),
						emissiveIntensity: 0.3,
						transparent: true,
						opacity: 0.9,
					});

					const mesh = new THREE.Mesh(geom, mat);
					mesh.position.set(-4 + col * 1.2, (row - rows / 2 + 0.5) * 1.2, 0);
					mesh.userData = { headIndex: h, isHead: true, baseEmissive: 0.3 };
					scene.add(mesh);
					objectsRef.current.push(mesh);

					// Q K V indicators
					["Q", "K", "V"].forEach((_, idx) => {
						const qkvGeom = new THREE.SphereGeometry(0.12);
						const qkvMat = new THREE.MeshBasicMaterial({
							color: [0x60a5fa, 0x4ade80, 0xfbbf24][idx],
						});
						const qkv = new THREE.Mesh(qkvGeom, qkvMat);
						qkv.position.set(
							-4 + col * 1.2 + (idx - 1) * 0.25,
							(row - rows / 2 + 0.5) * 1.2 + 0.7,
							0,
						);
						scene.add(qkv);
						objectsRef.current.push(qkv);
					});
				}

				// Label for attention section
				const labelGeom = new THREE.PlaneGeometry(3, 0.4);
				const labelMat = new THREE.MeshBasicMaterial({
					color: colors.attention,
					transparent: true,
					opacity: 0.3,
					side: THREE.DoubleSide,
				});
				const label = new THREE.Mesh(labelGeom, labelMat);
				label.position.set(-2.5, -2.5, 0);
				scene.add(label);
				objectsRef.current.push(label);
			}

			// FFN structure on the right
			if (ffnLayer) {
				const layers3D = [
					{ dim: ffnLayer.in_dim, x: 3, color: colors.embedding, sample: 24 },
					{ dim: ffnLayer.hidden_dim, x: 5.5, color: colors.ffn, sample: 48 },
					{ dim: ffnLayer.out_dim, x: 8, color: colors.linear, sample: 24 },
				];

				layers3D.forEach((l, lidx) => {
					const cols = Math.ceil(Math.sqrt(l.sample));
					const rows = Math.ceil(l.sample / cols);

					for (let i = 0; i < l.sample; i++) {
						const row = Math.floor(i / cols);
						const col = i % cols;

						// Get activation if available
						let intensity = 0.5;
						if (showActivations && typeof ffnLayer.name === "string") {
							const act = activations[ffnLayer.name];
							if (isFFNActivation(act)) {
								const values = lidx === 1 ? act.hidden : act.output;
								if (Array.isArray(values) && values[inputToken]) {
									const dim =
										lidx === 1 ? ffnLayer.hidden_dim : ffnLayer.out_dim;
									const idx = Math.floor((i * dim) / l.sample);
									intensity = Math.min(
										1,
										Math.abs(values[inputToken][idx] || 0),
									);
								}
							}
						}

						const geom = new THREE.SphereGeometry(0.12);
						const mat = new THREE.MeshBasicMaterial({
							color: new THREE.Color().setHSL(
								showActivations ? 0.35 - intensity * 0.35 : 0.6,
								0.8,
								0.35 + intensity * 0.4,
							),
						});

						const mesh = new THREE.Mesh(geom, mat);
						mesh.position.set(
							l.x,
							(row - rows / 2 + 0.5) * 0.35,
							(col - cols / 2 + 0.5) * 0.35,
						);
						scene.add(mesh);
						objectsRef.current.push(mesh);
					}
				});

				// Sample connections between FFN layers
				for (let c = 0; c < 40; c++) {
					const fromL = c < 20 ? 0 : 1;
					const toL = fromL + 1;
					const from = layers3D[fromL];
					const to = layers3D[toL];

					const fromCols = Math.ceil(Math.sqrt(from.sample));
					const fromRows = Math.ceil(from.sample / fromCols);
					const toCols = Math.ceil(Math.sqrt(to.sample));
					const toRows = Math.ceil(to.sample / toCols);

					const fi = Math.floor(Math.random() * from.sample);
					const ti = Math.floor(Math.random() * to.sample);

					const fromRow = Math.floor(fi / fromCols);
					const fromCol = fi % fromCols;
					const toRow = Math.floor(ti / toCols);
					const toCol = ti % toCols;

					const fy = (fromRow - fromRows / 2 + 0.5) * 0.35;
					const fz = (fromCol - fromCols / 2 + 0.5) * 0.35;
					const ty = (toRow - toRows / 2 + 0.5) * 0.35;
					const tz = (toCol - toCols / 2 + 0.5) * 0.35;

					const curve = new THREE.CatmullRomCurve3([
						new THREE.Vector3(from.x, fy, fz),
						new THREE.Vector3(
							(from.x + to.x) / 2,
							(fy + ty) / 2,
							(fz + tz) / 2 + 0.2,
						),
						new THREE.Vector3(to.x, ty, tz),
					]);

					const geom = new THREE.TubeGeometry(curve, 8, 0.015, 4, false);
					const mat = new THREE.MeshBasicMaterial({
						color: 0x475569,
						transparent: true,
						opacity: 0.25,
					});
					const tube = new THREE.Mesh(geom, mat);
					scene.add(tube);
					objectsRef.current.push(tube);
				}
			}
		} else if (layer.type === "embedding" || layer.type === "output") {
			// Show embedding/output as matrix
			const vocabSample = 48;
			const dimSample = 32;
			const spacing = 0.2;

			for (let v = 0; v < vocabSample; v++) {
				for (let d = 0; d < dimSample; d++) {
					const value = (Math.sin(v * 0.15) * Math.cos(d * 0.2) + 1) / 2;

					const geom = new THREE.BoxGeometry(
						spacing * 0.85,
						value * 0.6 + 0.05,
						spacing * 0.85,
					);
					const mat = new THREE.MeshPhongMaterial({
						color: new THREE.Color().setHSL(0.55, 0.7, 0.3 + value * 0.4),
					});

					const mesh = new THREE.Mesh(geom, mat);
					mesh.position.set(
						(v - vocabSample / 2) * spacing,
						value * 0.3,
						(d - dimSample / 2) * spacing,
					);
					scene.add(mesh);
					objectsRef.current.push(mesh);
				}
			}
		}

		if (controlsRef.current) {
			controlsRef.current.target.set(
				layer.type === "transformer" ? 2 : 0,
				0,
				0,
			);
		}
	}, [selectedLayer, showActivations, activations, inputToken]);

	// Build attention view
	const buildAttentionView = useCallback(() => {
		const scene = sceneRef.current;
		if (!scene) return;

		const seqLen = 16;
		const numHeadsToShow = Math.max(1, Math.floor(headCountRef.current || 1));
		const cols = Math.ceil(Math.sqrt(numHeadsToShow));
		const rows = Math.ceil(numHeadsToShow / cols);
		const gridSize = 0.25;

		const mats = vizByIndexRef.current.get(selectedBlockNumRef.current)?.attn;

		for (let h = 0; h < numHeadsToShow; h++) {
			const col = h % cols;
			const row = Math.floor(h / cols);
			const offsetX = (col - (cols - 1) / 2) * (seqLen * gridSize + 2.0);
			const offsetZ = (row - (rows - 1) / 2) * (seqLen * gridSize + 2.0);

			// Prefer real attention matrices from training viz; fall back to demo patterns.
			const pattern =
				mats && Array.isArray(mats[h]) ? (mats[h] as number[][]) : undefined;

			// Add a subtle base plate for each head
			const basePlate = new THREE.Mesh(
				new THREE.PlaneGeometry(
					seqLen * gridSize + 0.5,
					seqLen * gridSize + 0.5,
				),
				new THREE.MeshBasicMaterial({
					color: 0x0a0a1a,
					transparent: true,
					opacity: 0.6,
					side: THREE.DoubleSide,
				}),
			);
			basePlate.rotation.x = -Math.PI / 2;
			basePlate.position.set(offsetX, -1.05, offsetZ);
			basePlate.userData = { isBasePlate: true, head: h };
			scene.add(basePlate);
			objectsRef.current.push(basePlate);

			for (let i = 0; i < seqLen; i++) {
				for (let j = 0; j < seqLen; j++) {
					const value = pattern?.[i]?.[j] ?? Math.random() * 0.3;
					const height = Math.max(value * 3.0, 0.02);

					// Use unit-height geometry + scale.y so we can animate smoothly.
					const geom = new THREE.BoxGeometry(
						gridSize * 0.88,
						1.0,
						gridSize * 0.88,
					);

					// Enhanced material with emissive glow for high values
					const mat = new THREE.MeshPhongMaterial({
						color: new THREE.Color().setHSL(
							0.55 - value * 0.45,
							0.9,
							0.3 + value * 0.4,
						),
						emissive: new THREE.Color().setHSL(
							0.55 - value * 0.45,
							1,
							value * 0.3,
						),
						emissiveIntensity: value * 0.5,
						transparent: true,
						opacity: 0.9,
						shininess: 80,
					});

					const mesh = new THREE.Mesh(geom, mat);
					mesh.scale.y = height;
					mesh.position.set(
						offsetX + (i - seqLen / 2) * gridSize,
						height / 2 - 1,
						offsetZ + (j - seqLen / 2) * gridSize,
					);
					mesh.userData = {
						isAttnBar: true,
						head: h,
						i,
						j,
						baseY: -1,
						targetHeight: height,
					};
					scene.add(mesh);
					objectsRef.current.push(mesh);
				}
			}

			// Enhanced head marker with glow
			const indicatorGeom = new THREE.IcosahedronGeometry(0.22, 1);
			const indicatorMat = new THREE.MeshPhongMaterial({
				color: colors.highlight,
				emissive: colors.highlight,
				emissiveIntensity: 0.5,
				transparent: true,
				opacity: 0.9,
			});
			const indicator = new THREE.Mesh(indicatorGeom, indicatorMat);
			indicator.position.set(
				offsetX,
				2.0,
				offsetZ - (seqLen * gridSize) / 2 - 0.8,
			);
			indicator.userData = { isHeadIndicator: true, head: h };
			scene.add(indicator);
			objectsRef.current.push(indicator);

			// Add orbiting ring around active head indicator
			const orbitRing = new THREE.Mesh(
				new THREE.TorusGeometry(0.35, 0.02, 8, 32),
				new THREE.MeshBasicMaterial({
					color: colors.highlight,
					transparent: true,
					opacity: 0.6,
				}),
			);
			orbitRing.position.copy(indicator.position);
			orbitRing.userData = { isOrbitRing: true, head: h };
			scene.add(orbitRing);
			objectsRef.current.push(orbitRing);
		}

		if (controlsRef.current) {
			controlsRef.current.target.set(0, 0, 0);
		}
		// IMPORTANT: don't depend on vizByIndex; animation loop reads refs.
	}, []);

	// Build activation flow view
	const buildActivationView = useCallback(() => {
		const scene = sceneRef.current;
		if (!scene) return;

		const seqLen = 16;
		const numLayers = modelConfig.num_layers;
		const dimSample = 48;

		for (let l = 0; l < numLayers; l++) {
			const layerName = `block_${l}_attn`;
			const actRaw = activations[layerName];
			let act: Array<Array<number>> | undefined;
			if (isAttentionActivation(actRaw)) {
				act = actRaw.output;
			}

			for (let t = 0; t < seqLen; t++) {
				for (let d = 0; d < dimSample; d++) {
					// Prefer real activation samples from training viz.
					const sample = vizByIndexRef.current.get(l)?.act;
					const dimIdx = sample
						? Math.floor((d * (sample[0]?.length ?? 1)) / dimSample)
						: Math.floor((d * 768) / dimSample);
					const value = sample
						? Math.abs(sample[t]?.[dimIdx] ?? 0)
						: act
							? Math.abs(act[t]?.[dimIdx] || 0)
							: Math.random() * 0.5;

					// Unit-height bars for smooth animation.
					const height = value * 0.4 + 0.02;
					const geom = new THREE.BoxGeometry(0.15, 1.0, 0.15);
					const mat = new THREE.MeshBasicMaterial({
						color: new THREE.Color().setHSL(
							0.55 - value * 0.4,
							0.8,
							0.4 + value * 0.3,
						),
						transparent: true,
						opacity: 0.75,
					});

					const mesh = new THREE.Mesh(geom, mat);
					mesh.scale.y = height;
					mesh.position.set(
						(l - numLayers / 2) * 1.8,
						(d - dimSample / 2) * 0.18,
						(t - seqLen / 2) * 0.5,
					);
					mesh.userData = {
						isActBar: true,
						layer: l,
						token: t,
						dim: d,
						targetHeight: height,
					};
					scene.add(mesh);
					objectsRef.current.push(mesh);
				}
			}
		}

		if (controlsRef.current) {
			controlsRef.current.target.set(0, 0, 0);
		}
		// IMPORTANT: don't depend on vizByIndex; animation loop reads refs.
	}, [modelConfig.num_layers, activations]);

	// Rebuild scene when view changes
	useEffect(() => {
		clearScene();

		if (activeView === "graph") {
			if (zoomLevel === "model") {
				buildModelView();
			} else if (zoomLevel === "layer") {
				buildLayerView();
			}
		} else if (activeView === "attention") {
			buildAttentionView();
		} else if (activeView === "activations") {
			buildActivationView();
		}
	}, [
		activeView,
		zoomLevel,
		clearScene,
		buildModelView,
		buildLayerView,
		buildAttentionView,
		buildActivationView,
	]);

	// Animation loop
	useEffect(() => {
		if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;

		const animate = () => {
			frameRef.current = requestAnimationFrame(animate);
			timeRef.current += 0.016;

			// Telemetry-driven visuals (pulses on new metric steps).
			const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
			const pulse = pulseRef.current;
			pulseRef.current = Math.max(0, pulseRef.current - 0.025);
			const loss = telemetryRef.current.loss;
			const tokS = telemetryRef.current.tok_s;
			const step = telemetryRef.current.step;
			const blocksCount = modelConfig.num_layers + 2;
			const activeBlock =
				blocksCount > 0
					? ((step % blocksCount) + blocksCount) % blocksCount
					: -1;
			const heat = clamp01((loss - 2.0) / 6.0); // approx: 0=good, 1=bad
			const tokFactor = tokS > 0 ? Math.min(3, tokS / 1500.0) : 1;

			// Dynamic bloom intensity based on activity
			if (bloomPassRef.current) {
				const targetStrength = 0.5 + pulse * 0.6;
				bloomPassRef.current.strength +=
					(targetStrength - bloomPassRef.current.strength) * 0.1;
			}

			// Animate grid shader
			objectsRef.current.forEach((obj) => {
				if (
					obj.userData.isAnimatedGrid &&
					obj.userData.material?.uniforms?.uTime
				) {
					obj.userData.material.uniforms.uTime.value = timeRef.current;
				}
			});

			// Update OrbitControls (enables smooth damping)
			if (controlsRef.current) {
				// Sync auto-rotate state with isRotating
				controlsRef.current.autoRotate = isRotating;
				controlsRef.current.update();
			}

			// Animate objects
			objectsRef.current.forEach((obj) => {
				if (!(obj instanceof THREE.Mesh)) return;

				// Smoothly animate attention heatmap bars from live viz.
				if (obj.userData.isAttnBar && activeViewRef.current === "attention") {
					const mats = vizByIndexRef.current.get(
						selectedBlockNumRef.current,
					)?.attn;
					const h = obj.userData.head as number;
					const i = obj.userData.i as number;
					const j = obj.userData.j as number;
					const baseY = (obj.userData.baseY as number) ?? -1;
					// Cycle one "active" head based on training step.
					const headCount = Math.max(1, Math.floor(headCountRef.current || 1));
					const step = telemetryRef.current.step || 0;
					const activeHead = headCount > 0 ? step % headCount : 0;
					const isActive = h === activeHead;

					// If we don't have data for this head, treat it as inactive.
					const hasData = Boolean(mats?.[h]);
					const v =
						mats?.[h]?.[i]?.[j] ??
						(Math.max(
							0,
							Math.min(1, Number(obj.userData.lastValue ?? 0)),
						) as number);
					obj.userData.lastValue = v;
					const target = Math.max(v * 2.5, 0.03);
					obj.userData.targetHeight = target;

					const cur = typeof obj.scale.y === "number" ? obj.scale.y : 0.03;
					const next = cur + (target - cur) * 0.18;
					obj.scale.y = next;
					obj.position.y = baseY + next / 2;

					const mat = obj.material;
					if (mat && !Array.isArray(mat)) {
						try {
							if ("color" in mat) {
								if (isActive && hasData) {
									// Active head: colorful with glow based on value
									mat.color.setHSL(0.55 - v * 0.45, 0.9, 0.35 + v * 0.4);
								} else {
									// Inactive head: greyscale
									const l = 0.15 + Math.min(0.5, v * 0.6);
									mat.color.setHSL(0.0, 0.0, l);
								}
							}
							if ("opacity" in mat) {
								// Active: nearly opaque, inactive: more transparent
								const targetOpacity = isActive ? 0.92 : 0.4;
								mat.opacity += (targetOpacity - mat.opacity) * 0.15;
							}
							if ("emissiveIntensity" in mat) {
								// Glow based on value for active heads
								const targetGlow = isActive ? v * 0.6 : 0;
								mat.emissiveIntensity +=
									(targetGlow - mat.emissiveIntensity) * 0.15;
							}
						} catch {
							// ignore
						}
					}
				}

				// Head indicator: colored for active, greyscale for inactive
				if (
					obj.userData.isHeadIndicator &&
					activeViewRef.current === "attention"
				) {
					const headCount = Math.max(1, Math.floor(headCountRef.current || 1));
					const step = telemetryRef.current.step || 0;
					const activeHead = headCount > 0 ? step % headCount : 0;
					const h = obj.userData.head as number;
					const isActive = h === activeHead;
					const mat = obj.material;

					// Gentle rotation
					obj.rotation.y += isActive ? 0.03 : 0.005;

					if (mat && !Array.isArray(mat)) {
						try {
							if ("color" in mat) {
								if (isActive) {
									// Colored for active
									mat.color.setHSL(0.12, 0.9, 0.55);
								} else {
									// Greyscale for inactive
									mat.color.setHSL(0, 0, 0.3);
								}
							}
							if ("opacity" in mat) {
								// Active: less transparent, inactive: more transparent
								const targetOpacity = isActive ? 0.95 : 0.35;
								mat.opacity += (targetOpacity - mat.opacity) * 0.12;
							}
							if ("emissiveIntensity" in mat) {
								// Glow for active
								const targetGlow = isActive ? 0.6 + pulse * 0.3 : 0;
								mat.emissiveIntensity +=
									(targetGlow - mat.emissiveIntensity) * 0.15;
							}
						} catch {
							// ignore
						}
					}
				}

				// Animate orbit rings around head indicators
				if (obj.userData.isOrbitRing && activeViewRef.current === "attention") {
					const headCount = Math.max(1, Math.floor(headCountRef.current || 1));
					const step = telemetryRef.current.step || 0;
					const activeHead = headCount > 0 ? step % headCount : 0;
					const h = obj.userData.head as number;
					const isActive = h === activeHead;

					obj.rotation.x = Math.PI / 2;
					obj.rotation.z += isActive ? 0.02 : 0.003;

					const mat = obj.material;
					if (mat && !Array.isArray(mat) && "opacity" in mat) {
						// Active: visible + colored, inactive: nearly invisible + grey
						const targetOpacity = isActive ? 0.7 : 0.1;
						mat.opacity += (targetOpacity - mat.opacity) * 0.1;
						if ("color" in mat) {
							if (isActive) {
								mat.color.setHSL(0.12, 0.85, 0.5);
							} else {
								mat.color.setHSL(0, 0, 0.25);
							}
						}
					}
				}

				// Animate base plates (subtle pulse for active heads)
				if (obj.userData.isBasePlate && activeViewRef.current === "attention") {
					const headCount = Math.max(1, Math.floor(headCountRef.current || 1));
					const step = telemetryRef.current.step || 0;
					const activeHead = headCount > 0 ? step % headCount : 0;
					const h = obj.userData.head as number;
					const isActive = h === activeHead;

					const mat = obj.material;
					if (mat && !Array.isArray(mat) && "opacity" in mat) {
						const targetOpacity = isActive ? 0.4 : 0.15;
						mat.opacity += (targetOpacity - mat.opacity) * 0.1;
						if ("color" in mat) {
							if (isActive) {
								mat.color.setHSL(0.55, 0.3, 0.15);
							} else {
								mat.color.set(0x080810);
							}
						}
					}
				}

				// Smoothly animate activation-flow bars (layer/token/dim) from live viz.
				if (obj.userData.isActBar && activeViewRef.current === "activations") {
					const layer = obj.userData.layer as number;
					const token = obj.userData.token as number;
					const dim = obj.userData.dim as number;
					const sample = vizByIndexRef.current.get(layer)?.act;
					const v =
						sample?.[token]?.[dim] ??
						(Math.max(
							0,
							Math.min(1, Number(obj.userData.lastValue ?? 0)),
						) as number);
					obj.userData.lastValue = v;
					const value = Math.abs(v);
					const target = value * 0.4 + 0.02;
					obj.userData.targetHeight = target;

					const cur = typeof obj.scale.y === "number" ? obj.scale.y : 0.02;
					const next = cur + (target - cur) * 0.15;
					obj.scale.y = next;

					const mat = obj.material;
					if (mat && !Array.isArray(mat) && "color" in mat) {
						try {
							mat.color.setHSL(0.55 - value * 0.4, 0.8, 0.4 + value * 0.3);
						} catch {
							// ignore
						}
					}
				}

				// Animate block nodes: active = colored + slight transparency + glow, inactive = greyscale + more transparent
				if (obj.userData.blockIndex !== undefined) {
					const i = obj.userData.blockIndex;
					const isActive = i === activeBlock;
					const mat = obj.material;
					const baseColor = obj.userData.baseColor;

					// Get strength value for glow intensity (RMS from layer stats)
					const rmsRaw = obj.userData.layerMetricRms;
					const strength =
						typeof rmsRaw === "number" && Number.isFinite(rmsRaw)
							? Math.min(1, Math.log10(1 + rmsRaw) * 0.5)
							: 0.3; // default strength if no data

					if (mat && !Array.isArray(mat)) {
						try {
							if ("color" in mat && mat.color && baseColor) {
								if (isActive) {
									// Active: full color
									const targetColor = new THREE.Color(baseColor);
									mat.color.lerp(targetColor, 0.2);
								} else {
									// Inactive: complete greyscale (desaturated)
									// Convert base color to greyscale
									const baseC = new THREE.Color(baseColor);
									const grey =
										baseC.r * 0.299 + baseC.g * 0.587 + baseC.b * 0.114;
									const greyColor = new THREE.Color(grey, grey, grey);
									mat.color.lerp(greyColor, 0.15);
								}
							}

							// Emissive glow - active gets glow based on strength, inactive gets none
							if ("emissive" in mat && mat.emissive && baseColor) {
								if (isActive) {
									const emissiveColor = new THREE.Color(baseColor);
									mat.emissive.lerp(emissiveColor, 0.2);
								} else {
									mat.emissive.lerp(new THREE.Color(0x000000), 0.15);
								}
							}

							if ("emissiveIntensity" in mat) {
								// Active: glow intensity based on strength value
								// Inactive: no glow
								const targetIntensity = isActive
									? 0.3 + strength * 0.7 + pulse * 0.5
									: 0;
								mat.emissiveIntensity +=
									(targetIntensity - mat.emissiveIntensity) * 0.15;
							}

							if ("opacity" in mat) {
								// Active: slightly transparent (0.92), Inactive: more transparent (0.45)
								const targetOpacity = isActive ? 0.92 : 0.45;
								mat.opacity += (targetOpacity - mat.opacity) * 0.12;
							}
						} catch {
							// ignore
						}
					}

					// Gentle rotation
					obj.rotation.y += isActive ? 0.015 : 0.003;
					obj.rotation.x =
						Math.sin(timeRef.current * 0.5 + i * 0.5) *
						(isActive ? 0.08 : 0.02);

					// Subtle floating animation
					const floatAmplitude = isActive ? 0.15 : 0.03;
					obj.position.y =
						Math.sin(timeRef.current * 1.0 + i * 0.4) * floatAmplitude;

					// Keep scale consistent (no shrinking of inactive)
					obj.scale.setScalar(1.0);
				}

				// Glow rings beneath blocks - active = colored, inactive = greyscale
				if (obj.userData.isGlowRing) {
					const i = obj.userData.blockIndex;
					const isActive = i === activeBlock;
					const mat = obj.material;

					if (mat && !Array.isArray(mat)) {
						if ("opacity" in mat) {
							// Active: visible, inactive: barely visible
							const targetOpacity = isActive ? 0.7 + pulse * 0.2 : 0.08;
							mat.opacity += (targetOpacity - mat.opacity) * 0.12;
						}
						try {
							if ("color" in mat && mat.color?.setHSL) {
								if (isActive) {
									// Colored for active
									mat.color.setHSL(0.35 - heat * 0.3, 0.9, 0.5);
								} else {
									// Greyscale for inactive
									mat.color.setHSL(0, 0, 0.2);
								}
							}
						} catch {
							// ignore
						}
					}
				}
				// Animate particles along curves
				if (obj.userData.isParticle && obj.userData.curve) {
					const baseSpeed = obj.userData.baseSpeed ?? obj.userData.speed ?? 0.3;
					obj.userData.speed =
						baseSpeed * (0.6 + 0.4 * tokFactor) * (1 + pulse * 0.2);
					const t =
						(timeRef.current * obj.userData.speed + obj.userData.offset) % 1;
					const point = obj.userData.curve.getPoint(t);
					obj.position.copy(point);

					// Particle color/opacity based on connection activity
					const connectionIdx = obj.userData.connectionIndex ?? 0;
					const isActiveConnection =
						connectionIdx === activeBlock || connectionIdx === activeBlock - 1;

					if (obj.material && !Array.isArray(obj.material)) {
						if ("opacity" in obj.material) {
							// Active connections: bright, inactive: dim
							const baseOpacity = isActiveConnection ? 0.9 : 0.25;
							obj.material.opacity = Math.sin(t * Math.PI) * baseOpacity;
						}
						if ("color" in obj.material) {
							if (isActiveConnection) {
								// Colored for active
								obj.material.color.set(0x60a5fa);
							} else {
								// Greyscale for inactive
								obj.material.color.set(0x555555);
							}
						}
					}
				}
				// Animate attention heads
				if (obj.userData.isHead) {
					obj.rotation.y += 0.02 + pulse * 0.04;
					obj.rotation.x =
						Math.sin(timeRef.current * 2 + obj.userData.headIndex) * 0.1;
					obj.scale.setScalar(1 + pulse * 0.15);
					const mat = obj.material;
					if (mat && !Array.isArray(mat) && "emissiveIntensity" in mat) {
						const base =
							typeof obj.userData.baseEmissive === "number"
								? obj.userData.baseEmissive
								: 0.3;
						mat.emissiveIntensity = base + pulse * 0.8 + (1 - heat) * 0.1;
					}
				}
				// Animate indicators
				if (obj.userData.isIndicator) {
					obj.rotation.y += 0.03;
					obj.scale.setScalar(1 + pulse * 0.25);
				}
			});

			const composer = composerRef.current;
			const scene = sceneRef.current;
			const camera = cameraRef.current;
			if (composer && scene && camera) {
				composer.render();
			}
		};

		animate();
		return () => {
			if (frameRef.current !== null) {
				cancelAnimationFrame(frameRef.current);
			}
		};
	}, [isRotating, modelConfig]);

	const handleBackToModel = () => {
		setZoomLevel("model");
		setSelectedLayer(null);
		// Reset camera target to center
		if (controlsRef.current) {
			controlsRef.current.target.set(0, 0, 0);
		}
	};

	return (
		<div className="w-full h-screen bg-slate-900 flex flex-col overflow-hidden">
			{/* Header */}
			<div className="bg-slate-800 border-b border-slate-700 px-4 py-3 shrink-0">
				<div className="flex items-center justify-between">
					<div className="flex items-center gap-4">
						<div>
							<h1 className="text-lg font-bold text-white">
								{modelConfig.name}
							</h1>
							<p className="text-slate-400 text-xs">
								{modelConfig.num_layers} layers • {modelConfig.hidden_dim}{" "}
								hidden • {modelConfig.num_heads} heads
							</p>
						</div>

						{/* Breadcrumb */}
						<div className="flex items-center gap-2 text-sm">
							<button
								type="button"
								onClick={handleBackToModel}
								className={`px-2 py-1 rounded transition-colors ${
									zoomLevel === "model"
										? "bg-blue-600 text-white"
										: "text-slate-400 hover:text-white hover:bg-slate-700"
								}`}
							>
								Model
							</button>
							{selectedLayer && "name" in selectedLayer && (
								<>
									<span className="text-slate-600">›</span>
									<span className="px-2 py-1 bg-blue-600 text-white rounded">
										{selectedLayer.name}
									</span>
								</>
							)}
						</div>
					</div>

					<div className="flex items-center gap-3">
						<label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
							<input
								type="checkbox"
								checked={showActivations}
								onChange={(e) => setShowActivations(e.target.checked)}
								className="rounded"
							/>
							Activations
						</label>
						<button
							type="button"
							onClick={() => setIsRotating(!isRotating)}
							className={`px-3 py-1.5 rounded text-sm transition-colors ${
								isRotating
									? "bg-blue-600 text-white"
									: "bg-slate-700 text-slate-300 hover:bg-slate-600"
							}`}
						>
							{isRotating ? "⟳ Rotating" : "⟳ Paused"}
						</button>
					</div>
				</div>
			</div>

			<div className="flex flex-1 min-h-0">
				{/* Sidebar */}
				<div className="w-52 bg-slate-800 border-r border-slate-700 p-3 flex flex-col gap-3 shrink-0">
					<div>
						<h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
							View
						</h2>
						<div className="flex flex-col gap-1">
							{[
								{ id: "graph", label: "Architecture", icon: "◇" },
								{ id: "attention", label: "Attention", icon: "▦" },
								{ id: "activations", label: "Activations", icon: "≋" },
							].map((view) => (
								<button
									type="button"
									key={view.id}
									onClick={() => {
										setActiveView(view.id);
										setZoomLevel("model");
										setSelectedLayer(null);
										// Reset camera target to center
										if (controlsRef.current) {
											controlsRef.current.target.set(0, 0, 0);
										}
									}}
									className={`flex items-center gap-2 px-2 py-1.5 rounded text-left text-sm transition-colors ${
										activeView === view.id
											? "bg-blue-600 text-white"
											: "text-slate-300 hover:bg-slate-700"
									}`}
								>
									<span>{view.icon}</span>
									<span>{view.label}</span>
								</button>
							))}
						</div>
					</div>

					{activeView === "graph" && (
						<div>
							<h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
								Token
							</h2>
							<input
								type="range"
								min="0"
								max="15"
								value={inputToken}
								onChange={(e) => setInputToken(parseInt(e.target.value, 10))}
								className="w-full accent-blue-500"
							/>
							<p className="text-xs text-slate-400 mt-1">
								Position {inputToken}
							</p>
						</div>
					)}

					{selectedLayer && "name" in selectedLayer && (
						<div className="border-t border-slate-700 pt-3">
							<h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
								Layer Info
							</h2>
							<div className="text-sm text-slate-300 space-y-1">
								<p className="font-medium text-white">{selectedLayer.name}</p>
								{"type" in selectedLayer && (
									<p className="text-slate-400 text-xs">
										Type: {selectedLayer.type}
									</p>
								)}
								{"layers" in selectedLayer && selectedLayer.layers && (
									<p className="text-slate-400 text-xs">
										Components: {selectedLayer.layers.length}
									</p>
								)}
								{"blockNum" in selectedLayer &&
									typeof selectedLayer.blockNum === "number" &&
									layerStatByIndex.get(selectedLayer.blockNum) && (
										<div className="pt-2 mt-2 border-t border-slate-700 space-y-1">
											<p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
												Activations (summary)
											</p>
											{(() => {
												const s = layerStatByIndex.get(selectedLayer.blockNum);
												if (!s) return null;
												return (
													<>
														<p className="text-slate-400 text-xs">
															rms:{" "}
															<span className="text-slate-200">
																{s.rms.toFixed(4)}
															</span>
														</p>
														<p className="text-slate-400 text-xs">
															mean_abs:{" "}
															<span className="text-slate-200">
																{s.mean_abs.toFixed(4)}
															</span>
															{" • "}
															max_abs:{" "}
															<span className="text-slate-200">
																{s.max_abs.toFixed(4)}
															</span>
														</p>
													</>
												);
											})()}
										</div>
									)}
								{"blockNum" in selectedLayer &&
									typeof selectedLayer.blockNum === "number" &&
									attentionLayers?.[selectedLayer.blockNum] && (
										<div className="pt-2 mt-2 border-t border-slate-700 space-y-1">
											<p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
												Attention
											</p>
											{(() => {
												const a = attentionLayers[selectedLayer.blockNum];
												return (
													<>
														<p className="text-slate-400 text-xs">
															Mode:{" "}
															<span className="text-slate-200">{a.mode}</span>
														</p>
														<p className="text-slate-400 text-xs">
															sem_dim / geo_dim:{" "}
															<span className="text-slate-200">
																{a.sem_dim ?? "—"} / {a.geo_dim ?? "—"}
															</span>
														</p>
														<p className="text-slate-400 text-xs">
															null_attn:{" "}
															<span className="text-slate-200">
																{a.null_attn ? "on" : "off"}
															</span>
															{" • "}
															tie_qk:{" "}
															<span className="text-slate-200">
																{a.tie_qk ? "on" : "off"}
															</span>
														</p>
														<p className="text-slate-400 text-xs">
															rope_semantic:{" "}
															<span className="text-slate-200">
																{a.rope_semantic ? "on" : "off"}
															</span>
															{" • "}
															gate:{" "}
															<span className="text-slate-200">
																{a.decoupled_gate ? "on" : "off"}
															</span>
														</p>
													</>
												);
											})()}
										</div>
									)}
							</div>
						</div>
					)}

					<div className="mt-auto border-t border-slate-700 pt-3">
						<h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
							Legend
						</h2>
						<div className="space-y-1 text-xs">
							{[
								["Embedding", colors.embedding],
								["Attention", colors.attention],
								["FFN", colors.ffn],
								["LayerNorm", colors.layernorm],
								["Linear", colors.linear],
							].map(([name, color]) => (
								<div key={name} className="flex items-center gap-2">
									<div
										className="w-3 h-3 rounded"
										style={{ backgroundColor: color }}
									/>
									<span className="text-slate-400">{name}</span>
								</div>
							))}
							<div className="pt-2 mt-2 border-t border-slate-700 space-y-1">
								<div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide">
									DBA markers
								</div>
								<div className="flex items-center gap-2">
									<div
										className="w-3 h-3 rounded"
										style={{ backgroundColor: "#f472b6" }}
									/>
									<span className="text-slate-400">Semantic plate</span>
								</div>
								<div className="flex items-center gap-2">
									<div
										className="w-3 h-3 rounded"
										style={{ backgroundColor: "#4ade80" }}
									/>
									<span className="text-slate-400">Geometric plate</span>
								</div>
								<div className="flex items-center gap-2">
									<div
										className="w-3 h-3 rounded-full"
										style={{ backgroundColor: "#e0af68" }}
									/>
									<span className="text-slate-400">Null sink (null_attn)</span>
								</div>
								<div className="flex items-center gap-2">
									<div
										className="w-3 h-3 rounded-full"
										style={{ backgroundColor: "#60a5fa" }}
									/>
									<span className="text-slate-400">tie_qk ring</span>
								</div>
								<div className="flex items-center gap-2">
									<div
										className="w-3 h-3 rounded-full"
										style={{ backgroundColor: "#bb9af7" }}
									/>
									<span className="text-slate-400">RoPE on semantic</span>
								</div>
							</div>
						</div>
					</div>
				</div>

				{/* Canvas */}
				<div className="flex-1 relative min-w-0">
					<canvas
						ref={canvasRef}
						className="w-full h-full cursor-grab active:cursor-grabbing"
					/>

					<div className="absolute bottom-4 left-4 bg-slate-800/80 backdrop-blur rounded px-3 py-2 text-xs text-slate-400">
						Drag to rotate • Scroll to zoom • Click blocks to inspect
					</div>
				</div>
			</div>
		</div>
	);
};

export default RealisticMLVisualization;

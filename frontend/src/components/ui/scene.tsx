import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";

import Stats from "three/addons/libs/stats.module.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { BloomPass } from "three/addons/postprocessing/BloomPass.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { FilmPass } from "three/addons/postprocessing/FilmPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { FocusShader } from "three/addons/shaders/FocusShader.js";
import { Flex } from "./flex";

type CloneMesh = {
	mesh: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>;
	speed: number;
};

type AnimatedMesh = {
	mesh: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>;
	verticesDown: number;
	verticesUp: number;
	direction: number;
	speed: number;
	delay: number;
	start: number;
};

export const Scene = () => {
	const containerRef = useRef<HTMLDivElement>(null);
	const cameraRef = useRef<THREE.PerspectiveCamera>(null);
	const sceneRef = useRef<THREE.Scene>(null);
	const rendererRef = useRef<THREE.WebGLRenderer>(null);
	const clockRef = useRef<THREE.Clock>(null);
	const composerRef = useRef<EffectComposer>(null);
	const statsRef = useRef<Stats>(null);
	const effectFocusRef = useRef<ShaderPass>(null);
	const meshesRef = useRef<AnimatedMesh[]>([]);
	const clonesRef = useRef<CloneMesh[]>([]);
	const parentRef = useRef<THREE.Object3D>(null);

	const combineBuffer = useCallback(
		(model: THREE.Object3D, bufferName: string) => {
			let count = 0;

			model.traverse((child) => {
				if (child instanceof THREE.Mesh) {
					const buffer = child.geometry.getAttribute(bufferName);
					count += buffer.array.length;
				}
			});

			const combined = new Float32Array(count);

			let offset = 0;

			model.traverse((child) => {
				if (child instanceof THREE.Mesh) {
					const buffer = child.geometry.getAttribute(bufferName);

					combined.set(buffer.array, offset);
					offset += buffer.array.length;
				}
			});

			return new THREE.BufferAttribute(combined, 3);
		},
		[],
	);

	const createMesh = useCallback(
		(
			positions: THREE.BufferAttribute,
			scale: number,
			x: number,
			y: number,
			z: number,
			color: number,
		) => {
			const geometry = new THREE.BufferGeometry();
			geometry.setAttribute("position", positions.clone());
			geometry.setAttribute("initialPosition", positions.clone());

			(geometry.attributes.position as THREE.BufferAttribute).setUsage(
				THREE.DynamicDrawUsage,
			);

			const clones = [
				[6000, 0, -4000],
				[5000, 0, 0],
				[1000, 0, 5000],
				[1000, 0, -5000],
				[4000, 0, 2000],
				[-4000, 0, 1000],
				[-5000, 0, -5000],

				[0, 0, 0],
			];

			let mainMesh: THREE.Points<
				THREE.BufferGeometry,
				THREE.PointsMaterial
			> | null = null;

			for (let i = 0; i < clones.length; i++) {
				const c = i < clones.length - 1 ? 0x252525 : color;

				const mesh = new THREE.Points<
					THREE.BufferGeometry,
					THREE.PointsMaterial
				>(geometry, new THREE.PointsMaterial({ size: 30, color: c }));
				mesh.scale.x = mesh.scale.y = mesh.scale.z = scale;

				mesh.position.x = x + clones[i][0];
				mesh.position.y = y + clones[i][1];
				mesh.position.z = z + clones[i][2];

				parentRef.current?.add(mesh);

				clonesRef.current.push({ mesh, speed: 0.5 + Math.random() });

				if (i === clones.length - 1) {
					mainMesh = mesh;
				}
			}

			if (mainMesh) {
				meshesRef.current.push({
					mesh: mainMesh,
					verticesDown: 0,
					verticesUp: 0,
					direction: 0,
					speed: 15,
					delay: Math.floor(200 + 200 * Math.random()),
					start: Math.floor(100 + 200 * Math.random()),
				});
			}
		},
		[],
	);

	const onWindowResize = useCallback(() => {
		const camera = cameraRef.current;
		const scene = sceneRef.current;
		const renderer = rendererRef.current;
		const composer = composerRef.current;
		const effectFocus = effectFocusRef.current;

		if (!camera || !scene || !renderer || !composer || !effectFocus) return;

		camera.aspect = window.innerWidth / window.innerHeight;
		camera.updateProjectionMatrix();
		camera.lookAt(scene.position);

		renderer.setSize(window.innerWidth, window.innerHeight);
		composer.setSize(window.innerWidth, window.innerHeight);

		effectFocus.uniforms["screenWidth"].value =
			window.innerWidth * window.devicePixelRatio;
		effectFocus.uniforms["screenHeight"].value =
			window.innerHeight * window.devicePixelRatio;
	}, []);

	const render = useCallback(() => {
		const deltaRaw = clockRef.current?.getDelta() ?? 0;
		let delta = 10 * deltaRaw;

		delta = delta < 2 ? delta : 2;

		const parent = parentRef.current;
		if (!parent) return;
		parent.rotation.y += -0.02 * delta;

		for (let j = 0; j < clonesRef.current.length; j++) {
			const cm = clonesRef.current[j];
			cm.mesh.rotation.y += -0.1 * delta * cm.speed;
		}

		for (let j = 0; j < meshesRef.current.length; j++) {
			const data = meshesRef.current[j];
			const positions = data.mesh.geometry.getAttribute(
				"position",
			) as THREE.BufferAttribute;
			const initialPositions = data.mesh.geometry.getAttribute(
				"initialPosition",
			) as THREE.BufferAttribute;

			const count = positions.count;

			if (data.start > 0) {
				data.start -= 1;
			} else {
				if (data.direction === 0) {
					data.direction = -1;
				}
			}

			for (let i = 0; i < count; i++) {
				const px = positions.getX(i);
				const py = positions.getY(i);
				const pz = positions.getZ(i);

				// falling down
				if (data.direction < 0) {
					if (py > 0) {
						positions.setXYZ(
							i,
							px + 1.5 * (0.5 - Math.random()) * data.speed * delta,
							py + 3.0 * (0.25 - Math.random()) * data.speed * delta,
							pz + 1.5 * (0.5 - Math.random()) * data.speed * delta,
						);
					} else {
						data.verticesDown += 1;
					}
				}

				// rising up
				if (data.direction > 0) {
					const ix = initialPositions.getX(i);
					const iy = initialPositions.getY(i);
					const iz = initialPositions.getZ(i);

					const dx = Math.abs(px - ix);
					const dy = Math.abs(py - iy);
					const dz = Math.abs(pz - iz);

					const d = dx + dy + dz;

					if (d > 1) {
						// Guard against division by zero
						const dirX = dx > 0 ? (px - ix) / dx : 0;
						const dirY = dy > 0 ? (py - iy) / dy : 0;
						const dirZ = dz > 0 ? (pz - iz) / dz : 0;

						positions.setXYZ(
							i,
							px - dirX * data.speed * delta * (0.85 - Math.random()),
							py - dirY * data.speed * delta * (1 + Math.random()),
							pz - dirZ * data.speed * delta * (0.85 - Math.random()),
						);
					} else {
						data.verticesUp += 1;
					}
				}
			}

			// all vertices down
			if (data.verticesDown >= count) {
				if (data.delay <= 0) {
					data.direction = 1;
					data.speed = 5;
					data.verticesDown = 0;
					data.delay = 320;
				} else {
					data.delay -= 1;
				}
			}

			// all vertices up
			if (data.verticesUp >= count) {
				if (data.delay <= 0) {
					data.direction = -1;
					data.speed = 15;
					data.verticesUp = 0;
					data.delay = 120;
				} else {
					data.delay -= 1;
				}
			}

			positions.needsUpdate = true;
		}

		composerRef.current?.render(0.01);
	}, []);

	const animate = useCallback(() => {
		render();
		statsRef.current?.update();
	}, [render]);

	useEffect(() => {
		if (!containerRef.current) return;
		cameraRef.current = new THREE.PerspectiveCamera(
			20,
			window.innerWidth / window.innerHeight,
			1,
			50000,
		);

		cameraRef.current.position.set(0, 700, 7000);

		sceneRef.current = new THREE.Scene();
		sceneRef.current.background = new THREE.Color(0x000104);
		sceneRef.current.fog = new THREE.FogExp2(0x000104, 0.0000675);

		cameraRef.current.lookAt(sceneRef.current.position);

		clockRef.current = new THREE.Clock();

		const loader = new OBJLoader();

		loader.load(
			"/male02/male02.obj",
			(object) => {
				const positions = combineBuffer(object, "position");

				createMesh(positions, 4.05, -500, -350, 600, 0xff7744);
				createMesh(positions, 4.05, 500, -350, 0, 0xff5522);
				createMesh(positions, 4.05, -250, -350, 1500, 0xff9922);
				createMesh(positions, 4.05, -250, -350, -1500, 0xff99ff);
			},
			undefined,
			(error) => {
				console.error("Failed to load male02.obj:", error);
			},
		);

		loader.load(
			"/female02/female02.obj",
			(object) => {
				const positions = combineBuffer(object, "position");

				createMesh(positions, 4.05, -1000, -350, 0, 0xffdd44);
				createMesh(positions, 4.05, 0, -350, 0, 0xffffff);
				createMesh(positions, 4.05, 1000, -350, 400, 0xff4422);
				createMesh(positions, 4.05, 250, -350, 1500, 0xff9955);
				createMesh(positions, 4.05, 250, -350, 2500, 0xff77dd);
			},
			undefined,
			(error) => {
				console.error("Failed to load female02.obj:", error);
			},
		);

		rendererRef.current = new THREE.WebGLRenderer();
		rendererRef.current.setPixelRatio(window.devicePixelRatio);
		rendererRef.current.setSize(window.innerWidth, window.innerHeight);
		rendererRef.current.setAnimationLoop(animate);
		rendererRef.current.autoClear = false;
		containerRef.current?.appendChild(rendererRef.current.domElement);

		parentRef.current = new THREE.Object3D();
		sceneRef.current.add(parentRef.current);

		const grid = new THREE.Points(
			new THREE.PlaneGeometry(15000, 15000, 64, 64),
			new THREE.PointsMaterial({ color: 0xff0000, size: 10 }),
		);
		grid.position.y = -400;
		grid.rotation.x = -Math.PI / 2;
		parentRef.current.add(grid);

		// postprocessing

		const renderModel = new RenderPass(sceneRef.current, cameraRef.current);
		const effectBloom = new BloomPass(0.75);
		const effectFilm = new FilmPass();

		effectFocusRef.current = new ShaderPass(FocusShader);

		effectFocusRef.current.uniforms["screenWidth"].value =
			window.innerWidth * window.devicePixelRatio;
		effectFocusRef.current.uniforms["screenHeight"].value =
			window.innerHeight * window.devicePixelRatio;

		const outputPass = new OutputPass();

		composerRef.current = new EffectComposer(rendererRef.current);

		composerRef.current.addPass(renderModel);
		composerRef.current.addPass(effectBloom);
		composerRef.current.addPass(effectFilm);
		composerRef.current.addPass(effectFocusRef.current);
		composerRef.current.addPass(outputPass);

		//stats (development only)
		if (import.meta.env.DEV) {
			statsRef.current = new Stats();
			containerRef.current?.appendChild(statsRef.current.dom);
		}

		window.addEventListener("resize", onWindowResize);
		return () => {
			window.removeEventListener("resize", onWindowResize);
			rendererRef.current?.setAnimationLoop(null);
			rendererRef.current?.dispose();
		};
	}, [animate, combineBuffer, createMesh, onWindowResize]);

	return (
		<Flex
			ref={containerRef}
			direction="column"
			align="center"
			justify="center"
			fullWidth
			fullHeight
		></Flex>
	);
};

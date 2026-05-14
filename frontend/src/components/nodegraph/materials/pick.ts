import * as THREE from "three";
import { MAX_NODES, NODE_H, NODE_W, nodesTexture } from "../vector";

/**
 * PickLayer — node bodies rendered to an off-screen target with their
 * instance id encoded into RGB. Sampling a single pixel under the cursor
 * gives O(1) hit-testing regardless of node count.
 */
export class PickLayer {
	readonly scene = new THREE.Scene();
	readonly mesh: THREE.Mesh;
	private mat: THREE.RawShaderMaterial;
	private target: THREE.WebGLRenderTarget;
	private pixel = new Uint8Array(4);

	constructor(nodeGeom: THREE.InstancedBufferGeometry) {
		this.target = new THREE.WebGLRenderTarget(1, 1, {
			minFilter: THREE.NearestFilter,
			magFilter: THREE.NearestFilter,
			format: THREE.RGBAFormat,
			type: THREE.UnsignedByteType,
		});

		this.mat = new THREE.RawShaderMaterial({
			glslVersion: THREE.GLSL3,
			uniforms: {
				uNodes: { value: nodesTexture },
				uMax: { value: MAX_NODES },
				uSize: { value: new THREE.Vector2(NODE_W, NODE_H) },
				uProj: { value: new THREE.Matrix4() },
				uView: { value: new THREE.Matrix4() },
			},
			vertexShader: /* glsl */ `
				in vec3 position;
				uniform sampler2D uNodes;
				uniform float uMax;
				uniform vec2 uSize;
				uniform mat4 uProj;
				uniform mat4 uView;
				flat out int vId;
				flat out float vAlive;
				void main() {
					float u = (float(gl_InstanceID) + 0.5) / uMax;
					vec4 n = texture(uNodes, vec2(u, 0.5));
					vAlive = n.a;
					vec2 world = n.xy + position.xy * uSize;
					gl_Position = uProj * uView * vec4(world, 0.0, 1.0);
					vId = gl_InstanceID + 1;
				}
			`,
			fragmentShader: /* glsl */ `
				precision highp float;
				flat in int vId;
				flat in float vAlive;
				out vec4 outColor;
				void main() {
					if (vAlive < 0.5) discard;
					int v = vId;
					outColor = vec4(
						float(v & 0xff) / 255.0,
						float((v >> 8) & 0xff) / 255.0,
						float((v >> 16) & 0xff) / 255.0,
						1.0
					);
				}
			`,
		});

		this.mesh = new THREE.Mesh(nodeGeom, this.mat);
		this.mesh.frustumCulled = false;
		this.scene.add(this.mesh);
	}

	setView(cam: THREE.OrthographicCamera): void {
		this.mat.uniforms.uProj.value = cam.projectionMatrix;
		this.mat.uniforms.uView.value = cam.matrixWorldInverse;
	}

	pick(
		renderer: THREE.WebGLRenderer,
		camera: THREE.OrthographicCamera,
		sx: number,
		sy: number,
		clearColor: number,
	): number {
		const rect = renderer.domElement.getBoundingClientRect();
		const dpr = renderer.getPixelRatio();
		const w = Math.max(1, Math.floor(rect.width * dpr));
		const h = Math.max(1, Math.floor(rect.height * dpr));

		if (this.target.width !== w || this.target.height !== h) {
			this.target.setSize(w, h);
		}

		const px = Math.floor((sx - rect.left) * dpr);
		const py = Math.floor((rect.height - (sy - rect.top)) * dpr);

		if (px < 0 || py < 0 || px >= w || py >= h) return -1;

		renderer.setRenderTarget(this.target);
		renderer.setClearColor(0x000000, 1);
		renderer.clear();
		renderer.render(this.scene, camera);
		renderer.readRenderTargetPixels(this.target, px, py, 1, 1, this.pixel);
		renderer.setRenderTarget(null);
		renderer.setClearColor(clearColor, 1);

		const id = this.pixel[0] | (this.pixel[1] << 8) | (this.pixel[2] << 16);
		return id - 1;
	}

	dispose(): void {
		this.mat.dispose();
		this.target.dispose();
	}
}

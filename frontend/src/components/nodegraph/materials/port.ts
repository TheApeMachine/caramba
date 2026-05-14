import * as THREE from "three";
import {
	INPUT_OFFSET_X,
	INPUT_OFFSET_Y,
	MAX_NODES,
	nodesTexture,
	OUTPUT_OFFSET_X,
	OUTPUT_OFFSET_Y,
	PORT_RADIUS,
} from "../vector";

const PORTS_PER_NODE = 2;

/**
 * PortLayer — one instanced quad per (node, kind). The geometry is 3× the
 * port radius to accommodate the halo glow; the fragment shader composes
 * a disc, rim, and halo from a single distance.
 */
export class PortLayer {
	readonly mesh: THREE.Mesh;
	private mat: THREE.RawShaderMaterial;
	private geom: THREE.InstancedBufferGeometry;

	constructor() {
		const quad = new THREE.PlaneGeometry(1, 1);
		this.geom = new THREE.InstancedBufferGeometry();
		this.geom.setAttribute("position", quad.getAttribute("position"));
		this.geom.setAttribute("uv", quad.getAttribute("uv"));
		this.geom.setIndex(quad.getIndex());
		this.geom.instanceCount = MAX_NODES * PORTS_PER_NODE;

		this.mat = new THREE.RawShaderMaterial({
			glslVersion: THREE.GLSL3,
			transparent: true,
			depthWrite: false,
			uniforms: {
				uNodes: { value: nodesTexture },
				uMax: { value: MAX_NODES },
				uPortsPerNode: { value: PORTS_PER_NODE },
				uRadius: { value: PORT_RADIUS },
				uInputOffset: {
					value: new THREE.Vector2(INPUT_OFFSET_X, INPUT_OFFSET_Y),
				},
				uOutputOffset: {
					value: new THREE.Vector2(OUTPUT_OFFSET_X, OUTPUT_OFFSET_Y),
				},
				uProj: { value: new THREE.Matrix4() },
				uView: { value: new THREE.Matrix4() },
			},
			vertexShader: /* glsl */ `
				in vec3 position;
				uniform sampler2D uNodes;
				uniform float uMax;
				uniform float uPortsPerNode;
				uniform float uRadius;
				uniform vec2 uInputOffset;
				uniform vec2 uOutputOffset;
				uniform mat4 uProj;
				uniform mat4 uView;
				out vec2 vUv;
				out float vAlive;
				out float vKind;
				void main() {
					float id = float(gl_InstanceID);
					float nodeIdx = floor(id / uPortsPerNode);
					float kind = mod(id, uPortsPerNode);
					float u = (nodeIdx + 0.5) / uMax;
					vec4 n = texture(uNodes, vec2(u, 0.5));
					vAlive = n.a;
					vUv = position.xy + 0.5;
					vKind = kind;
					vec2 offset = (kind < 0.5) ? uInputOffset : uOutputOffset;
					vec2 world = n.xy + offset + position.xy * (uRadius * 3.0);
					gl_Position = uProj * uView * vec4(world, 0.0, 1.0);
				}
			`,
			fragmentShader: /* glsl */ `
				precision highp float;
				in vec2 vUv;
				in float vAlive;
				in float vKind;
				out vec4 outColor;
				void main() {
					if (vAlive < 0.5) discard;
					vec2 p = vUv - 0.5;
					float d = length(p);
					// Flat half-disc notched into the card's border. Flat edge is
					// at x = 0 (the card border); curved half pokes inward.
					float side = (vKind < 0.5) ? 1.0 : -1.0;
					float xInward = p.x * side;
					if (xInward < 0.0) discard;

					float discR = 0.25;
					float aa = 0.012;
					if (d > discR + aa) discard;

					// Desaturated type fills.
					vec3 fillIn  = vec3(0.38, 0.72, 0.92);
					vec3 fillOut = vec3(0.92, 0.68, 0.36);
					vec3 portCol = (vKind < 0.5) ? fillIn : fillOut;
					vec3 color = portCol;

					// Inner shadow along the curved rim — same falloff vocabulary
					// as the card's border AA so the port reads as part of the
					// same continuous bevelled surface.
					float rimShade = smoothstep(discR - 0.080, discR, d);
					color *= 1.0 - rimShade * 0.45;

					// Occlusion gradient along the flat edge (the side touching
					// the card's rim). Darkens the port pixels nearest the card
					// border so the joint reads as 'cast shadow from the rim'
					// rather than 'two flat surfaces butted together'.
					float flatShade = 1.0 - smoothstep(0.0, 0.060, xInward);
					color *= 1.0 - flatShade * 0.35;

					float discMask = 1.0 - smoothstep(discR - aa, discR + aa, d);
					outColor = vec4(color, discMask);
				}
			`,
		});

		this.mesh = new THREE.Mesh(this.geom, this.mat);
		this.mesh.frustumCulled = false;
	}

	setView(cam: THREE.OrthographicCamera): void {
		this.mat.uniforms.uProj.value = cam.projectionMatrix;
		this.mat.uniforms.uView.value = cam.matrixWorldInverse;
	}

	dispose(): void {
		this.mat.dispose();
		this.geom.dispose();
	}
}

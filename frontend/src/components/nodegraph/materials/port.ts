import * as THREE from "three";
import { PORT_COLORS, PORT_TYPES } from "../types";
import {
	MAX_NODES,
	MAX_PORTS,
	nodesTexture,
	PORT_RADIUS,
	portsTexture,
} from "../vector";

/*
PortLayer — one instanced quad per port slot. Each instance reads its own
texel from portsTexture: (nodeIdx, offsetX, offsetY, packed). It then reads
the parent node's centre from nodesTexture, adds the offset, and renders
a half-disc notched into the card's border.

Color comes from a small uniform palette indexed by the port's type id.
*/
const PORT_PALETTE = new Float32Array(PORT_TYPES.length * 3);
for (let i = 0; i < PORT_TYPES.length; i++) {
	const col = PORT_COLORS[PORT_TYPES[i]];
	PORT_PALETTE[i * 3 + 0] = col[0];
	PORT_PALETTE[i * 3 + 1] = col[1];
	PORT_PALETTE[i * 3 + 2] = col[2];
}

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
		this.geom.instanceCount = MAX_PORTS;

		this.mat = new THREE.RawShaderMaterial({
			glslVersion: THREE.GLSL3,
			transparent: true,
			depthWrite: false,
			uniforms: {
				uNodes: { value: nodesTexture },
				uPorts: { value: portsTexture },
				uMaxNodes: { value: MAX_NODES },
				uMaxPorts: { value: MAX_PORTS },
				uRadius: { value: PORT_RADIUS },
				uPalette: { value: PORT_PALETTE },
				uPaletteSize: { value: PORT_TYPES.length },
				uProj: { value: new THREE.Matrix4() },
				uView: { value: new THREE.Matrix4() },
			},
			vertexShader: /* glsl */ `
				in vec3 position;
				uniform sampler2D uNodes;
				uniform sampler2D uPorts;
				uniform float uMaxNodes;
				uniform float uMaxPorts;
				uniform float uRadius;
				uniform vec3 uPalette[${PORT_TYPES.length}];
				uniform mat4 uProj;
				uniform mat4 uView;
				out vec2 vUv;
				out float vAlive;
				out float vKind;
				flat out vec3 vColor;
				void main() {
					float pu = (float(gl_InstanceID) + 0.5) / uMaxPorts;
					vec4 port = texture(uPorts, vec2(pu, 0.5));
					float nodeIdx = port.r;
					vAlive = (nodeIdx < 0.0) ? 0.0 : 1.0;
					if (vAlive < 0.5) {
						gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
						return;
					}
					vec2 offset = vec2(port.g, port.b);
					float packed = port.a;
					float kind = mod(packed, 8.0);
					float typeId = floor(packed / 8.0);
					vKind = kind;
					vColor = uPalette[int(typeId)];

					float nu = (nodeIdx + 0.5) / uMaxNodes;
					vec4 n = texture(uNodes, vec2(nu, 0.5));
					if (n.a < 0.5) {
						vAlive = 0.0;
						gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
						return;
					}
					vUv = position.xy + 0.5;
					vec2 world = n.xy + offset + position.xy * (uRadius * 3.0);
					gl_Position = uProj * uView * vec4(world, 0.0, 1.0);
				}
			`,
			fragmentShader: /* glsl */ `
				precision highp float;
				in vec2 vUv;
				in float vAlive;
				in float vKind;
				flat in vec3 vColor;
				out vec4 outColor;
				void main() {
					if (vAlive < 0.5) discard;
					vec2 p = vUv - 0.5;
					float d = length(p);
					// Flat half-disc notched into the card's border. Flat edge
					// at x = 0 (card border); curved half pokes inward. Side
					// flips based on kind so inputs notch right-ward and
					// outputs notch left-ward.
					float side = (vKind < 0.5) ? 1.0 : -1.0;
					float xInward = p.x * side;
					if (xInward < 0.0) discard;

					float discR = 0.25;
					float aa = 0.012;
					if (d > discR + aa) discard;

					vec3 color = vColor;

					// Inner shadow along the curved rim — same bevel vocabulary
					// as the card's border so the joint reads as a single
					// continuous surface.
					float rimShade = smoothstep(discR - 0.080, discR, d);
					color *= 1.0 - rimShade * 0.45;

					// Occlusion gradient along the flat edge (the side touching
					// the card's rim). Darkens pixels nearest the card border
					// so the joint reads as a cast shadow rather than two
					// butted surfaces.
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

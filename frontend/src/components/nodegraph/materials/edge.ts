import * as THREE from "three";
import { PORT_COLORS, PORT_TYPES } from "../types";
import {
	edgesTexture,
	MAX_EDGES,
	MAX_PORTS,
	PATH_TEXELS_PER_EDGE,
	portsTexture,
} from "../vector";

const SEGMENTS_PER_EDGE = PATH_TEXELS_PER_EDGE - 1;

const EDGE_PALETTE = new Float32Array(PORT_TYPES.length * 3);
for (let i = 0; i < PORT_TYPES.length; i++) {
	const col = PORT_COLORS[PORT_TYPES[i]];
	// Slightly desaturate the port palette for edges so the line reads as
	// "data flowing" rather than "a port slid sideways".
	EDGE_PALETTE[i * 3 + 0] = col[0] * 0.82;
	EDGE_PALETTE[i * 3 + 1] = col[1] * 0.82;
	EDGE_PALETTE[i * 3 + 2] = col[2] * 0.82;
}

/**
 * EdgeLayer — instanced thick-line quads. Each instance is one segment of
 * one edge; the vertex shader samples the two corner texels from a path
 * texture and extrudes a quad perpendicular to the segment.
 */
export class EdgeLayer {
	readonly mesh: THREE.Mesh;
	private mat: THREE.RawShaderMaterial;
	private geom: THREE.InstancedBufferGeometry;

	constructor(pathsTexture: THREE.Texture) {
		this.geom = new THREE.InstancedBufferGeometry();
	
		const verts = new Float32Array([
			0, -1, 0, 1, -1, 0, 1, 1, 0, 0, -1, 0, 1, 1, 0, 0, 1, 0,
		]);
	
		this.geom.setAttribute("position", new THREE.BufferAttribute(verts, 3));
		const segIdx = new Float32Array(MAX_EDGES * SEGMENTS_PER_EDGE * 2);
	
		for (let i = 0; i < MAX_EDGES * SEGMENTS_PER_EDGE; i++) {
			segIdx[i * 2 + 0] = Math.floor(i / SEGMENTS_PER_EDGE);
			segIdx[i * 2 + 1] = i % SEGMENTS_PER_EDGE;
		}
	
		this.geom.setAttribute(
			"iSeg",
			new THREE.InstancedBufferAttribute(segIdx, 2),
		);
	
		this.geom.instanceCount = MAX_EDGES * SEGMENTS_PER_EDGE;

		this.mat = new THREE.RawShaderMaterial({
			glslVersion: THREE.GLSL3,
			transparent: false,
			depthWrite: false,
			uniforms: {
				uEdges: { value: edgesTexture },
				uPaths: { value: pathsTexture },
				uPorts: { value: portsTexture },
				uMaxEdges: { value: MAX_EDGES },
				uMaxPorts: { value: MAX_PORTS },
				uTexelsPerEdge: { value: PATH_TEXELS_PER_EDGE },
				uPalette: { value: EDGE_PALETTE },
				uProj: { value: new THREE.Matrix4() },
				uView: { value: new THREE.Matrix4() },
				// Edge thickness in world units.
				uThickness: { value: 6.0 },
			},
			vertexShader: /* glsl */ `
				in vec3 position;
				in vec2 iSeg;
				uniform sampler2D uEdges;
				uniform sampler2D uPaths;
				uniform sampler2D uPorts;
				uniform float uMaxEdges;
				uniform float uMaxPorts;
				uniform float uTexelsPerEdge;
				uniform vec3 uPalette[${PORT_TYPES.length}];
				uniform mat4 uProj;
				uniform mat4 uView;
				uniform float uThickness;
				out float vAlive;
				flat out vec3 vColor;
				void main() {
					float edgeIdx = iSeg.x;
					float segIdx = iSeg.y;
					float eu = (edgeIdx + 0.5) / uMaxEdges;
					vec4 e = texture(uEdges, vec2(eu, 0.5));
					vAlive = e.b;

					// Edge colour = output port's type colour. e.r holds the
					// from-port (output) index; deref the ports texture to
					// recover the packed (type, kind) byte.
					float fromPortIdx = e.r;
					vec4 fromPort = texture(uPorts,
						vec2((fromPortIdx + 0.5) / uMaxPorts, 0.5));
					float packed = fromPort.a;
					float typeId = floor(packed / 8.0);
					vColor = uPalette[int(typeId)];

					vec2 a = texture(uPaths, vec2(
						(segIdx + 0.5) / uTexelsPerEdge,
						(edgeIdx + 0.5) / uMaxEdges)).xy;
					vec2 b = texture(uPaths, vec2(
						(segIdx + 1.5) / uTexelsPerEdge,
						(edgeIdx + 0.5) / uMaxEdges)).xy;

					vec2 dir = b - a;
					float len = length(dir);
					vec2 segTan = len > 0.0001 ? dir / len : vec2(1.0, 0.0);
					vec2 nrm = vec2(-segTan.y, segTan.x);
					float halfT = uThickness * 0.5;
					vec2 along = mix(a - segTan * halfT, b + segTan * halfT, position.x);
					vec2 p = along + nrm * position.y * halfT;
					gl_Position = uProj * uView * vec4(p, 0.0, 1.0);
				}
			`,
			fragmentShader: /* glsl */ `
				precision highp float;
				in float vAlive;
				flat in vec3 vColor;
				out vec4 outColor;
				void main() {
					if (vAlive < 0.5) discard;
					outColor = vec4(vColor, 1.0);
				}
			`,
		});

		this.mesh = new THREE.Mesh(this.geom, this.mat);
		this.mesh.frustumCulled = false;
		this.mesh.renderOrder = -5;
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

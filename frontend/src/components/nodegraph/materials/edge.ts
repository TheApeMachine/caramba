import * as THREE from "three";
import { edgesTexture, MAX_EDGES, PATH_TEXELS_PER_EDGE } from "../vector";

const SEGMENTS_PER_EDGE = PATH_TEXELS_PER_EDGE - 1;

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
				uMaxEdges: { value: MAX_EDGES },
				uTexelsPerEdge: { value: PATH_TEXELS_PER_EDGE },
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
				uniform float uMaxEdges;
				uniform float uTexelsPerEdge;
				uniform mat4 uProj;
				uniform mat4 uView;
				uniform float uThickness;
				out float vAlive;
				out float vSide;
				void main() {
					float edgeIdx = iSeg.x;
					float segIdx = iSeg.y;
					float eu = (edgeIdx + 0.5) / uMaxEdges;
					vec4 e = texture(uEdges, vec2(eu, 0.5));
					vAlive = e.b;

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
					// Extend each segment by half-thickness past both ends along
					// its tangent. Adjacent perpendicular segments now overlap
					// at corners and the bends read as solid right-angle joints.
					float halfT = uThickness * 0.5;
					vec2 along = mix(a - segTan * halfT, b + segTan * halfT, position.x);
					vec2 p = along + nrm * position.y * halfT;
					gl_Position = uProj * uView * vec4(p, 0.0, 1.0);
					vSide = position.y;
				}
			`,
			fragmentShader: /* glsl */ `
				precision highp float;
				in float vAlive;
				in float vSide;
				out vec4 outColor;
				void main() {
					if (vAlive < 0.5) discard;
					// Muted version of the output port's amber — edges read as
					// "data flowing out of an output port".
					outColor = vec4(vec3(0.55, 0.46, 0.30), 1.0);
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

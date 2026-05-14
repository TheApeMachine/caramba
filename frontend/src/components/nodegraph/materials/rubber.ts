import * as THREE from "three";

/**
 * RubberEdgeLayer — single thick-line segment driven by two world-space
 * uniforms. Drawn while the user is dragging a connection from a port; the
 * mesh stays in the scene but its alpha goes to 0 when uActive = 0.
 */
export class RubberEdgeLayer {
	readonly mesh: THREE.Mesh;
	private mat: THREE.RawShaderMaterial;
	private geom: THREE.BufferGeometry;

	constructor() {
		this.geom = new THREE.BufferGeometry();
		this.geom.setAttribute(
			"position",
			new THREE.BufferAttribute(
				new Float32Array([
					0, -1, 0,
					1, -1, 0,
					1,  1, 0,
					0, -1, 0,
					1,  1, 0,
					0,  1, 0,
				]),
				3,
			),
		);

		this.mat = new THREE.RawShaderMaterial({
			glslVersion: THREE.GLSL3,
			transparent: false,
			depthWrite: false,
			uniforms: {
				uA: { value: new THREE.Vector2() },
				uB: { value: new THREE.Vector2() },
				uProj: { value: new THREE.Matrix4() },
				uView: { value: new THREE.Matrix4() },
				uThickness: { value: 6.0 },
				uActive: { value: 0 },
			},
			vertexShader: /* glsl */ `
				in vec3 position;
				uniform vec2 uA;
				uniform vec2 uB;
				uniform mat4 uProj;
				uniform mat4 uView;
				uniform float uThickness;
				uniform float uActive;
				void main() {
					if (uActive < 0.5) {
						gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
						return;
					}
					vec2 dir = uB - uA;
					float len = length(dir);
					vec2 segTan = len > 0.0001 ? dir / len : vec2(1.0, 0.0);
					vec2 nrm = vec2(-segTan.y, segTan.x);
					float halfT = uThickness * 0.5;
					vec2 along = mix(uA - segTan * halfT, uB + segTan * halfT, position.x);
					vec2 p = along + nrm * position.y * halfT;
					gl_Position = uProj * uView * vec4(p, 0.0, 1.0);
				}
			`,
			fragmentShader: /* glsl */ `
				precision highp float;
				out vec4 outColor;
				void main() {
					outColor = vec4(vec3(0.55, 0.46, 0.30), 1.0);
				}
			`,
		});

		this.mesh = new THREE.Mesh(this.geom, this.mat);
		this.mesh.frustumCulled = false;
		this.mesh.renderOrder = -4;
	}

	setView(cam: THREE.OrthographicCamera): void {
		this.mat.uniforms.uProj.value = cam.projectionMatrix;
		this.mat.uniforms.uView.value = cam.matrixWorldInverse;
	}

	setEndpoints(ax: number, ay: number, bx: number, by: number): void {
		this.mat.uniforms.uA.value.set(ax, ay);
		this.mat.uniforms.uB.value.set(bx, by);
	}

	setActive(active: boolean): void {
		this.mat.uniforms.uActive.value = active ? 1 : 0;
	}

	dispose(): void {
		this.mat.dispose();
		this.geom.dispose();
	}
}

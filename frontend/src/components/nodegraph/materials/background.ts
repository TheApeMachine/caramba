import * as THREE from "three";

/**
 * Fullscreen-quad background. Paints a faint world-space grid that fades
 * with zoom, giving the canvas a paper-on-paper feel. Lives in NDC; its
 * shader reads camera + viewport uniforms.
 */
export class BackgroundLayer {
	readonly mesh: THREE.Mesh;
	private mat: THREE.RawShaderMaterial;

	constructor() {
		this.mat = new THREE.RawShaderMaterial({
			glslVersion: THREE.GLSL3,
			uniforms: {
				uCamX: { value: 0 },
				uCamY: { value: 0 },
				uZoom: { value: 1 },
				uViewport: { value: new THREE.Vector2(1, 1) },
			},
			vertexShader: /* glsl */ `
				in vec3 position;
				out vec2 vNdc;
				void main() {
					vNdc = position.xy;
					gl_Position = vec4(position.xy, 0.999, 1.0);
				}
			`,
			fragmentShader: /* glsl */ `
				precision highp float;
				in vec2 vNdc;
				uniform float uCamX;
				uniform float uCamY;
				uniform float uZoom;
				uniform vec2 uViewport;
				out vec4 outColor;
				// Lines fall on multiples of step (not half-cells), so a node
				// centred on a grid intersection with width = 2N*step has
				// its edges land on grid lines.
				float grid(vec2 p, float step) {
					vec2 g = abs(fract(p / step - 0.5) - 0.5);
					float d = min(g.x, g.y) * step;
					return 1.0 - smoothstep(0.0, 1.5 / uZoom, d);
				}
				void main() {
					vec2 world = vec2(
						uCamX + vNdc.x * (uViewport.x * 0.5 / uZoom),
						uCamY + vNdc.y * (uViewport.y * 0.5 / uZoom)
					);

					// Radial gradient: a hair brighter in the center, deeper at edges.
					// Cool charcoal base with a faint blue cast.
					float r = length(vNdc);
					vec3 center = vec3(0.060, 0.068, 0.082);
					vec3 edge = vec3(0.022, 0.028, 0.038);
					vec3 bg = mix(center, edge, smoothstep(0.0, 1.4, r));

					// Grid fades out at very low and very high zoom so it never moires.
					float fade = smoothstep(0.15, 0.5, uZoom) * (1.0 - smoothstep(4.0, 8.0, uZoom));
					float minor = grid(world, 80.0)  * 0.06;
					float major = grid(world, 320.0) * 0.16;
					bg += vec3(minor + major) * fade;

					outColor = vec4(bg, 1.0);
				}
			`,
			depthWrite: false,
			depthTest: false,
		});

		this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.mat);
		this.mesh.frustumCulled = false;
		this.mesh.renderOrder = -100;
	}

	update(
		camX: number,
		camY: number,
		zoom: number,
		viewportW: number,
		viewportH: number,
	): void {
		const u = this.mat.uniforms;
		u.uCamX.value = camX;
		u.uCamY.value = camY;
		u.uZoom.value = zoom;
		(u.uViewport.value as THREE.Vector2).set(viewportW, viewportH);
	}

	dispose(): void {
		this.mat.dispose();
		(this.mesh.geometry as THREE.BufferGeometry).dispose();
	}
}

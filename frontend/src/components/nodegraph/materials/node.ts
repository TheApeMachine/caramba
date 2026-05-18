import * as THREE from "three";
import {
	MAX_NODES,
	NODE_H,
	NODE_W,
	nestedGraphPreviewCompositeInsets,
	nodesTexture,
} from "../vector";

/**
 * NodeLayer — the node bodies plus a soft drop-shadow underlay. Shares one
 * instanced quad geometry between both meshes. The body shader composites
 * an off-screen preview texture inside the framed node's interior.
 */
export class NodeLayer {
	readonly geom: THREE.InstancedBufferGeometry;
	readonly bodyMesh: THREE.Mesh;
	readonly shadowMesh: THREE.Mesh;
	private bodyMat: THREE.RawShaderMaterial;
	private shadowMat: THREE.RawShaderMaterial;

	constructor(previewTexture: THREE.Texture) {
		const quad = new THREE.PlaneGeometry(1, 1);
		this.geom = new THREE.InstancedBufferGeometry();
		this.geom.setAttribute("position", quad.getAttribute("position"));
		this.geom.setAttribute("uv", quad.getAttribute("uv"));
		this.geom.setIndex(quad.getIndex());
		this.geom.instanceCount = MAX_NODES;

		this.shadowMat = new THREE.RawShaderMaterial({
			glslVersion: THREE.GLSL3,
			transparent: true,
			depthWrite: false,
			blending: THREE.NormalBlending,
			uniforms: {
				uNodes: { value: nodesTexture },
				uMax: { value: MAX_NODES },
				uSize: { value: new THREE.Vector2(NODE_W + 56, NODE_H + 56) },
				uOffset: { value: new THREE.Vector2(4, -14) },
				uProj: { value: new THREE.Matrix4() },
				uView: { value: new THREE.Matrix4() },
			},
			vertexShader: /* glsl */ `
				in vec3 position;
				uniform sampler2D uNodes;
				uniform float uMax;
				uniform vec2 uSize;
				uniform vec2 uOffset;
				uniform mat4 uProj;
				uniform mat4 uView;
				out vec2 vUv;
				out float vAlive;
				void main() {
					float u = (float(gl_InstanceID) + 0.5) / uMax;
					vec4 n = texture(uNodes, vec2(u, 0.5));
					vAlive = n.a;
					vUv = position.xy + 0.5;
					vec2 world = n.xy + uOffset + position.xy * uSize;
					gl_Position = uProj * uView * vec4(world, 0.0, 1.0);
				}
			`,
			fragmentShader: /* glsl */ `
				precision highp float;
				in vec2 vUv;
				in float vAlive;
				out vec4 outColor;
				void main() {
					if (vAlive < 0.5) discard;
					vec2 p = vUv - 0.5;
					// soft elongated falloff, slightly cool-tinted
					float r = length(p * vec2(1.0, 1.15));
					float a = smoothstep(0.5, 0.06, r);
					a = pow(a, 1.6); // softer, more falloff
					vec3 tint = vec3(0.020, 0.030, 0.045);
					outColor = vec4(tint, a * 0.55);
				}
			`,
		});

		this.shadowMesh = new THREE.Mesh(this.geom, this.shadowMat);
		this.shadowMesh.frustumCulled = false;
		this.shadowMesh.renderOrder = -10;

		const previewInsets = nestedGraphPreviewCompositeInsets();

		this.bodyMat = new THREE.RawShaderMaterial({
			glslVersion: THREE.GLSL3,
			transparent: true,
			depthWrite: false,
			uniforms: {
				uNodes: { value: nodesTexture },
				uMax: { value: MAX_NODES },
				uSize: { value: new THREE.Vector2(NODE_W, NODE_H) },
				uProj: { value: new THREE.Matrix4() },
				uView: { value: new THREE.Matrix4() },
				uHovered: { value: -1 },
				uSelected: { value: -1 },
				uPreviewId: { value: -1 },
				uPreviewBlend: { value: 0.0 },
				uPreviewSideInsetUv: { value: previewInsets.horizontalUv },
				uPreviewBandTrimUv: { value: previewInsets.bandTrimUv },
				uPreview: { value: previewTexture },
			},
			vertexShader: /* glsl */ `
				in vec3 position;
				uniform sampler2D uNodes;
				uniform float uMax;
				uniform vec2 uSize;
				uniform mat4 uProj;
				uniform mat4 uView;
				uniform float uHovered;
				uniform float uSelected;
				uniform float uPreviewId;
				out vec2 vUv;
				out float vAlive;
				out float vState;
				flat out float vIsPreview;
				void main() {
					float u = (float(gl_InstanceID) + 0.5) / uMax;
					vec4 n = texture(uNodes, vec2(u, 0.5));
					vAlive = n.a;
					vUv = position.xy + 0.5;
					vec2 world = n.xy + position.xy * uSize;
					gl_Position = uProj * uView * vec4(world, 0.0, 1.0);
					vState = (float(gl_InstanceID) == uHovered) ? 1.0
					       : (float(gl_InstanceID) == uSelected) ? 2.0 : 0.0;
					vIsPreview = float(gl_InstanceID) == uPreviewId ? 1.0 : 0.0;
				}
			`,
			fragmentShader: /* glsl */ `
				precision highp float;
				in vec2 vUv;
				in float vAlive;
				in float vState;
				flat in float vIsPreview;
				uniform sampler2D uPreview;
				uniform float uPreviewBlend;
				uniform float uPreviewSideInsetUv;
				uniform float uPreviewBandTrimUv;
				out vec4 outColor;

				// Card layout, in UV space (origin bottom-left, 0..1 on each axis):
				//   y >= HEADER_TOP  → header band
				//   FOOTER_TOP <= y < HEADER_TOP → body
				//   y < FOOTER_TOP  → footer band
				// Header/footer UVs match vector.ts CARD_HEADER_TOP_UV / CARD_FOOTER_TOP_UV.
				const float HEADER_TOP = 0.75;
				const float FOOTER_TOP = 0.25;

				float sdRoundedBox(vec2 p, vec2 b, float r) {
					vec2 q = abs(p) - b + vec2(r);
					return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
				}
				void main() {
					if (vAlive < 0.5) discard;
					vec2 uv = vUv;
					// Tight corners + thin border match the reference cards.
					vec2 halfSize = vec2(0.495, 0.495);
					float r = 0.035;
					vec2 p = uv - 0.5;
					float d = sdRoundedBox(p, halfSize, r);
					float aa = 0.005;
					float inside = 1.0 - smoothstep(-aa, aa, d);
					if (inside <= 0.001) discard;

					// Flat near-black body, single brighter shade for header + footer.
					vec3 body   = vec3(0.058, 0.064, 0.078);
					vec3 band   = vec3(0.092, 0.100, 0.120);
					vec3 fill = body;
					if (uv.y >= HEADER_TOP) fill = band;
					if (uv.y <  FOOTER_TOP) fill = band;

					// Hairline dividers between bands.
					vec3 divider = vec3(0.155, 0.170, 0.205);
					float headerDiv = smoothstep(0.004, 0.0, abs(uv.y - HEADER_TOP));
					fill = mix(fill, divider, headerDiv * 0.85);
					float footerDiv = smoothstep(0.004, 0.0, abs(uv.y - FOOTER_TOP));
					fill = mix(fill, divider, footerDiv * 0.85);

					// Thin neutral border, brighter on hover/framed.
					float borderMask = smoothstep(-0.004, -0.001, d);
					vec3 borderCol   = vec3(0.175, 0.190, 0.230);
					vec3 borderHover = vec3(0.345, 0.375, 0.450);
					vec3 borderFram  = vec3(0.260, 0.640, 0.880);
					vec3 border = borderCol;
					if (vState > 0.5) border = borderHover;
					border = mix(border, borderFram, vIsPreview * (0.5 + 0.5 * uPreviewBlend));
					fill = mix(fill, border, borderMask);

					// Preview window — spans the flat body rectangle between dividers.
					if (vIsPreview > 0.5 && uPreviewBlend > 0.001) {
						float sx = clamp(uPreviewSideInsetUv, 0.001, 0.24);
						float bandTrim = clamp(uPreviewBandTrimUv, 0.001, 0.12);
						vec2 insetLo = vec2(sx, FOOTER_TOP + bandTrim);
						vec2 insetHi = vec2(1.0 - sx, HEADER_TOP - bandTrim);
						vec2 t = (uv - insetLo) / (insetHi - insetLo);
						if (t.x > 0.0 && t.x < 1.0 && t.y > 0.0 && t.y < 1.0) {
							vec3 prev = texture(uPreview, vec2(t.x, 1.0 - t.y)).rgb;
							fill = mix(fill, prev, uPreviewBlend);
						}
					}

					outColor = vec4(fill, inside);
				}
			`,
		});

		this.bodyMesh = new THREE.Mesh(this.geom, this.bodyMat);
		this.bodyMesh.frustumCulled = false;
	}

	setView(cam: THREE.OrthographicCamera): void {
		this.bodyMat.uniforms.uProj.value = cam.projectionMatrix;
		this.bodyMat.uniforms.uView.value = cam.matrixWorldInverse;
		this.shadowMat.uniforms.uProj.value = cam.projectionMatrix;
		this.shadowMat.uniforms.uView.value = cam.matrixWorldInverse;
	}

	setPreview(slot: number, blend: number): void {
		this.bodyMat.uniforms.uPreviewId.value = slot;
		this.bodyMat.uniforms.uPreviewBlend.value = blend;
	}

	setPreviewTexture(tex: THREE.Texture | null): void {
		this.bodyMat.uniforms.uPreview.value = tex;
	}

	dispose(): void {
		this.bodyMat.dispose();
		this.shadowMat.dispose();
		this.geom.dispose();
	}
}

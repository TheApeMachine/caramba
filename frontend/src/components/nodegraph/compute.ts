import * as THREE from "three";
import {
	edgesTexture,
	INPUT_OFFSET_X,
	INPUT_OFFSET_Y,
	MAX_EDGES,
	MAX_NODES,
	NODE_H,
	NODE_W,
	nodesTexture,
	OUTPUT_OFFSET_X,
	OUTPUT_OFFSET_Y,
	PATH_TEXELS_PER_EDGE,
} from "./vector";

/*
GPU compute pass: one fragment per (corner, edge) texel of edgePathsTexture.
Reads nodesTexture (node positions) and edgesTexture (endpoint indices),
runs Manhattan routing in the fragment shader with obstacle avoidance, and
writes a single corner's (x, y) as fragment color into a float render target
that aliases edgePathsTexture.

Per-edge layout (8 corners):
  forward case (output's stub_x < input's stub_x):
    p0 = output port
    p1 = (output_x + STUB, output_y)
    p2 = (vx, output_y)
    p3 = (vx, input_y)
    p4 = (input_x - STUB, input_y)
    p5 = input port
    p6,p7 = duplicate of input port (zero-length tail segments)
  backward case (input's stub_x <= output's stub_x):
    p0 = output port
    p1 = (output_x + STUB, output_y)
    p2 = (east, output_y)
    p3 = (east, vy)
    p4 = (west, vy)
    p5 = (west, input_y)
    p6 = (input_x - STUB, input_y)
    p7 = input port

vx in the forward case is scanned outward from the midpoint between the
two stubs until it clears all other live node bodies (including source and
sink). vy in the backward case is the closer of (above-all-nodes,
below-all-nodes); east/west are clear vertical buses outside the bounding
x-extent of source and sink.

This module renders once whenever the store revision changes.
*/

const NODE_HALF_W = NODE_W / 2;
const NODE_HALF_H = NODE_H / 2;
const CLEARANCE = 12;
const STUB = 24;

export type PathCompute = {
	run: () => void;
	dispose: () => void;
	texture: THREE.Texture;
};

export function createPathCompute(renderer: THREE.WebGLRenderer): PathCompute {
	const target = new THREE.WebGLRenderTarget(
		PATH_TEXELS_PER_EDGE,
		MAX_EDGES,
		{
			minFilter: THREE.NearestFilter,
			magFilter: THREE.NearestFilter,
			format: THREE.RGBAFormat,
			type: THREE.FloatType,
		},
	);

	/*
	The render target's texture replaces edgePathsTexture as the path
	source for the edge shader. Callers should bind compute.texture to
	their edge material's uPaths uniform.
	*/

	const scene = new THREE.Scene();
	const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
	camera.position.z = 1;

	const quad = new THREE.PlaneGeometry(2, 2);
	const mat = new THREE.RawShaderMaterial({
		glslVersion: THREE.GLSL3,
		uniforms: {
			uNodes: { value: nodesTexture },
			uEdges: { value: edgesTexture },
			uMaxNodes: { value: MAX_NODES },
			uMaxEdges: { value: MAX_EDGES },
			uTexelsPerEdge: { value: PATH_TEXELS_PER_EDGE },
			uOutOff: {
				value: new THREE.Vector2(OUTPUT_OFFSET_X, OUTPUT_OFFSET_Y),
			},
			uInOff: { value: new THREE.Vector2(INPUT_OFFSET_X, INPUT_OFFSET_Y) },
			uHalfW: { value: NODE_HALF_W },
			uHalfH: { value: NODE_HALF_H },
			uClearance: { value: CLEARANCE },
			uStub: { value: STUB },
		},
		vertexShader: /* glsl */ `
			in vec3 position;
			void main() {
				gl_Position = vec4(position.xy, 0.0, 1.0);
			}
		`,
		fragmentShader: /* glsl */ `
			precision highp float;
			uniform sampler2D uNodes;
			uniform sampler2D uEdges;
			uniform float uMaxNodes;
			uniform float uMaxEdges;
			uniform float uTexelsPerEdge;
			uniform vec2 uOutOff;
			uniform vec2 uInOff;
			uniform float uHalfW;
			uniform float uHalfH;
			uniform float uClearance;
			uniform float uStub;
			out vec4 outColor;

			vec4 sampleNode(float idx) {
				return texture(uNodes, vec2((idx + 0.5) / uMaxNodes, 0.5));
			}

			// Returns true if x is inside any live obstacle's horizontal span
			// (with clearance), excluding the source/sink themselves.
			bool xHits(float x, float skipA, float skipB) {
				for (int i = 0; i < 4096; i++) {
					if (float(i) >= uMaxNodes) break;
					if (float(i) == skipA || float(i) == skipB) continue;
					vec4 n = sampleNode(float(i));
					if (n.a < 0.5) continue;
					float left = n.x - uHalfW - uClearance;
					float right = n.x + uHalfW + uClearance;
					if (x > left && x < right) return true;
				}
				return false;
			}

			void main() {
				vec2 uv = gl_FragCoord.xy;
				int corner = int(uv.x);
				float edgeIdx = floor(uv.y);

				vec4 e = texture(uEdges,
					vec2((edgeIdx + 0.5) / uMaxEdges, 0.5));
				if (e.b < 0.5) { outColor = vec4(0.0); return; }

				float fromIdx = e.r;
				float toIdx = e.g;
				vec4 fromN = sampleNode(fromIdx);
				vec4 toN = sampleNode(toIdx);

				vec2 p0 = fromN.xy + uOutOff;   // output port
				vec2 pIn = toN.xy + uInOff;     // input port
				float p1x = p0.x + uStub;
				float p4x = pIn.x - uStub;

				bool forward = p1x < p4x;

				vec2 c;

				if (forward) {
					// scan vx outward from midpoint
					float vx = (p1x + p4x) * 0.5;
					for (int i = 1; i <= 32; i++) {
						if (!xHits(vx, fromIdx, toIdx)) break;
						float step = float(i) * uStub;
						float right = (p1x + p4x) * 0.5 + step;
						float left  = (p1x + p4x) * 0.5 - step;
						if (!xHits(right, fromIdx, toIdx)) { vx = right; break; }
						if (!xHits(left,  fromIdx, toIdx)) { vx = left;  break; }
					}

					if      (corner == 0) c = p0;
					else if (corner == 1) c = vec2(p1x, p0.y);
					else if (corner == 2) c = vec2(vx,  p0.y);
					else if (corner == 3) c = vec2(vx,  pIn.y);
					else if (corner == 4) c = vec2(p4x, pIn.y);
					else                  c = pIn;
				} else {
					// backward: route through east/west buses + bypass vy
					float east = max(fromN.x, toN.x) + uHalfW + uClearance + uStub;
					float west = min(fromN.x, toN.x) - uHalfW - uClearance - uStub;
					float above = min(fromN.y, toN.y) - uHalfH - uClearance - uStub;
					float below = max(fromN.y, toN.y) + uHalfH + uClearance + uStub;
					float midPortY = (p0.y + pIn.y) * 0.5;
					float vy = abs(below - midPortY) <= abs(above - midPortY)
						? below : above;

					if      (corner == 0) c = p0;
					else if (corner == 1) c = vec2(p1x, p0.y);
					else if (corner == 2) c = vec2(east, p0.y);
					else if (corner == 3) c = vec2(east, vy);
					else if (corner == 4) c = vec2(west, vy);
					else if (corner == 5) c = vec2(west, pIn.y);
					else if (corner == 6) c = vec2(p4x, pIn.y);
					else                  c = pIn;
				}

				outColor = vec4(c, 1.0, 1.0);
			}
		`,
	});

	const mesh = new THREE.Mesh(quad, mat);
	scene.add(mesh);

	const run = () => {
		const prevTarget = renderer.getRenderTarget();
		renderer.setRenderTarget(target);
		renderer.render(scene, camera);
		renderer.setRenderTarget(prevTarget);
	};

	const dispose = () => {
		mat.dispose();
		quad.dispose();
		target.dispose();
	};

	return { run, dispose, texture: target.texture };
}

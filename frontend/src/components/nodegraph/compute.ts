import * as THREE from "three";
import {
	edgesTexture,
	MAX_EDGES,
	MAX_NODES,
	MAX_PORTS,
	NODE_H,
	NODE_W,
	nodesTexture,
	PATH_TEXELS_PER_EDGE,
	portsTexture,
} from "./vector";

/*
GPU compute pass: one fragment per (corner, edge) texel of the path target.
Reads nodesTexture, edgesTexture, portsTexture. Edges store port indices,
not node indices, so each endpoint resolves through ports → (nodeIdx,
offset) → nodes → (x, y). This keeps the multi-port layout the only place
that knows where ports live.

Manhattan routing matches the previous version: forward case uses a
midpoint vertical bus; backward case wraps around to a clear east/west
bus then crosses on an above/below row.
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

	const scene = new THREE.Scene();
	const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
	camera.position.z = 1;

	const quad = new THREE.PlaneGeometry(2, 2);
	const mat = new THREE.RawShaderMaterial({
		glslVersion: THREE.GLSL3,
		uniforms: {
			uNodes: { value: nodesTexture },
			uEdges: { value: edgesTexture },
			uPorts: { value: portsTexture },
			uMaxNodes: { value: MAX_NODES },
			uMaxEdges: { value: MAX_EDGES },
			uMaxPorts: { value: MAX_PORTS },
			uTexelsPerEdge: { value: PATH_TEXELS_PER_EDGE },
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
			uniform sampler2D uPorts;
			uniform float uMaxNodes;
			uniform float uMaxEdges;
			uniform float uMaxPorts;
			uniform float uTexelsPerEdge;
			uniform float uHalfW;
			uniform float uHalfH;
			uniform float uClearance;
			uniform float uStub;
			out vec4 outColor;

			vec4 sampleNode(float idx) {
				return texture(uNodes, vec2((idx + 0.5) / uMaxNodes, 0.5));
			}

			vec4 samplePort(float idx) {
				return texture(uPorts, vec2((idx + 0.5) / uMaxPorts, 0.5));
			}

			// Resolves a port's world position via its parent node.
			// Returns vec3(world.x, world.y, nodeIdx). Port texel layout:
			//   .r = nodeIdx (or -1 sentinel), .g/.b = offset, .a = packed.
			vec3 portWorld(float portIdx) {
				vec4 port = samplePort(portIdx);
				float nodeIdx = port.r;
				if (nodeIdx < 0.0) return vec3(0.0, 0.0, -1.0);
				vec4 n = sampleNode(nodeIdx);
				if (n.a < 0.5) return vec3(0.0, 0.0, -1.0);
				return vec3(n.x + port.g, n.y + port.b, nodeIdx);
			}

			// Returns true if x falls inside any live obstacle's horizontal
			// span (with clearance), excluding the source and sink nodes.
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

				vec3 fromW = portWorld(e.r);
				vec3 toW = portWorld(e.g);
				if (fromW.z < 0.0 || toW.z < 0.0) {
					outColor = vec4(0.0);
					return;
				}

				vec2 p0 = fromW.xy;
				vec2 pIn = toW.xy;
				vec4 fromN = sampleNode(fromW.z);
				vec4 toN = sampleNode(toW.z);
				float p1x = p0.x + uStub;
				float p4x = pIn.x - uStub;

				// Forward whenever the output port sits left of the input
				// port. Tight gaps (stub spans overlap, p1x > p4x) still take
				// the forward branch — the result is a short Z with overlapping
				// stub segments, which reads fine at edge thickness. The
				// backward branch (east/west bus detour) only fires when the
				// output is actually right of the input.
				bool forward = p0.x <= pIn.x;
				vec2 c;

				if (forward) {
					float vx = (p1x + p4x) * 0.5;
					for (int i = 1; i <= 32; i++) {
						if (!xHits(vx, fromW.z, toW.z)) break;
						float step = float(i) * uStub;
						float right = (p1x + p4x) * 0.5 + step;
						float left = (p1x + p4x) * 0.5 - step;
						if (!xHits(right, fromW.z, toW.z)) { vx = right; break; }
						if (!xHits(left, fromW.z, toW.z)) { vx = left; break; }
					}

					if      (corner == 0) c = p0;
					else if (corner == 1) c = vec2(p1x, p0.y);
					else if (corner == 2) c = vec2(vx,  p0.y);
					else if (corner == 3) c = vec2(vx,  pIn.y);
					else if (corner == 4) c = vec2(p4x, pIn.y);
					else                  c = pIn;
				} else {
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

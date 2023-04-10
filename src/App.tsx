import { useFBO, PerspectiveCamera as DPerCam, OrthographicCamera } from "@react-three/drei";
import { Canvas, createPortal, useFrame, useThree } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import { PerspectiveCamera, Scene, Color, ShaderMaterial, Mesh } from "three";

/* Simplex noise fragment shader */

const noiseFrag = `
  uniform float uTime;
  uniform vec2 uResolution;
  varying vec2 vUv;

  const float PI = 3.1415926535897932384626433832795;

  //	Classic Perlin 3D Noise 
  //	by Stefan Gustavson
  //
  vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
  vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
  vec3 fade(vec3 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}
  
  float cnoise(vec3 P){
    vec3 Pi0 = floor(P); // Integer part for indexing
    vec3 Pi1 = Pi0 + vec3(1.0); // Integer part + 1
    Pi0 = mod(Pi0, 289.0);
    Pi1 = mod(Pi1, 289.0);
    vec3 Pf0 = fract(P); // Fractional part for interpolation
    vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = Pi0.zzzz;
    vec4 iz1 = Pi1.zzzz;
  
    vec4 ixy = permute(permute(ix) + iy);
    vec4 ixy0 = permute(ixy + iz0);
    vec4 ixy1 = permute(ixy + iz1);
  
    vec4 gx0 = ixy0 / 7.0;
    vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
    gx0 = fract(gx0);
    vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
    vec4 sz0 = step(gz0, vec4(0.0));
    gx0 -= sz0 * (step(0.0, gx0) - 0.5);
    gy0 -= sz0 * (step(0.0, gy0) - 0.5);
  
    vec4 gx1 = ixy1 / 7.0;
    vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
    gx1 = fract(gx1);
    vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
    vec4 sz1 = step(gz1, vec4(0.0));
    gx1 -= sz1 * (step(0.0, gx1) - 0.5);
    gy1 -= sz1 * (step(0.0, gy1) - 0.5);
  
    vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
    vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
    vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
    vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
    vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
    vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
    vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
    vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);
  
    vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
    g000 *= norm0.x;
    g010 *= norm0.y;
    g100 *= norm0.z;
    g110 *= norm0.w;
    vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
    g001 *= norm1.x;
    g011 *= norm1.y;
    g101 *= norm1.z;
    g111 *= norm1.w;
  
    float n000 = dot(g000, Pf0);
    float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
    float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
    float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
    float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
    float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
    float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
    float n111 = dot(g111, Pf1);
  
    vec3 fade_xyz = fade(Pf0);
    vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
    vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
    float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x); 
    return 2.2 * n_xyz;
  }

  const vec3 red = vec3(1., 0., 0.);
  const vec3 blue = vec3(0., 1., 1.);
  const vec3 violet = vec3(0.5, 0., 1.);
  const vec3 yellow = vec3(1., 1., 0.);
  const vec3 seagreen = vec3(0.18, 0.55, 0.34);
  const vec3 hotpink = vec3(1., 0.41, 0.71);
  const vec3 black = vec3(0., 0., 0.);

  void main() {
    vec2 uv = vUv * 10.0 - 1.0;

    float f = cnoise(vec3(uv, uTime * 0.5));

    vec3 color = mix(hotpink, violet, f);

    gl_FragColor = vec4(color, 1.0);
  }
`;

const noiseVert = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const pixelationFragment = `
  uniform vec2 uResolution;
  uniform float uTime;
  uniform sampler2D uTexture;
  uniform vec2 uMouse;

  varying vec2 vUv;

  const float pixelSize = 0.03;
  const vec2 maskSize = vec2(0.04);

  vec2 map(vec2 value, vec2 min1, vec2 max1, vec2 min2, vec2 max2) {
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
  }

  void main() {
    vec2 uv = vUv;

    vec2 mousePos = map(uMouse, vec2(-1.0), vec2(1.0), vec2(0.0), vec2(1.0));

    // Calculate the distance from the current fragment to the mouse position, taking into account the rectangle mask
    vec2 maskCenter = mousePos;
    vec2 maskedUv = abs(uv - maskCenter) - maskSize * 0.5;
    float distance = max(maskedUv.x, maskedUv.y);
    
    // Calculate the size of each pixel in texture coordinates
    vec2 pixelTexCoord = vec2(pixelSize, pixelSize);

    // Calculate the position of the current fragment in texture coordinates
    vec2 roundedTexCoord = floor(uv / pixelTexCoord) * pixelTexCoord;

    if (distance < 0.0) { // Use a threshold of 0.0 for the rectangle mask
      // Sample the texture using the rounded texture coordinates
      vec4 sampledColor = texture2D(uTexture, roundedTexCoord);

      gl_FragColor = sampledColor;
    } else {
      vec4 originalColor = texture2D(uTexture, uv);
      gl_FragColor = originalColor;
    }
  }
`;

function View() {
  const cam = useRef<PerspectiveCamera>();
  const scene = useMemo(() => {
    const myScene = new Scene();
    myScene.background = new Color("green");
    return myScene;
  }, []);

  const renderTarget = useFBO();

  const noiseShaderRef = useRef<ShaderMaterial>(null);
  const pixelationShaderRef = useRef<ShaderMaterial>(null);

  const displayRef = useRef<Mesh>(null);

  const { getCurrentViewport } = useThree(state => state.viewport)

  const { width, height } = getCurrentViewport();

  const ratio = (width / height) > 1 ? width : height;

  useFrame(( { gl, clock, mouse } ) => {
    if (!cam.current) return;
    gl.setRenderTarget(renderTarget);
    gl.render(scene, cam.current);
    gl.setRenderTarget(null);

    if (noiseShaderRef.current) {
      noiseShaderRef.current.uniforms.uTime.value = clock.getElapsedTime(); // TODO: remove * 5
      noiseShaderRef.current.uniforms.uResolution.value = [gl.domElement.width, gl.domElement.height];
    }

    if (pixelationShaderRef.current) {
      pixelationShaderRef.current.uniforms.uMouse.value = [mouse.x, mouse.y];
    }
  })

  return (
    <>
      <DPerCam ref={cam} position={[0, 0, 2]} makeDefault />
      {/* <OrthographicCamera
        ref={cam}
        makeDefault
        zoom={2}
        left={-1}
        right={1}
        top={1}
        bottom={-1}
        near={0.1}
        far={1000}
        position={[0, 0, 1]}
      /> */}
      {createPortal(
        <>
          <mesh scale={[2.7, 2, 1]}>
            <planeBufferGeometry />
            <shaderMaterial ref={noiseShaderRef} fragmentShader={noiseFrag} vertexShader={noiseVert} uniforms={{ uTime: { value: 0 }, uResolution: { value: [1, 1] } }} />
          </mesh>
        </>,
        scene
      )}

      <mesh ref={displayRef} scale={[ratio, ratio, 1]}>
        <planeBufferGeometry />
        <shaderMaterial
          ref={pixelationShaderRef}
          fragmentShader={pixelationFragment}
          vertexShader={noiseVert}
          uniforms={{ uTexture: { value: renderTarget.texture }, uMouse: { value: [0, 0] } }}
        />
        {/* <meshBasicMaterial map={renderTarget.texture} /> */}
      </mesh>
    </>
  )
}

export default function App(): JSX.Element {
  return (
    <div className="canvas-container">
      <Canvas>
        <color attach="background" args={["hotpink"]} />
        <View />
      </Canvas>
    </div>
  )
}
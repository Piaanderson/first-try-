import { Suspense, useEffect, useRef } from 'react'
import { Canvas, useLoader, useFrame } from '@react-three/fiber'
import { OrbitControls, Environment, Center } from '@react-three/drei'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader'
import * as THREE from 'three'

function STLModel({ url, color = '#c0c0d0' }) {
  const geometry = useLoader(STLLoader, url)
  const meshRef = useRef()

  useEffect(() => {
    if (geometry) geometry.computeVertexNormals()
  }, [geometry])

  useFrame((_, delta) => {
    if (meshRef.current) meshRef.current.rotation.y += delta * 0.2
  })

  return (
    <Center>
      <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow>
        <meshStandardMaterial
          color={color}
          metalness={0.6}
          roughness={0.35}
          side={THREE.DoubleSide}
        />
      </mesh>
    </Center>
  )
}

export default function ModelViewer({ url, label, color }) {
  return (
    <div className="viewer-wrap">
      {label && <div className="viewer-label">{label}</div>}
      <Canvas
        camera={{ position: [0, 0, 120], fov: 45 }}
        shadows
        style={{ background: '#0d0d16', borderRadius: '10px' }}
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[50, 80, 60]} intensity={1.2} castShadow />
        <directionalLight position={[-40, -30, -40]} intensity={0.3} />
        <Environment preset="city" />
        <Suspense fallback={null}>
          <STLModel url={url} color={color} />
        </Suspense>
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  )
}

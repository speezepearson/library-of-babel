import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { decode } from '../lib/decode';
import { DEFAULT_MODEL_ID } from '../lib/model';
import { DEFAULT_SAMPLER_CONFIG } from '../lib/types';
import type { ModelId, SamplerConfig } from '../lib/types';

type BookNumber = number;

const ROWS = 5;
const COLS = 32;

const BOOK_WIDTH = 0.12;
const BOOK_DEPTH = 0.6;
const BOOK_GAP = 0.02;
const SHELF_HEIGHT = 1.1;
const SHELF_DEPTH = 0.7;
const SHELF_THICKNESS = 0.04;

const SYSTEM_PROMPT = 'You are a helpful, concise assistant.';
const USER_MESSAGE = 'Tell me something interesting.';

function bookColor(n: BookNumber): THREE.Color {
  // Deterministic hue from book number, rich saturated colors
  const hue = ((n * 137.508) % 360) / 360;
  const sat = 0.4 + (n % 7) * 0.08;
  const lit = 0.25 + (n % 5) * 0.05;
  return new THREE.Color().setHSL(hue, sat, lit);
}

function bookHeight(n: BookNumber): number {
  // Slight height variation
  return 0.7 + ((n * 31 + 17) % 13) / 13 * 0.25;
}

interface BookMeshData {
  mesh: THREE.Mesh;
  bookNumber: BookNumber;
}

export function FancyUI() {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const booksRef = useRef<BookMeshData[]>([]);
  const raycasterRef = useRef(new THREE.Raycaster());
  const mouseRef = useRef(new THREE.Vector2());
  const animFrameRef = useRef(0);

  const [openBook, setOpenBook] = useState<BookNumber | null>(null);
  const [text, setText] = useState('');
  const [status, setStatus] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const stopRef = useRef(false);

  // Build the 3D scene
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x08090d);
    scene.fog = new THREE.Fog(0x08090d, 15, 30);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      50,
      container.clientWidth / container.clientHeight,
      0.1,
      50,
    );
    const shelfTotalWidth = COLS * (BOOK_WIDTH + BOOK_GAP);
    camera.position.set(shelfTotalWidth / 2, ROWS * SHELF_HEIGHT / 2, 8);
    camera.lookAt(shelfTotalWidth / 2, ROWS * SHELF_HEIGHT / 2 - 0.5, 0);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting
    const ambient = new THREE.AmbientLight(0x404060, 0.6);
    scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0xffeedd, 1.2);
    keyLight.position.set(5, 8, 10);
    keyLight.castShadow = true;
    scene.add(keyLight);

    const fillLight = new THREE.PointLight(0x7c85e0, 0.4, 20);
    fillLight.position.set(-3, 5, 6);
    scene.add(fillLight);

    // Build bookshelf
    const shelfMaterial = new THREE.MeshStandardMaterial({
      color: 0x3d2b1f,
      roughness: 0.8,
    });

    const books: BookMeshData[] = [];
    const shelfWidth = COLS * (BOOK_WIDTH + BOOK_GAP) + 0.1;

    // Back panel
    const backGeom = new THREE.BoxGeometry(shelfWidth + 0.1, ROWS * SHELF_HEIGHT + SHELF_THICKNESS, 0.03);
    const backMesh = new THREE.Mesh(backGeom, shelfMaterial);
    backMesh.position.set(
      shelfWidth / 2 - 0.05,
      (ROWS * SHELF_HEIGHT) / 2,
      -SHELF_DEPTH / 2,
    );
    backMesh.receiveShadow = true;
    scene.add(backMesh);

    // Side panels
    for (const side of [-1, 1]) {
      const sideGeom = new THREE.BoxGeometry(0.05, ROWS * SHELF_HEIGHT + SHELF_THICKNESS, SHELF_DEPTH);
      const sideMesh = new THREE.Mesh(sideGeom, shelfMaterial);
      sideMesh.position.set(
        side === -1 ? -0.075 : shelfWidth + 0.025,
        (ROWS * SHELF_HEIGHT) / 2,
        0,
      );
      sideMesh.receiveShadow = true;
      scene.add(sideMesh);
    }

    for (let row = 0; row < ROWS; row++) {
      const shelfY = row * SHELF_HEIGHT;

      // Shelf plank
      const plankGeom = new THREE.BoxGeometry(shelfWidth, SHELF_THICKNESS, SHELF_DEPTH);
      const plankMesh = new THREE.Mesh(plankGeom, shelfMaterial);
      plankMesh.position.set(shelfWidth / 2 - 0.05, shelfY, 0);
      plankMesh.receiveShadow = true;
      scene.add(plankMesh);

      // Books
      for (let col = 0; col < COLS; col++) {
        const bookNum = row * COLS + col;
        const h = bookHeight(bookNum);
        const geom = new THREE.BoxGeometry(BOOK_WIDTH, h, BOOK_DEPTH);
        const mat = new THREE.MeshStandardMaterial({
          color: bookColor(bookNum),
          roughness: 0.7,
          metalness: 0.1,
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.position.set(
          col * (BOOK_WIDTH + BOOK_GAP) + BOOK_WIDTH / 2,
          shelfY + SHELF_THICKNESS / 2 + h / 2,
          0,
        );
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
        books.push({ mesh, bookNumber: bookNum });
      }
    }

    // Top shelf
    const topPlankGeom = new THREE.BoxGeometry(shelfWidth, SHELF_THICKNESS, SHELF_DEPTH);
    const topPlankMesh = new THREE.Mesh(topPlankGeom, shelfMaterial);
    topPlankMesh.position.set(shelfWidth / 2 - 0.05, ROWS * SHELF_HEIGHT, 0);
    topPlankMesh.receiveShadow = true;
    scene.add(topPlankMesh);

    booksRef.current = books;

    // Animate
    function animate() {
      animFrameRef.current = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    }
    animate();

    // Resize handler
    function onResize() {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    window.addEventListener('resize', onResize);

    return () => {
      window.removeEventListener('resize', onResize);
      cancelAnimationFrame(animFrameRef.current);
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  // Click handler for raycasting
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const container = containerRef.current;
      const camera = cameraRef.current;
      if (!container || !camera || openBook !== null) return;

      const rect = container.getBoundingClientRect();
      mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycasterRef.current.setFromCamera(mouseRef.current, camera);
      const meshes = booksRef.current.map((b) => b.mesh);
      const intersects = raycasterRef.current.intersectObjects(meshes);

      if (intersects.length > 0) {
        const hit = booksRef.current.find((b) => b.mesh === intersects[0].object);
        if (hit) {
          setOpenBook(hit.bookNumber);
        }
      }
    },
    [openBook],
  );

  // Decode when a book is opened
  useEffect(() => {
    if (openBook === null) return;

    setText('');
    setStatus('Loading model...');
    setIsLoading(true);
    stopRef.current = false;

    const seed = BigInt(openBook);

    decode(SYSTEM_PROMPT, USER_MESSAGE, seed, {
      onToken(piece) {
        setText((prev) => prev + piece);
      },
      onStatus(msg) {
        setStatus(msg);
      },
      shouldStop() {
        return stopRef.current;
      },
    }, DEFAULT_MODEL_ID, DEFAULT_SAMPLER_CONFIG)
      .then(() => {
        setIsLoading(false);
        setStatus('');
      })
      .catch((err) => {
        setIsLoading(false);
        setStatus(`Error: ${(err as Error).message}`);
      });

    return () => {
      stopRef.current = true;
    };
  }, [openBook]);

  const handleClose = useCallback(() => {
    stopRef.current = true;
    setOpenBook(null);
    setText('');
    setStatus('');
    setIsLoading(false);
  }, []);

  // Close on Escape
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape' && openBook !== null) {
        handleClose();
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [openBook, handleClose]);

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden' }}>
      <div
        ref={containerRef}
        onClick={handleCanvasClick}
        style={{ width: '100%', height: '100%', cursor: openBook === null ? 'pointer' : 'default' }}
      />

      {/* Title overlay */}
      <div style={{
        position: 'absolute',
        top: '1.5rem',
        left: '1.5rem',
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: '0.9rem',
        color: '#7c85e0',
        pointerEvents: 'none',
      }}>
        arithmetic_llm_decoder
        <div style={{
          fontSize: '0.7rem',
          color: '#5c5f77',
          marginTop: '0.3rem',
        }}>
          click a book to decode its seed
        </div>
      </div>

      {/* Paper overlay */}
      {openBook !== null && (
        <div
          onClick={handleClose}
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(8, 9, 13, 0.85)',
            cursor: 'pointer',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: 'min(600px, 90vw)',
              maxHeight: '80vh',
              backgroundColor: '#f5f0e8',
              borderRadius: '4px',
              padding: '2.5rem',
              boxShadow: '0 20px 60px rgba(0,0,0,0.6)',
              cursor: 'default',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            {/* Paper header */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '1.5rem',
              borderBottom: '1px solid #d4cfc4',
              paddingBottom: '0.75rem',
            }}>
              <span style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: '0.72rem',
                color: '#8b8578',
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
              }}>
                Book #{openBook} &middot; seed = {openBook}
              </span>
              <button
                onClick={handleClose}
                style={{
                  background: 'none',
                  border: '1px solid #d4cfc4',
                  borderRadius: '4px',
                  padding: '0.3rem 0.6rem',
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: '0.7rem',
                  color: '#8b8578',
                  cursor: 'pointer',
                }}
              >
                close
              </button>
            </div>

            {/* Scrollable text area */}
            <div style={{
              flex: 1,
              overflowY: 'auto',
              fontFamily: "'DM Sans', serif",
              fontSize: '0.95rem',
              lineHeight: 1.8,
              color: '#2a2520',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}>
              {text || (
                <span style={{ color: '#b0a898', fontStyle: 'italic' }}>
                  {status || 'Loading...'}
                </span>
              )}
              {isLoading && text && <span style={{
                display: 'inline-block',
                width: '2px',
                height: '1em',
                backgroundColor: '#7c85e0',
                marginLeft: '2px',
                verticalAlign: 'text-bottom',
                animation: 'blink 1s step-end infinite',
              }} />}
            </div>

            {/* Status footer */}
            {status && (
              <div style={{
                marginTop: '0.75rem',
                paddingTop: '0.5rem',
                borderTop: '1px solid #d4cfc4',
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: '0.65rem',
                color: '#b0a898',
              }}>
                {status}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

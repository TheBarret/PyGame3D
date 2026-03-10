# PyGame3D

This project extends PyGame with a lightweight 3D wireframe rendering layer. 
It demonstrates that PyGame, combined with NumPy for mathematical operations, 
can achieve smooth 3D wireframe rendering without requiring external 3D engines or OpenGL knowledge. 

The goal is to provide a clean, Pythonic API that feels native to PyGame developers while handling the complexities of 3D transformation, 
projection, and rendering behind the scenes. 


# Development Process

This project was developed through collaboration between multiple AI language models, each contributing their specific expertise: 
 
Qwen provided the complete mathematical implementation, including perspective projection matrices, 
homogeneous clipping algorithms, quaternion operations, frustum plane extraction, and ray intersection tests. 
All core mathematical functions in math_core.py originated from Qwen's contributions. 
 
Claude designed the overall architecture, including the transform system with dirty flag propagation, camera class with orbital controls, scene graph management, renderer pipeline, and the clean public API. Claude also wrote the comprehensive demo and ensured the architectural patterns matched PyGame conventions. 
 
DeepSeek served as the orchestrator, maintaining the project vision, coordinating between the other models, identifying integration points, and ensuring the final package remained coherent. DeepSeek also handled the adaptation layer that allowed Qwen's math to plug seamlessly into Claude's architecture. 
 
This multi model approach combined Qwen's mathematical depth with Claude's architectural clarity, guided by DeepSeek's project management, resulting in a cohesive extension that would have been difficult for any single model to produce alone. 
 
# Core features & Goals
- 3D wireframe rendering using only line drawing primitives
- Proper perspective projection with homogeneous coordinates
- Smooth camera controls (orbit, pan, zoom) without gimbal lock
- Scene graph with parent child transform inheritance
- Frustum culling for performance optimization
- Object picking via ray casting
- Reasonable performance for thousands of edges

# Demo

![pygame3d](https://github.com/user-attachments/assets/4c095932-25a3-45d6-b89e-4c41c9e7c4f0)

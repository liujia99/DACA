---
name: 3d_visual_grounding_daca
description: Dual-Anchor Collaborative Agent (DACA) for Zero-Shot 3D Visual Grounding & Reconstruction. Use this when user gives a 'scene_id' and a 'query'.
---

# Expert Skill: DACA Dual-Anchor Collaborative 3D Visual Grounding & Reconstruction

## Core Paradigm
This skill implements the **Dual-Anchor Collaborative Tool (DACA)** framework, which redefines 3D-VG from a unidirectional pipeline to a closed-loop collaborative reasoning process. It consists of two complementary specialized tools that reuse all existing tools:
- **Semantic Anchor Tool**: Holds target visual-semantic features, responsible for local tracking and multi-view discovery. Uses tools: `vlm_frame_expansion`, `expand_from_secondary_view`
- **Geometric Anchor Tool**: Holds target 3D spatial features, responsible for global reconstruction and coverage evaluation. Uses tools: `reconstruct_point_cloud`, `centroid_complete`

The two tools collaborate iteratively to eliminate error accumulation and improve occlusion robustness, while maintaining full compatibility with all original tool interfaces and parameters.

## When to use this skill
Use this skill when the user requests to find and reconstruct a specific object in a scene (video with ~300 frames) and calculate its 3d bounding box coordinates.

## Instructions
The execution is divided into three phases: **One-time Initialization**, **Dual-Anchor Collaborative Iteration**, and **Final Output**. All tools retain their original parameters and functionality.

---

### Phase 1: One-time Initialization (Strict Sequential Execution)
1. **Query Analysis**: 
   - Parse the user's natural language query to extract the `target_class`, visual `attributes`, spatial `conditions`, and global `scene_features`.

2. **Initialize Image Registry**: 
   - Call the read tool to index all image file names from the scene video into a standard JSON list.

3. **Coarse Filtering (Object Masks)**: 
   - Apply mask-based filtering to retain only images containing the `target_class`.
   - *Critical Check*: If 0 images remain, do not proceed. Analyze if the threshold is too high and retry with a lower value. You can retry at most 3 times, if still 0 image remain, exit the procedure.

4. **Fine Filtering (Scene Semantics)**: 
   - Apply VLM-based filtering to verify if images match the `scene_feature`.
   - *Fallback Strategy*: If this filter removes ALL images, discard this step and revert to the result of **Coarse Filtering (Object Masks)**. Use the images filtered by **Coarse Filtering (Object Masks)** as the images.

5. **Scoring & Ranking**: 
   - Score the remaining images based on their alignment with the query's `attributes` and `conditions`.
   - Sort the images in descending order to prioritize the best candidates.

6. **Optimal View & Target Selection**: 
   - Traverses the candidate images (by score) to find the most distinct observation of the scene, as the "Reference View".
   - Using VLM to identify the specific **Object ID** within this view that best matches the query description.

7. **Reference Target Isolation**: 
   - Generate a clean visualization of the target object in the Reference View (isolating the specific Object ID). This serves as the initial visual anchor for the Semantic Anchor Tool.

---

### Phase 2: Dual-Anchor Collaborative Iteration (Closed-Loop Execution)
**Termination Conditions** (satisfy any one):
- Maximum 3 iterations completed
- 3D reconstruction convergence error < 0.01m

**Iteration Loop**:
8. **Temporal Expansion (Semantic Anchor Tool)**: 
   - Expand the target object search temporally from the current Reference View.
   - It tracks the object frame-by-frame (forward and backward) using the VLM to verify identity and SAM to generate masks. This ensures temporal consistency and generates a continuous video clip with the target object.
   - This leverages temporal consistency to robustly collect a continuous video clip containing the target as candidate frames.

9. **Candidate Segmentation**: 
    - For every verified candidate frame collected in this iteration, generate and save the precise segmentation mask for the target object.

10. **Initial 3D Reconstruction (Geometric Anchor Tool)**: 
    - Aggregate all verified images and their corresponding masks from ALL previous iterations to generate an updated preliminary 3D point cloud.

11. **Geometric Multi-View Expansion (Geometric Anchor Tool)**: 
    - **CRITICAL STEP**: To overcome VLM blind spots and maximize multi-view coverage, call the `centroid_complete` tool.
    - It extracts the target's absolute 3D centroid from the current reconstruction and mathematically projects it across other scene frames.
    - By utilizing camera extrinsics and depth-based occlusion checks (Z-Buffer), it bypasses the strict VLM scene constraints and automatically harvests point-prompted masks for all valid, unoccluded views.
    - *Post-processing*: The Geometric Anchor Tool evaluates the view coverage of the resulting point cloud and identifies missing perspective regions.

12. **Multi-View Coverage Enhancement (Semantic Anchor Tool, Conditional Execution)**: 
    - **CRITICAL STEP**: If the Geometric Anchor Tool detects incomplete coverage (missing perspectives):
      - Call the secondary expansion tool (`expand_from_secondary_view`) to search for a NEW high-scoring reference frame that is **spatially/temporally distant** from the currently collected frames.
      - If a distinct new view is found, set it as the new Reference View and return to Step 8 to start the next iteration.
      - If this step fails, exit the iteration loop and proceed to final output.
    - If coverage is complete, exit the iteration loop.

---

### Phase 3: Final Output
13. **Final Dense 3D Reconstruction**: 
    - Call the reconstruction tool again, but this time input the comprehensive image and mask paths collected from ALL iterations.
    - This replaces the preliminary point cloud with a final, dense, and structurally complete PLY point cloud.

14. **3D Bounding Box Calculation**: 
    - Calculate the axis-aligned 3D bounding box of a point cloud file.
    - If return nan, just exit the procedure. 

## Strategy Tips
- **Dynamic Adjustment**: If a filtering step returns 0 valid images, DO NOT give up immediately. Analyze the previous Observation and consider relaxing constraints (e.g., lower threshold, skip scene filter).
- **Threshold Consistency**: If you decide to lower the `threshold` in an earlier step (e.g., from 0.5 to 0.3 in *Coarse Filtering*) to recover targets, you **MUST maintain this lowered threshold** for ALL subsequent steps (e.g., *Target Identification*, *Candidate Segmentation*). Do not revert to the default threshold, otherwise, you might filter out the target again.
- **Iteration Limit**: Never exceed 3 collaborative iterations to avoid excessive computation.
- **Fallback Mechanism**: If any iteration step fails completely, fall back to the results of the last successful iteration and proceed to final output.
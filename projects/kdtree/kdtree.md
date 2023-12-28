---
layout: default
title: Exploring KD-Trees
---

# Exploring Accelerated KD-Tree Construction Techniques for Efficient Ray Tracing

## Motivations

The driving force behind this project is to explore how to significantly improve the efficiency and practicality of ray tracing, a technique integral to producing high-fidelity graphics in computer simulations, visual effects, and architectural visualizations. Ray tracing's ability to simulate optical phenomena, such as reflections, refractions, and shadows, with a high degree of realism makes it a cornerstone technique in computer graphics. However, its computational intensity, particularly in scenes with complex geometries or dynamic elements, poses significant challenges.

KD-trees (k-dimensional trees) are central to this optimization strategy. They are spatial data structures that efficiently partition the 3D space, reducing the number of intersection tests needed by culling large portions of the scene that are not intersected by a ray. However, the construction and updating of KD-trees, particularly for dynamic scenes where the scene's geometry changes in real time, remains a bottleneck. Traditional construction methods are often slow and not well-suited for scenes that require frequent updates, a common scenario in interactive applications such as virtual reality, gaming, and real-time design simulations, which is why this project aimed to explore more efficient construction techniques.

## Background

### SAH-Based KD-Tree

The paper by Ingo Wald and Vlastimil Havran [2] represents a significant advancement in the field of KD-tree construction for ray tracing. KD-trees have been identified as a preferred method for efficiency structures in ray tracing, particularly those built with cost estimation functions like the Surface Area Heuristic (SAH). However, the traditional algorithms for building SAH KD-trees have been plagued by time complexities of O(N log^2 N) or worse. This paper proposes an innovative algorithm that constructs SAH KD-trees in O(N log N), meeting the theoretical lower bound for such processes.

The focus is not only on the traversal of these trees but also on their efficient construction. The authors note that while KD-trees are well understood in terms of traversal and efficiency, the construction time has often been overlooked. This oversight becomes problematic with the increasing complexity of scenes in ray tracing. This work provides a comprehensive summary of building efficient KD-trees using the Surface Area Heuristic, a discussion on different schemes for building SAH-optimized KD-trees and analyzing their computational complexity, and the introduction of an algorithm that builds an SAH KD-tree in O(N log N), representing an asymptotic improvement in efficiency.

The Surface Area Heuristic is central to this discussion, and is a method to estimate the expected cost of traversing a voxel split by a plane. It assumes uniform distribution of rays and known costs for traversal steps and triangle intersections. The paper outlines a local greedy SAH heuristic, an approximation that simplifies the computation and has been found effective in practice. They compare the O(N log N) algorithm with the previously used O(N log^2 N) method, demonstrating a significant improvement in build times across different models. This section underscores the practicality and effectiveness of the algorithm in real-world scenarios.

### SAH-Approximate Parallel KD-tree Construction

Shevtsov et al. [1] focus on addressing the challenge of ray tracing dynamic scenes, where the geometry changes frequently. Traditionally, constructing high-quality KD-trees for such scenes has been computationally complex and difficult to parallelize. This paper contributes to this area by proposing a parallel initial spatial partitioning technique that allows for independent construction of KD-tree branches by multiple threads and implementing a method that rebuilds the KD-tree for the entire scene each frame, leveraging the parallelism of modern multi-core CPUs.

The paper delves into several key areas. SAH-approximation utilizing triangle Axis-Aligned Bounding Boxes (AABBs) as proxies during fast SAH estimation reduces the number of potential split candidates compared to traditional SAH-based KD-tree construction methods, such as the one presented in Wald and Havran’s paper. Furthermore, the Min-Max Binning Algorithm uses separate bins to track the start and end of each triangle’s AABB. This technique enables accurate counting of triangles for SAH estimation. Finally, parallelized construction using a hybrid parallelization scheme, alongside a thread-safe memory pool to efficiently manage memory, further speeds up building times.

The paper provides empirical results demonstrating the efficacy of their approach. They achieve interactive frame rates for high-resolution models of various complexities. The implementation details include the use of SIMD (Single Instruction, Multiple Data) instructions for vectorization, enhancing performance. They highlight potential future research areas, including more efficient memory bandwidth handling, on-demand construction of KD-tree nodes, and finer task management to accommodate a large number of threads.

## Approaches and Techniques

### Naïve KD-Tree

In KD-tree construction, the decision about where to place the splitting plane and when to stop the recursion process is crucial. The naïve approach, often referred to as "spatial median splitting," adopts a straightforward strategy for this purpose. The splitting plane is chosen based on the spatial median of the current voxel. This method rotates through the dimensions x, y, z in a round-robin fashion. The dimension p_k for the split is determined by the current subdivision depth. The position of the plane p_ξ is set to the midpoint of the voxel along the chosen dimension, thus splitting the space in half.

The recursion for subdividing the space is terminated when the number of triangles in a voxel falls below a specific threshold or when the subdivision depth exceeds a certain maximum depth.

#### Limitations in Ray-Tracing Applications

Naïve KD-trees often prove suboptimal for ray-tracing due to their inefficient spatial partitioning. This approach leads to unbalanced trees, especially in scenes with uneven geometry distribution, resulting in increased ray traversal costs. The simplistic splitting criteria, along with non-adaptive termination conditions, fail to account for the actual distribution of surfaces, causing either over-partitioning or under-partitioning.

### SAH-Based KD-Tree

#### Surface Area Heuristic

The Surface Area Heuristic (SAH) is a heuristic cost estimate used to evaluate the efficiency of different potential splits. It considers the probability of a ray hitting each side of a split voxel and the cost associated with traversing the voxel and intersecting triangles. The heuristic is based on the surface area of the child voxels and the number of triangles overlapping them. More simply, it is based on the ratio of the surface area whatever object is in a voxel, relative to the surface area of the voxel, plus the expected cost of the traversal of that voxel K_T. 

Instead of a simple spatial median, the SAH approach evaluates multiple potential split planes and selects the one that minimizes the estimated cost. This often leads to a more balanced and efficient tree, especially for scenes with unevenly distributed geometry. The termination criterion in SAH-based KD-trees is more dynamic and is based on comparing the cost of further subdivision against the cost of creating a leaf node. If splitting a voxel results in a higher cost than not splitting it (based on the SAH), the subdivision process is halted, and a leaf node is created.

In order to build an SAH-based KD-tree, we can just iterate over every triangle, and determine all of its split candidates, thus allowing for a naïve O(N^2) solution to be present.

#### O(N log^2 N) Implementation

The O(N log^2 N) algorithm for constructing KD-trees is an advancement over the naïve O(N^2) method, incorporating a more sophisticated approach to manage and evaluate potential split planes within each voxel. This method significantly reduces computational complexity by employing an incremental and sorted event-based process.

The algorithm categorizes each triangle into a set of "events" based on its interaction with potential split planes. These events include the starting point, ending point, and any intersection points of the triangle with the potential planes. Each event is tagged with information about the triangle it originates from and whether it is of type "start", "end", or "planar". The events are sorted in ascending order based on their position along the axis of consideration. This sorting is critical for the efficient computation of the number of triangles on either side of a potential split plane, contributing to the O(N log N) complexity in this step.

To determine where relative to the plane a triangle is placed, a hypothetical split plane "sweeps" across the voxel. The algorithm incrementally updates the count of triangles N_L, N_R, and N_P to the left, right, and in the plane of the split. The algorithm then evaluates the Surface Area Heuristic (SAH) cost function at each potential split position, and the split that yields the lowest SAH cost is chosen as the optimal plane for subdivision.

Once the optimal split plane is determined, the algorithm recursively applies the same process to the resulting sub-voxels. The complexity accumulates over the recursive steps, leading to the overall O(N log^2 N) complexity.

#### O(N log N) Implementation

The O(N log N) algorithm for KD-tree construction enhances the previous O(N log^2 N) approach by eliminating the O(N log N) sorting cost in each partitioning step. This is achieved by sorting the event list only once at the beginning and then maintaining the sort order during plane selection and partitioning, leading to an overall more efficient process.

The algorithm first creates a single event list for all dimensions, tagging each event with its corresponding dimension. This unified list includes events from all triangles, indicating where they start, end, or intersect with potential split planes. The entire event list is sorted once at the beginning based on the plane position and dimension, forming the basis for the subsequent steps. Similar to the O(N log^2 N) method, this algorithm incrementally updates triangle counts N_L, N_R, and N_P for each potential split plane. The process is streamlined across all dimensions, reducing the complexity of determining the best split plane.

After identifying the optimal split plane, the algorithm classifies triangles as being on the left, right, or both sides of the plane. It then generates two new event lists for the child nodes by splicing the original list and merging it with new events created by triangles intersecting the split plane. This step is crucial as it maintains the sorted order without the need for additional sorting.

By avoiding repetitive sorting and utilizing a sort-free approach, the algorithm achieves O(N log N) complexity, which is a significant improvement over the O(N log^2 N) method. This approach meets the theoretical lower bound for KD-tree construction, making it highly efficient for large-scale data. The method is particularly suitable for complex models where maintaining sort order and efficiently partitioning the data is crucial.

### SAH-Approximate Parallel KD-tree Construction

This method uses Axis-Aligned Bounding Boxes (AABBs) as proxies for triangles during SAH estimation, simplifying and speeding up the process. The SAH-approximation is adapted for both large and small objects, replacing the need to store object references at each bin with a simpler object counter. This approach, requiring just a single pass over the geometry as opposed to sorting, is more efficient.

The Conventional Binning Algorithm involves dividing a 1-D interval into equal-sized bins (a regular grid) and calculating the number of triangles in each bin. The method updates bins based on the triangle's interaction with potential split planes, allowing for fast, though imprecise, SAH approximation. Inspired by this, the min-max binning algorithm tracks where each triangle's AABB begins and ends in two separate bin sets. This approach eliminates dependency on the total number of bins, crucial for initial clustering. The algorithm is suitable for 2-D and 3-D binning and is more efficient than storing individual primitive lists in bins.

We can run SAH approximation using min-max bins, via a pass over the primitives for min-max binning and another pass for estimating SAH values. The algorithm computes the number of primitives on either side of a potential split and selects the position that minimizes the SAH value. An additional pass adjusts the final split plane position and distributes geometry among child nodes. The researchers found it beneficial to switch to exact SAH computation at deeper tree levels when the number of primitives equals or is less than the number of bins.

The approach presented in Shevtsov et al. implements an efficient memory allocation system using chunks linked into lists. Memory requests shift an end-pointer, with new chunks allocated only when necessary. This memory management matches the KD-tree's top-down and left-first construction nature. The construction uses two types of memory pools: one for nodes and leaves of the KD-tree and another for storing primitive indices at each recursive step. This method allows for efficient allocation and deallocation, fitting the pattern of KD-tree construction.

## Results

### SAH-Based KD-Tree

In this implementation of the KD-tree construction algorithms described in Wald's paper, I focused on two specific approaches: the O(N log^2 N) and the O(N log N) methods for SAH-based KD-tree construction. These implementations were tested on the Stanford bunny and the armadillo models to assess their performance in terms of KD-tree build times.

| Model                  | O(N log^2 N)   | O(N log N)     |
|------------------------|----------------|-----------------|
| Stanford bunny         | 9.8 sec        | 5.3 sec         |
| Armadillo (345,944 triangles) | 23.8 sec       | 13.1 sec        |

The results clearly demonstrate the superior efficiency of the O(N log N) algorithm over the O(N log^2 N) method. For both the Stanford bunny and the armadillo models, the build times were significantly lower with the O(N log N) approach. This is consistent with the theoretical expectations, as the O(N log N) algorithm has a lower computational complexity and is thus expected to perform better, especially as the number of triangles increases.

## Conclusion

This project focused on exploring and implementing advanced KD-tree construction techniques to enhance the efficiency of ray tracing in dynamic scenes. Two distinct methods from the literature were implemented: the O(N log^2 N) and O(N log N) algorithms for SAH-based KD-tree construction. These implementations were tested on the Stanford bunny and armadillo models, demonstrating the superior performance of the O(N log N) algorithm in terms of build times, consistent with theoretical expectations.

The O(N log^2 N) algorithm was overshadowed by the O(N log N) approach in terms of efficiency, with the latter's implementation aligning with the theoretical lower bound of KD-tree construction, making it an optimal choice for handling large-scale geometric data. The results indicate that for scenes with a high number of triangles, the O(N log N) algorithm significantly reduces construction time, thus making it a more suitable choice for real-time applications.

There was also an exploration into the SAH-approximate parallel KD-tree construction techniques proposed by Shevtsov et al., though the implementation of these techniques was not completed. Early exploration suggests that they hold promise for further reducing KD-tree construction times.

Future work could focus on completing the implementation of the parallel KD-tree construction method. Exploring hybrid approaches that combine the strengths of different algorithms could lead to more efficient KD-tree constructions for specific types of scenes or applications.

In conclusion, this research contributes to the field of ray tracing by implementing and analyzing efficient KD-tree construction algorithms, with a particular focus on improving performance for dynamic scenes. The findings from this study provide a foundation for future research aimed at further optimizing ray tracing techniques, particularly for interactive and real-time applications.

## References

1. Shevtsov, M., Soupikov, A., and Kapustin, A. (2007). Highly parallel fast KD‐tree construction for interactive ray tracing of dynamic scenes. Computer Graphics Forum, 26(3), 395–404. [DOI](https://doi.org/10.1111/j.1467-8659.2007.01062.x)
   
2. Wald, I., and Havran, V. (2006). On building fast kd-Trees for Ray Tracing, and on doing that in O(N log N). IEEE Xplore. [DOI](https://doi.org/10.1109/rt.2006.280216)

[*back to top*](#)

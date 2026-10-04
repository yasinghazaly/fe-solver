# Theory

## 1. Element Stiffness from the Principle of Virtual Work

The element stiffness matrix follows from the principle of virtual work. The principle states that, for a body in equilibrium, the work done by the external forces through a virtual displacement is equal to the work done by the internal stresses through the corresponding virtual strain. For an element of thickness $`t`$ this is expressed in equation 1:

```math
\delta u^{T} F = \iint \delta\varepsilon^{T} \sigma \, t \, dx \, dy \qquad (1)
```

Where $`u`$ is the nodal displacement vector, $`F`$ is the nodal force vector, $`\varepsilon`$ is the strain and $`\sigma`$ is the stress. The strain is related to the nodal displacements through the strain-displacement matrix, $`\varepsilon = Bu`$, and the stress is related to the strain through the constitutive material matrix, $`\sigma = D\varepsilon`$. Substituting both into equation 1, and noting that the virtual displacement is arbitrary, results in $`Ku = F`$, where the stiffness matrix is:

```math
K = \iint B^{T} D B \, t \, dx \, dy \qquad (2)
```

## 2. 8-Node Quadrilateral Element Method

An 8-node element is a second order quadrilateral element defined by eight nodes, each with two degrees of freedom (DOF): translation in the x-direction and translation in the y-direction [1]. Therefore, the total number of DOF is 16.

2D plane stress is assumed in this analysis. This is appropriate when the thickness is small relative to the in-plane dimensions. Furthermore, the out of plane stress is not being considered.

An 8-node element is a type of isoparametric element, where the geometry and the deformation of the elements are described using the same set of mathematical shape functions.

The shape functions are defined in natural coordinates $`(\xi, \eta)`$, and then mapped to the real coordinate system $`(x, y)`$ through the utilisation of a Jacobian transformation. In the natural coordinate system the element is a square with its corners at $`\xi = \pm 1`$ and $`\eta = \pm 1`$. The corner nodes are numbered 1 to 4 anticlockwise and the midside nodes are numbered 5 to 8. Note that the code numbers the nodes from 0. Nine Gauss points will be later utilised to formulate the stiffness matrix through numerical integration.

## 3. Shape Functions

The shape functions for all 8 nodes are defined below:

```math
N_1(\xi, \eta) = -\frac{1}{4}(1 - \xi)(1 - \eta)(1 + \xi + \eta) \qquad (3.1)
```

```math
N_2(\xi, \eta) = -\frac{1}{4}(1 + \xi)(1 - \eta)(1 - \xi + \eta) \qquad (3.2)
```

```math
N_3(\xi, \eta) = -\frac{1}{4}(1 + \xi)(1 + \eta)(1 - \xi - \eta) \qquad (3.3)
```

```math
N_4(\xi, \eta) = -\frac{1}{4}(1 - \xi)(1 + \eta)(1 + \xi - \eta) \qquad (3.4)
```

```math
N_5(\xi, \eta) = \frac{1}{2}(1 - \xi^2)(1 - \eta) \qquad (3.5)
```

```math
N_6(\xi, \eta) = \frac{1}{2}(1 + \xi)(1 - \eta^2) \qquad (3.6)
```

```math
N_7(\xi, \eta) = \frac{1}{2}(1 - \xi^2)(1 + \eta) \qquad (3.7)
```

```math
N_8(\xi, \eta) = \frac{1}{2}(1 - \xi)(1 - \eta^2) \qquad (3.8)
```

## 4. Formulating the Strain-Displacement Matrix (B)

The first step in the analysis is the formulation of the strain-displacement matrix, B. The B matrix relates the nodal displacements to strains within the element. In real coordinates the B matrix is expressed in equation 4:

```math
B =
\begin{bmatrix}
\dfrac{\partial N_1}{\partial x} & 0 & \cdots & \dfrac{\partial N_8}{\partial x} & 0 \\
0 & \dfrac{\partial N_1}{\partial y} & \cdots & 0 & \dfrac{\partial N_8}{\partial y} \\
\dfrac{\partial N_1}{\partial y} & \dfrac{\partial N_1}{\partial x} & \cdots & \dfrac{\partial N_8}{\partial y} & \dfrac{\partial N_8}{\partial x}
\end{bmatrix}
\qquad (4)
```

Since the shape functions are in terms of natural coordinates, coordinate mapping is necessary through the Jacobian matrix as shown in equation 5:

```math
\begin{bmatrix}
\dfrac{\partial N_i}{\partial x} \\
\dfrac{\partial N_i}{\partial y}
\end{bmatrix}
=
\begin{bmatrix}
\sum\limits_{j=1}^{8} \dfrac{\partial N_j}{\partial \xi} x_j & \sum\limits_{j=1}^{8} \dfrac{\partial N_j}{\partial \xi} y_j \\
\sum\limits_{j=1}^{8} \dfrac{\partial N_j}{\partial \eta} x_j & \sum\limits_{j=1}^{8} \dfrac{\partial N_j}{\partial \eta} y_j
\end{bmatrix}^{-1}
\begin{bmatrix}
\dfrac{\partial N_i}{\partial \xi} \\
\dfrac{\partial N_i}{\partial \eta}
\end{bmatrix}
\qquad (5)
```

The rows of the Jacobian correspond to $`\xi`$ and $`\eta`$, and the columns correspond to $`x`$ and $`y`$.

## 5. Stiffness Matrix Calculation Using Gauss Integration

Evaluating the stiffness matrix through analytical integration is difficult. Thus, it is necessary to use numerical integration for the evaluation of integrals of isoparametric elements.

The standard method is the use of Gauss Numerical Integration because a minimal number of sample points (Gauss points) are required to achieve an accurate result. The integrand is multiplied by the determinant of the Jacobian to account for the coordinate transformation. The integral is expressed in equation 6:

```math
K = \iint B^{T} D B \, t \, dx \, dy = \sum_{i}^{n} \sum_{j}^{n} B^{T} D B \, t \, \det[J] \, w_i w_j \qquad (6)
```

Where $`w_i w_j`$ are the Gauss weights (5/9, 8/9, 5/9). $`D`$ is the constitutive material matrix, $`t`$ is the thickness of the element, and $`\det[J]`$ is the determinant of the Jacobian. $`n`$ is the number of Gauss points in each direction, equal to 3. The Gauss points are located at $`-\sqrt{3/5}`$, $`0`$ and $`\sqrt{3/5}`$ in each of the natural coordinates, which results in nine Gauss points in total.

## 6. Rigid-Body Check of the Stiffness Matrix

A rigid-body motion produces no strain within the element. Thus, the product of the stiffness matrix and a rigid-body displacement vector must be zero. In 2D there are three rigid-body motions: translation in the x-direction, translation in the y-direction and rotation in the plane. As a result, a correct element stiffness matrix has exactly three zero eigenvalues before any boundary conditions are applied. This check is sensitive to the Jacobian. If the Jacobian in equation 5 is formulated with the wrong convention, the stiffness matrix for a general element shape does not have three zero eigenvalues.

## 7. CST Method

The CST element is characterised by having linear shape functions and a constant strain through the element. As a result, the stress is also constant. The three nodes are numbered anticlockwise and are located at the natural coordinates (0, 0), (1, 0) and (0, 1). The shape functions are:

```math
N_1 = 1 - \xi - \eta, \qquad N_2 = \xi, \qquad N_3 = \eta \qquad (7)
```

The derivatives of the shape functions are constant, so the Jacobian in equation 5 becomes:

```math
J =
\begin{bmatrix}
x_2 - x_1 & y_2 - y_1 \\
x_3 - x_1 & y_3 - y_1
\end{bmatrix},
\qquad \det[J] = 2A
\qquad (8)
```

Evaluating equation 5 for each node results in the B matrix:

```math
B = \frac{1}{2A}
\begin{bmatrix}
y_2 - y_3 & 0 & y_3 - y_1 & 0 & y_1 - y_2 & 0 \\
0 & x_3 - x_2 & 0 & x_1 - x_3 & 0 & x_2 - x_1 \\
x_3 - x_2 & y_2 - y_3 & x_1 - x_3 & y_3 - y_1 & x_2 - x_1 & y_1 - y_2
\end{bmatrix}
\qquad (9)
```

The B matrix does not vary within the element. Due to its characteristics the stiffness matrix is evaluated in a simplified form compared to the 8 node FE element:

```math
K = B^{T} D B \, t \, A \qquad (10)
```

Where $`A`$ is the area of the triangle. The integrand is constant, so a single evaluation point is sufficient and no further numerical integration is required.

An 8-node element contains quadratic shape functions; this allows it to accurately capture the curvature and model a varying stress field within the element [2]. On the contrary, a CST contains linear shape functions. This will cause stress and strain to remain constant within the element. Additionally, a well-documented drawback of the CST is its prediction of overestimated shear stress within the element [3].

## 8. Assembly of the Global Stiffness Matrix

A mesh contains more than one element, so the element stiffness matrices must be assembled into a single global stiffness matrix. The global DOF are ordered as $`[u_1, v_1, u_2, v_2, \dots]`$, so each node owns two consecutive DOF. Each entry of an element stiffness matrix is added to the global matrix at the row and column of the corresponding global DOF. Thus, where a node is shared between elements, the contributions of those elements are summed.

## 9. Consistent Nodal Forces for an Edge Traction

A load distributed along an element edge must be converted to forces at the nodes. An edge of the 8-node element contains three nodes: an end node, a middle node and a second end node. Three quadratic shape functions are defined along the edge in terms of a natural coordinate $`s`$, which runs from $`-1`$ at the first end node to $`+1`$ at the second:

```math
\hat{N}_1(s) = \frac{s(s - 1)}{2}, \qquad \hat{N}_2(s) = 1 - s^2, \qquad \hat{N}_3(s) = \frac{s(s + 1)}{2} \qquad (11)
```

The force at each node is obtained by weighting the traction with the shape function of that node and integrating along the edge. Thus, the nodal forces do the same work as the distributed load. The edge is assumed to be straight with the middle node at its midpoint, so the coordinate transformation reduces to a factor of $`l/2`$. The integral is evaluated using three Gauss points, as expressed in equation 12:

```math
F_i = \int_{-1}^{1} \hat{N}_i(s) \, q \, t \, \frac{l}{2} \, ds = \sum_{k}^{3} \hat{N}_i(s_k) \, q(x_k, y_k) \, t \, \frac{l}{2} \, w_k \qquad (12)
```

Where $`q = (q_x, q_y)`$ is the traction, $`l`$ is the length of the edge and $`w_k`$ are the Gauss weights. $`(x_k, y_k)`$ is the position of Gauss point $`k`$, which is obtained by interpolating the coordinates of the edge nodes with the same shape functions. Three Gauss points integrate the load exactly for tractions up to cubic variation along the edge. For a uniform traction, the load is shared 1/6, 4/6 and 1/6 between the end, middle and end nodes, not equally.

## 10. Boundary Conditions and Solution

After the stiffness matrix is assembled, boundary conditions must be applied. The DOF are partitioned into free DOF, denoted by the subscript $`f`$, and prescribed DOF, denoted by the subscript $`p`$:

```math
\begin{bmatrix}
K_{ff} & K_{fp} \\
K_{pf} & K_{pp}
\end{bmatrix}
\begin{bmatrix}
u_f \\
u_p
\end{bmatrix}
=
\begin{bmatrix}
F_f \\
F_p
\end{bmatrix}
\qquad (13)
```

The prescribed displacements $`u_p`$ are known. Thus, the first row of equation 13 is rearranged to obtain the free displacements:

```math
K_{ff} \, u_f = F_f - K_{fp} \, u_p \qquad (14)
```

This allows non-zero displacements to be prescribed. If the prescribed nodes are fully constrained, so no displacement in x or y, the last term in equation 14 is zero. The nodal displacements are then obtained by evaluating equation 15:

```math
u_f = K_{ff}^{-1} \left( F_f - K_{fp} \, u_p \right) \qquad (15)
```

## 11. Computation of the Stress and Strain

The nodal displacement vector is back substituted to obtain the strain at each Gauss point:

```math
\varepsilon = Bu \qquad (16)
```

The corresponding stresses are obtained using Hooke's Law:

```math
\sigma = D\varepsilon = \frac{E}{1 - \nu^2}
\begin{bmatrix}
1 & \nu & 0 \\
\nu & 1 & 0 \\
0 & 0 & \dfrac{1 - \nu}{2}
\end{bmatrix}
\varepsilon
\qquad (17)
```

## 12. Extrapolation of the Stresses to the Nodes

The stresses computed in equation 17 are obtained at the location of the Gauss points within the element. However, for post-processing and visualisation it is more practical to obtain stress at the nodal positions.

The nodal stresses are estimated using an extrapolation technique [4]. The underlying assumption behind the method is the nodal stresses are related to the stresses at the Gauss points via the shape functions. The relationship can be observed in equation 18:

```math
\sigma_{nodal} = (L^{T} L)^{-1} L^{T} \sigma_{gp} \qquad (18)
```

Where $`L`$ is a 9x8 matrix formed by evaluating the element's shape functions at the nine Gauss points. This procedure is only valid when the number of Gauss points is equal to or greater than the number of nodes. In the code this 9x8 matrix is named `E`, and the name `L` is given to the product $`(L^{T} L)^{-1} L^{T}`$. For the CST there is a single Gauss point, so the constant stress is assigned unchanged to all three nodes. Where a node is shared between elements, the nodal stresses obtained from those elements are averaged.

## 13. Von Mises Stress

The Equivalent von Mises stress is also calculated for each node. First the principal stresses are found using equation 19:

```math
\sigma_{1,2} = \frac{\sigma_x + \sigma_y}{2} \pm \sqrt{\frac{(\sigma_x - \sigma_y)^2}{4} + \tau_{xy}^2} \qquad (19)
```

Next equation 20 is evaluated to obtain the equivalent von Mises Stress:

```math
\sigma_e = \sqrt{\sigma_1^2 - \sigma_1 \sigma_2 + \sigma_2^2} \qquad (20)
```

Substituting equation 19 into equation 20 yields the equivalent stress directly in terms of the stress components, which is the form utilised in the package:

```math
\sigma_e = \sqrt{\sigma_x^2 - \sigma_x \sigma_y + \sigma_y^2 + 3\tau_{xy}^2} \qquad (21)
```

## 14. Error Norms and Convergence Rates

To measure the accuracy of the solution, the finite element displacement $`u_h`$ is compared against an exact solution $`u`$. Two error norms are utilised. The L2 norm measures the error in displacement:

```math
\lVert e \rVert_{L2} = \sqrt{\iint \lvert u - u_h \rvert^2 \, dx \, dy} \qquad (22)
```

The energy norm measures the error in strain:

```math
\lVert e \rVert_{E} = \sqrt{\iint (\varepsilon - \varepsilon_h)^{T} D \, (\varepsilon - \varepsilon_h) \, dx \, dy} \qquad (23)
```

Where $`\varepsilon`$ is the exact strain and $`\varepsilon_h`$ is the finite element strain. Both errors are absolute, and no thickness factor is included. The integrals are evaluated using Gauss integration with a higher-order rule than the one used for the stiffness matrix, with 5 Gauss points in each direction by default.

For quadratic elements such as the 8-node element, the L2 error is expected to fall as $`h^3`$ and the energy error as $`h^2`$, where $`h`$ is the element size. The measured rate between two consecutive meshes is obtained from equation 24:

```math
\text{rate} = \frac{\log(e_1 / e_2)}{\log(h_1 / h_2)} \qquad (24)
```

Where $`e_1`$ and $`e_2`$ are the errors on the two meshes and $`h_1`$ and $`h_2`$ are the corresponding element sizes.

## References

1. Rajendran, S. and Liew, K.M. (2003). A novel unsymmetric 8-node plane element immune to mesh distortion under a quadratic displacement field. International Journal for Numerical Methods in Engineering, 58(11), pp.1713–1748. doi: https://doi.org/10.1002/nme.836.
2. Mahran, M., ELsabbagh, A. and Negm, H. (2017). A comparison between different finite elements for elastic and aero-elastic analyses. Journal of Advanced Research, 8(6), pp.635–648. doi: https://doi.org/10.1016/j.jare.2017.06.009.
3. Muftu, S. (2022). Rectangular and triangular elements for two-dimensional elastic solids. In: Finite Element Method: Physics and Solution Methods. Elsevier, pp.257–291.
4. Durand, R. and Farias, M.M. (2013). A local extrapolation method for finite elements. Advances in Engineering Software, 67, pp.1–9. doi: https://doi.org/10.1016/j.advengsoft.2013.07.002.
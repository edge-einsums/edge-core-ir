# Max-flow Full-Edge Einsum

Source of truth: `Push_Relabel_Max_Flow.ipynb` in the EDGE tutorial zoo. See
`CHANGES.md` for what moved relative to the previous version of this artifact.

Semantic assumption used throughout: **a stored value equal to a tensor's
`empty` value is treated as absent.** This is what lets a saturated residual
edge (`R = 0`) drop out of an intersect merge without an explicit positivity
guard.

The tutorial's Tensors block declares only the nine named tensors. The
cascade intermediates need empty values too, and the choice is not free — see
the second block below and `CHANGES.md` §11.

```tex
\begin{align}
▹ \text{Tensors} \\
G^{U \equiv |V|, V \equiv |V|} &\to \text{Integer},\ \text{empty} = 0 \\
C^{U \equiv |V|, V \equiv |V|} &\to \text{Integer},\ \text{empty} = 0 \\
F^{I, U \equiv |V|, V \equiv |V|} &\to \text{Integer},\ \text{empty} = 0 \\
R^{I, U \equiv |V|, V \equiv |V|} &\to \text{Integer},\ \text{empty} = 0 \\
E^{I, U \equiv |V|} &\to \text{Integer},\ \text{empty} = 0 \\
D^{I, U \equiv |V|} &\to \text{Integer},\ \text{empty} = +\infty \\
Act^{I, U \equiv |V|} &\to \text{Boolean},\ \text{empty} = \text{false} \\
S^{U \equiv |V|} &\to \text{Boolean},\ \text{empty} = \text{false} \\
T^{U \equiv |V|} &\to \text{Boolean},\ \text{empty} = \text{false} \\
\\
▹ \text{Cascade intermediates} \\
FS^{I, U, V},\ Delta^{I, U, V},\ InPush^{I, U},\ OutPush^{I, U}
&\to \text{Integer},\ \text{empty} = 0 \\
NeiLbl^{I, U, V},\ MinNeiLbl^{I, U},\ NewD^{I, U}
&\to \text{Integer},\ \text{empty} = +\infty \\
NST^{U},\ ActR^{I, U, V},\ Lbl^{I, U, V},\ Adm^{I, U, V},
PushCand^{I, U, V},\ HasAdm^{I, U},\ Rel^{I, U}
&\to \text{Boolean},\ \text{empty} = \text{false} \\
\\
▹ \text{Initializations} \\
F_{0,u,v} &= 0 \\
E_{0,u} &= 0 \\
D_{1,u} &= 0 \\
D_{1,u:u=s} &= |V| \\
S_{u:u=s} &= true \\
T_{u:u=t} &= true \\
\\
▹ \text{Extended Einsums} \\
FS_{1,u,v} &= S_u \cdot C_{u,v} :: \bigwedge *(\cap)
&& \text{...preflow to every edge out of the source: } f(s,v)=c(s,v) \\
F_{1,u,v} &= FS_{1,u,v} \cdot FS_{1,v,u} :: \bigwedge -(\cup)
&& \text{...record the negative flow on the reverse edge: } f(v,s)=-c(s,v) \\
E_{1,u} &= F_{1,v,u} :: \bigvee +(\cup)
&& \text{...excess is net inflow; outflow is already stored as negative entries} \\
R_{1,u,v} &=
\begin{cases}
0 & u = s \\
C_{v,u} & v = s\ \wedge\ u \neq s \\
C_{u,v} & otherwise
\end{cases}
&& \text{...residual from the source is spent; the reverse edge gains capacity} \\
\\
NST_u &= \neg S_u \cdot \neg T_u :: \bigwedge AND(\cap)
&& \text{...not a source/sink} \\
Act_{i,u} &= NST_u \cdot^1 (E_{i,u} \cdot^2 0)_{i,u}
:: \bigwedge^1 \leftarrow(\cap)\ \bigwedge^2 >(\cap)
&& \text{...active vertices: internal vertices with excess flow} \\
ActR_{i,u,v} &= Act_{i,u} \cdot R_{i,u,v} :: \bigwedge \leftarrow(\cap)
&& \text{...residual edges leaving an active vertex; saturated edges are empty, so } \cap \text{ skips them} \\
Lbl_{i,u,v} &= D_{i,u} \cdot (D_{i,v} + 1)_{i,v} :: \bigwedge \equiv(\cap)
&& \text{...check height rule } D(u)=D(v)+1 \\
Adm_{i,u,v} &= ActR_{i,u,v} \cdot Lbl_{i,u,v} :: \bigwedge AND(\cap)
&& \text{...active residual edges satisfying the height rule} \\
\\
\textbf{Push Step} \\
PushCand_{i,u,v^*} &= Adm_{i,u,v} \lll_{v^*} \mathbf{1}(\text{pick-admissible-edge})
&& \text{...choose one admissible neighbor for each pushing vertex} \\
delta_{i,u,v} &= (E_{i,u} \cdot^1 R_{i,u,v})_{i,u,v} \cdot^2 PushCand_{i,u,v}
:: \bigwedge^1 min(\cap)\ \bigwedge^2 \leftarrow(\cap)
&& \text{...push } \min(\text{excess},\text{residual}) \text{ on selected edges} \\
F_{i+1,u,v} &= (F_{i,u,v} \cdot^1 delta_{i,u,v})_{i,u,v} \cdot^2 delta_{i,v,u}
:: \bigwedge^1 +(\cup)\ \bigwedge^2 -(\cup)
&& \text{...adjust flow on selected edges} \\
InPush_{i,u} &= delta_{i,v,u} :: \bigvee +(\cup)
&& \text{...total new flow pushed into each vertex} \\
OutPush_{i,u} &= delta_{i,u,v} :: \bigvee +(\cup)
&& \text{...total flow pushed out of each vertex} \\
E_{i+1,u} &= (E_{i,u} \cdot^1 InPush_{i,u})_{i,u} \cdot^2 OutPush_{i,u}
:: \bigwedge^1 +(\cup)\ \bigwedge^2 -(\cup)
&& \text{...add received/subtract sent flow to excess} \\
R_{i+1,u,v} &= (R_{i,u,v} \cdot^1 delta_{i,u,v})_{i,u,v} \cdot^2 delta_{i,v,u}
:: \bigwedge^1 -(\cup)\ \bigwedge^2 +(\cup)
&& \text{...subtract used residual capacity and add reverse residual capacity} \\
\\
\textbf{Relabel Step} \\
Adm_{i+1,u,v} &= R_{i+1,u,v} \cdot Lbl_{i,u,v} :: \bigwedge \rightarrow(\cap)
&& \text{...admissible edges after the push update} \\
HasAdm_{i+1,u} &= Adm_{i+1,u,v} :: \bigvee OR(\cup)
&& \text{...whether each vertex can still push after the update} \\
Act_{i+1,u} &= NST_u \cdot^1 (E_{i+1,u} \cdot^2 0)_{i+1,u}
:: \bigwedge^1 \leftarrow(\cap)\ \bigwedge^2 >(\cap)
&& \text{...active vertices, recomputed from the excess the push produced} \\
Rel_{i+1,u} &= Act_{i+1,u} \cdot \neg HasAdm_{i+1,u} :: \bigwedge AND(\cap)
&& \text{...active vertices with no admissible edge will relabel} \\
NeiLbl_{i,u,v} &= (R_{i+1,u,v} \cdot^1 Rel_{i+1,u})_{i,u,v} \cdot^2 D_{i,v}
:: \bigwedge^1 \leftarrow(\cap)\ \bigwedge^2 \rightarrow(\cap)
&& \text{...get labels of residual neighbors for relabel vertices} \\
MinNeiLbl_{i,u} &= NeiLbl_{i,u,v} :: \bigvee min(\cup)
&& \text{...minimum label among residual neighbors} \\
NewD_{i,u} &= (MinNeiLbl_{i,u} \cdot 1)_{i,u} :: \bigwedge +(\cap)
&& \text{...add 1 to the label of the lowest residual neighbor} \\
D_{i+1,u} &= D_{i,u} \cdot NewD_{i,u} :: \bigwedge <\!\!<(\cup)
&& \text{...apply relabel; vertices that did not relabel keep their height} \\
\\
◇ &: \lVert Act_{i+1} \rVert \equiv 0
&& \text{...stop when no internal vertex has excess}
\end{align}
```

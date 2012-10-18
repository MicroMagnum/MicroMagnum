========
 Models 
========

What is actually computed in MagNum?

Landau Lifshitz Gilbert equation
================================

:math:`\frac{d\mathbf{M}}{dt} = - \frac{\gamma}{1+\alpha^2}\mathbf{M}\times\mathbf{H}_\mathsf{eff} + \frac{\gamma\alpha}{M_\mathsf{s}(1+\alpha^2)}\mathbf{M}\times(\mathbf{M}\times\mathbf{H}_\mathsf{eff})`

Effective fields
----------------

:math:`\mathbf{H}_\mathsf{eff} = \mathbf{H}_\mathsf{demag} + \mathbf{H}_\mathsf{exch} + \mathbf{H}_\mathsf{aniso} + \mathbf{H}_\mathsf{ext} + \mathsf{other\ terms}`

Exchange field
""""""""""""""

:math:`\mathbf{H}_{\mathsf{exch}}=\frac{2A}{\mu_0M^2_s}\Delta\mathbf{M}`

Demagnetization field
"""""""""""""""""""""

:math:`\mathbf{H_\mathsf{demag}}(\mathbf r) = \int \mathbf N(\mathbf r-\mathbf r') \cdot \mathbf M(\mathbf r') dV'`


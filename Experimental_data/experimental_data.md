**Summary of the extraction method**

Both PDFs have vector figures, so I read the data from the PDF drawing commands, not from pixels, using PyMuPDF (`page.get_drawings()`).

1. **Axis calibration.** I located each panel's frame, matched the tick marks to their numeric labels, and fitted a linear or log mapping from PDF coordinates to data units. Fit residuals were ≤1e-3 of axis units.
2. **Marker identification.** I grouped markers by (fill, color, shape), so each series is separated automatically. The centers of the filled or open circles, squares and triangles are the data points.
3. **Series labeling.** I matched each series to the legend by text proximity (CMS) or by the legend marker path (STAR).
4. **Uncertainties.**
   - Statistical error bars come from the straight line segments through each marker.
   - Systematic uncertainties come from the semi-transparent boxes. For CMS, the box edges are also the pT bin edges.
   - Where the stat bar is smaller than the marker, I report an upper bound (`hidden_stat_max`) instead of a value.
5. **Model curves.** Solid and dashed paths were extracted as polylines.
6. **Validation.**
   - The dashed y = 0 line falls at |y| < 0.002 in 31 panels.
   - STAR Fig. 1b / Fig. 1c reproduces the v₀ values of Fig. 1a to within 0.1%.
   - CMS Fig. 6 matches Fig. 1 × v₀ to within 1%, and Fig. 5 matches Fig. 4 to within 0.1%.

**Limitations:** stat errors are only recoverable for 65 of 978 STAR points and 135 of 2508 CMS points. The CMS Fig. 4 pp, Ntrk 10–20 panel shows only 21 of 29 points.

----

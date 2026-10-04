# Three.js vendoring record

- Package: `three`
- Version: `0.186.0` (`r186`)
- Source archive: `https://registry.npmjs.org/three/-/three-0.186.0.tgz`
- Archive SHA-256: `61eeff9d7616005c9a481c796f52287d81fbbbc0d55eaca5565322924252c1aa`
- License: MIT; retained in `LICENSE`
- Vendored files: the browser ES-module entrypoint, its `three.core.js`
  dependency, and the official `OrbitControls` and `TrackballControls` add-ons
  from the same release.

The controller serves these files locally. The viewer does not contact a CDN
at runtime. The npm package declares no runtime dependencies or install-time
scripts; only these reviewed browser modules are included.

Security boundary: Three.js receives only scene data already returned by the
Buildvision3D controller and renders it through WebGL. It has no direct access
to unrelated local files. Keep point-count limits in the controller to bound
browser memory and GPU consumption.

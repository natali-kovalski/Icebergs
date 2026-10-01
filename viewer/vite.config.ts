import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { viteStaticCopy } from "vite-plugin-static-copy";

// Cesium loads workers, widgets CSS and assets at runtime from CESIUM_BASE_URL,
// so they are copied next to the app instead of being bundled.
const cesiumSource = "node_modules/cesium/Build/Cesium";
const cesiumBaseUrl = "cesium";

export default defineConfig({
  base: "./",
  build: { chunkSizeWarningLimit: 6000 }, // Cesium alone is ~4 MB
  define: { CESIUM_BASE_URL: JSON.stringify(cesiumBaseUrl) },
  plugins: [
    vue(),
    viteStaticCopy({
      targets: ["ThirdParty", "Workers", "Assets", "Widgets"].map((dir) => ({
        src: `${cesiumSource}/${dir}`,
        dest: cesiumBaseUrl,
        rename: { stripBase: 4 }, // drop node_modules/cesium/Build/Cesium
      })),
    }),
  ],
});

import { createApp } from "vue";
import "cesium/Build/Cesium/Widgets/widgets.css";
import "./style.css";
import App from "./App.vue";

(window as unknown as { CESIUM_BASE_URL: string }).CESIUM_BASE_URL = CESIUM_BASE_URL;

createApp(App).mount("#app");

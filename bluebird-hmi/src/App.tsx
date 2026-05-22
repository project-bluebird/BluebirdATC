import { BrowserRouter, Routes, Route } from "react-router-dom";
import RadarPage from "pages/RadarPage";

function App() {
  const basename = import.meta.env.PROD ? "/hmi" : "/";
  return (
    <BrowserRouter basename={basename}>
      <Routes>
        <Route path="/" element={<RadarPage />} />
      </Routes>
    </BrowserRouter>
  );
}
export default App;

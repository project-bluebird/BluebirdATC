export function appendText(
  group,
  x,
  y,
  text,
  fill,
  fontSize = "15px",
  fontFamily = "sans-serif",
) {
  group
    .append("text")
    .attr("x", x)
    .attr("y", y)
    .attr("font-family", fontFamily)
    .attr("font-size", fontSize)
    .attr("fill", fill)
    .text(text);
}

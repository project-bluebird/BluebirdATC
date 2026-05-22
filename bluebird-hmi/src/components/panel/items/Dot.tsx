export default function Dot({ backgroundColor }: { backgroundColor: string }) {
  return (
    <span
      style={{
        height: "10px",
        width: "10px",
        backgroundColor,
        borderRadius: "50%",
        display: "inline-block",
      }}
    />
  );
}

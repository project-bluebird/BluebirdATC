// Truncate a string
export default function truncateString(
  string: string,
  position: number,
): string {
  return string.length > position
    ? string.substring(0, position - 5) + "..."
    : string;
}

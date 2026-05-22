// Compare two sets, given as arrays of strings
export function compareArraysAsSets(
  array1: string[],
  array2: string[],
): boolean {
  // Cast to set to remove duplicates
  const set1 = new Set(array1);
  const set2 = new Set(array2);

  // Compare the size of the two sets
  if (set1.size !== set2.size) {
    return false;
  }

  // Check if every element in set1 is present in set2
  return Array.from(set1).every((element) => set2.has(element));
}

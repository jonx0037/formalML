/** A topic's `videoId` frontmatter: one ID, or one per part in part order. */
export type VideoIdField = string | string[] | null | undefined;

/** Normalize a topic's `videoId` frontmatter into an ordered list of YouTube IDs. */
export function toVideoIds(videoId: VideoIdField): string[] {
  const ids = Array.isArray(videoId) ? videoId : [videoId];
  return ids.filter((id): id is string => Boolean(id?.trim()));
}

import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const topics = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/topics' }),
  schema: z.object({
    title: z.string(),
    subtitle: z.string().optional(),
    status: z.enum(['draft', 'review', 'published']),
    difficulty: z.enum(['foundational', 'intermediate', 'advanced']),
    prerequisites: z.array(z.string()).default([]),
    tags: z.array(z.string()),
    domain: z.enum([
      'topology',
      'geometry',
      'probability',
      'optimization',
      'linear-algebra',
      'information-theory',
      'graph-theory',
      'category-theory',
    ]),
    videoId: z.string().nullable().default(null),
    notebookPath: z.string().nullable().default(null),
    githubUrl: z.string().url().nullable().default(null),
    datePublished: z.coerce.date().optional(),
    dateUpdated: z.coerce.date().optional(),
    estimatedReadTime: z.number().optional(),
    abstract: z.string(),
    connections: z
      .array(
        z.object({
          topic: z.string(),
          relationship: z.string(),
        }),
      )
      .default([]),
    references: z
      .array(
        z.object({
          type: z.enum(['paper', 'book', 'course', 'blog', 'video']),
          title: z.string(),
          authors: z.string().optional(),
          year: z.number().optional(),
          url: z.string().url().optional(),
          note: z.string().optional(),
        }),
      )
      .default([]),
  }),
});

export const collections = { topics };

import dotenv from 'dotenv';
dotenv.config();

const { buildAnthropicLanguageModel, generateText } = await import('./dist/agent/llm.js');

console.log('Token set:', !!process.env.ANTHROPIC_AUTH_TOKEN);
console.log('Base URL:', process.env.ANTHROPIC_BASE_URL);
console.log('Model:', process.env.ANTHROPIC_MODEL);

const model = buildAnthropicLanguageModel({
  model: process.env.ANTHROPIC_MODEL || 'MiniMax-M2.7',
  authToken: process.env.ANTHROPIC_AUTH_TOKEN,
  baseURL: process.env.ANTHROPIC_BASE_URL,
});

console.log('Provider:', model.provider);

const timeout = setTimeout(() => {
  console.error('TIMEOUT after 20s');
  process.exit(1);
}, 20000);

try {
  console.log('Sending request...');
  const result = await generateText({
    model,
    messages: [
      { role: 'system', content: 'You are a helpful assistant. Reply briefly.' },
      { role: 'user', content: 'Say hello in one sentence.' },
    ],
  });
  clearTimeout(timeout);
  console.log('generateText OK:', result.text.slice(0, 200));

  // Test generateObject
  const { generateObject } = await import('./dist/agent/llm.js');
  const { z } = await import('zod');
  const schema = z.object({ greeting: z.string(), language: z.string() });
  const obj = await generateObject({
    model,
    messages: [
      { role: 'user', content: 'Return a JSON object with a greeting in French.' },
    ],
    schema,
  });
  console.log('generateObject OK:', JSON.stringify(obj.object));
  console.log('SUCCESS');
} catch (e) {
  clearTimeout(timeout);
  console.error('ERROR:', e.message);
  if (e.status) console.error('Status:', e.status);
  if (e.error) console.error('Body:', JSON.stringify(e.error).slice(0, 500));
}
